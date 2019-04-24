import numpy as np
import matplotlib.pyplot as plt
import timeit


# All Markov kernels are detailed in theses references:
# - Geometric MCMC for Infinite-Dimensional Inverse Problems
# - Dimension-independent likelihood-informed MCMC


class MC():
    # Monte-Carlo algorithms for inverse problems

    def __init__(self, negll, negll_geom=None, negll_hess=None,
                 lmax=2, Dmax=2, Dmin=1e-8):

        # Negative log-likelihood and its derivatives
        self.negll = negll
        self.negll_geom = negll_geom
        self.negll_hess = negll_hess

        # Non gaussian prior (truncation parameters)
        self.lmax = lmax
        self.Dmax = Dmax
        self.Dmin = Dmin
        stdl = np.sqrt(self.lmax**2/12)
        stdd = np.sqrt((self.Dmax-self.Dmin)**2/12)
        self.std = np.hstack((stdl, stdd))
        self.mean = np.hstack((self.lmax/2, (self.Dmax-self.Dmin)/2))

    def MCMC(self, alg, u0, h, n):
        # Run the MCMC algorithm with Metropolis-Hastings

        print('Running MCMC MH sampling')
        print('Proposal kernel: ', alg)
        print('# of total samples: ', n)

        # Initialization
        proba = 0.0
        results = np.zeros((n+1, u0.size+1))
        self.ui = u0
        self.u_mean = np.zeros(u0.shape)
        self.u_mean[0:2] = self.mean
        self.nll = self.negll(self.ui)
        results[0, :] = np.hstack((self.nll, self.ui))

        # Modified proposals for uniform distribution on 2 first components
        proposal = self.inf_mMALA_proposal_uni
        self.local = self.inf_HMALA_uni_local
        self.loc_data = self.inf_mMALA_uni_local(self.ui)
        self.loc_data = self.local(self.ui)

        print('Initial nll', results[0, 0])
        tic = timeit.default_timer()

        for j in range(n):
            acpt = proposal(h)
            proba += acpt/n
            results[j+1, :] = np.hstack((self.nll, self.ui))

        toc = timeit.default_timer()
        print('Run time is', toc-tic)
        print('Probability', proba)

        return proba, results

    def inf_HMALA_uni_local(self, u):
        # inf_HMALA local information

        nll, nll_grad = self.negll_geom(u)
        nll_hess = self.loc_data[6]
        ku = np.matmul(self.loc_data[2], self.loc_data[2].transpose())

        B = nll_grad - np.dot(nll_hess, u-self.u_mean)
        gu = -np.matmul(ku, B)

        term1 = -B
        term2 = np.dot(gu, term1)
        return nll, gu, self.loc_data[2], term1, term2, 1, nll_hess

    def inf_mMALA_proposal_uni(self, h):

        rho = (1-h/4) / (1+h/4)
        xi = np.matmul(self.loc_data[2], np.random.randn(self.ui.size))
        up = rho*(self.ui-self.u_mean) + self.u_mean
        up += np.sqrt(1-rho**2)*(np.sqrt(h)/2*self.loc_data[1] + xi)

        # Check boundaries
        cond1 = (up[0] > self.lmax) or (up[0] < 0)
        cond2 = (up[1] > self.Dmax) or (up[1] < self.Dmin)

        if (cond1 or cond2):
            acpt = 0
        else:
            w = (up-rho*(self.ui-self.u_mean) - self.u_mean)/np.sqrt(1-rho**2)
            l_u = -self.loc_data[0] - h/8*self.loc_data[4]
            l_u += np.sqrt(h) / 2 * np.dot(self.loc_data[3], w)
            l_u -= np.dot(np.dot(w, self.loc_data[6]), w) / 2

            loc_data = self.local(up)
            wp = (self.ui-rho*(up-self.u_mean) - self.u_mean)/np.sqrt(1-rho**2)

            l_up = -loc_data[0] - h/8*loc_data[4]
            l_up += np.sqrt(h) / 2 * np.dot(loc_data[3], wp)
            l_up -= np.dot(np.dot(wp, loc_data[6]), wp) / 2

            ratio = np.exp(l_up-l_u) * loc_data[5] / self.loc_data[5]

            u = self.ui - self.u_mean
            v = up - self.u_mean
            log_ratio = -np.dot(u[0:2]/self.std**2, u[0:2]) / 2
            log_ratio += np.dot(v[0:2]/self.std**2, v[0:2]) / 2
            ratio *= np.exp(log_ratio)
            acpt = (np.random.rand() <= np.minimum(1, ratio))

            if (acpt == 1):
                self.nll = loc_data[0]
                self.ui = up
                self.loc_data = loc_data[:]
        return acpt


def post_results(results, nburn, nskip):
    # Post treatment of MCMC results
    ntot = results.shape[0]-1
    idx = (nburn + np.arange(0, ntot+1-nburn, nskip)).astype(int)
    keep = results[idx, :].copy()
    CM = np.mean(keep[:, 1:], axis=0)
    return CM, keep


def plot_results(results, N=3):
    # Plot evolution of NLL, Lambda, D in MCMC

    f, axarr = plt.subplots(3+N, sharex=True)
    axarr[0].plot(results[:, 0])
    axarr[0].set_ylabel(r'$\Phi(u;y)$')
    axarr[1].plot(results[:, 1])
    axarr[1].set_ylabel(r'$\lambda$')
    axarr[2].plot(results[:, 2], label='$D$')
    axarr[2].set_ylabel('$D$')
    axarr[3].plot(results[:, 3], label=r'$\xi_0$')
    axarr[3].set_ylabel(r'$\xi_0$')
    axarr[4].plot(results[:, 4], label=r'$\xi_1$')
    axarr[4].set_ylabel(r'$\xi_1$')
    axarr[5].plot(results[:, 5], label=r'$\xi_2$')
    axarr[5].set_ylabel(r'$\xi_2$')
    plt.xlim(0, results[:, 0].size)
    plt.xlabel("Steps")
    plt.show()


def keep_MAP(PDE, keep):
    omf = PDE.omf(keep[0, 1:])
    MAP = keep[0, :]
    print("Omf", omf)
    for i in range(1, keep.shape[0]):
        omfn = PDE.omf(keep[i, 1:])
        if (omfn <= omf):
            omf = omfn.copy()
            MAP = keep[i, :]
            print("Omf", omf)
    return MAP


def plot_acf(results, burn, lags=200):
    # Plot acf for NLL, 1st, 10th and 50th coordinates
    NLL = acorr(results[burn:, 0])
    u0 = acorr(results[burn:, 1])
    u1 = acorr(results[burn:, 2])
    u2 = acorr(results[burn:, 3])
    u3 = acorr(results[burn:, 4])
    u4 = acorr(results[burn:, 5])

    plt.figure()
    plt.plot(np.arange(lags), NLL[0:lags], label=r'$\Phi(u;y)$')
    plt.plot(np.arange(lags), u0[0:lags], label=r'$\lambda$')
    plt.plot(np.arange(lags), u1[0:lags], label='$D$')
    plt.plot(np.arange(lags), u2[0:lags], label=r"$\xi_0$")
    plt.plot(np.arange(lags), u3[0:lags], label=r"$\xi_1$")
    plt.plot(np.arange(lags), u4[0:lags], label=r"$\xi_2$")

    plt.xlabel("Lags")
    plt.ylabel("Self-correlation")
    plt.xlim = (0, lags)
    plt.ylim = (-0.1, 1)
    plt.legend(loc='best')
    plt.show()


def plot_scatter(keep):
    plt.figure()
    plt.scatter(keep[:, 1], keep[:, 2])
    plt.xlabel(r"$\lambda\vert y$")
    plt.ylabel(r"$D\vert y$")
    plt.xlim=(keep[:, 1].min()*0.95, keep[:, 1].max()*1.05)
    plt.ylim=(keep[:, 2].min()*0.95, keep[:, 2].max()*1.05)


def acorr(x):
    # http://stackoverflow.com/q/14297012/190597
    # http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
