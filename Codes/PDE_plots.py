# Side file with PDE.py
# Plotting functions

import numpy as np
import matplotlib.style as st
import matplotlib.pyplot as plt
from PDE import func_to_vec, func_to_vec2
st.use('ggplot')

def plot_source(PDE, lam, D, source):
    # Plot source and solution
    PDE.negll_source(lam, D, source)
    print(PDE.missfit.max())
    sol = func_to_vec(PDE.Y)
    source = func_to_vec2(source)
    Xs, Ts = np.meshgrid(PDE.x, PDE.t)
    fig, axarr = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    levels = np.linspace(source.min(), source.max())
    plt1 = axarr[0].contourf(Xs, Ts, source.reshape(Xs.shape),
                             levels=levels,
                             cmap="viridis")
    axarr[0].set_title("Source $f$")
    axarr[0].set_xlabel('Space')
    axarr[0].set_ylabel('Time')
    axarr[0].set_xlim([0, PDE.body_length])
    axarr[0].set_ylim([0, PDE.final_time])
    plt.colorbar(plt1, ax=axarr[0])

    levels = np.linspace(sol.min(), sol.max())
    plt2 = axarr[1].contourf(Xs, Ts,
                             sol.reshape(Xs.shape),
                             levels=levels,
                             cmap="viridis")
    axarr[1].set_title('Solution $z$ with errors')
    axarr[1].set_xlabel('Space')
    plt.colorbar(plt2, ax=axarr[1])
    plt.xlabel('Space')
    plt.ylabel('Time')
    plt.show()
    data = np.vstack((PDE.data_x, PDE.data_t, PDE.missfit))
    plt3 = axarr[1].scatter(data[0, :], data[1, :],
                            c=np.abs(data[2, :]), s=10, cmap="Greys")
    plt.colorbar(plt3, ax=axarr[1])

def plot_param(PDE, param):
    
    source = PDE.source(param[2:])
    plot_source(PDE, param[0], param[1], source)


def plot_compare(position, result1, result2):
    # Compare 2 parameters
    fig, axarr = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)
    Xs = position[:, :, 0]
    Ts = position[:, :, 1]
    # Compare sources
    source1 = result1[1, :]
    source2 = result2[1, :]
    source3 = np.abs(source1-source2)
    m = np.minimum(source1.min(), source2.min())
    M = np.maximum(source1.max(), source2.max())

    levels = np.linspace(m, M)
    plt1 = axarr[0, 0].contourf(Xs, Ts, source1.reshape(Xs.shape),
                levels=levels, cmap="viridis")
    axarr[0, 0].set_title("$f_{CM}$")
    axarr[0, 0].set_ylabel('Time')
    
    plt.colorbar(plt1, ax=axarr[0, 0])
    plt2 = axarr[0, 1].contourf(Xs, Ts, source2.reshape(Xs.shape),
                levels=levels, cmap="viridis")
    plt.colorbar(plt2, ax=axarr[0, 1])
    axarr[0, 1].set_title("$f_{MAP}$")
    
    levels = np.linspace(source3.min(), source3.max())
    plt3 = axarr[0, 2].contourf(Xs, Ts, source3.reshape(Xs.shape),
                levels=levels, cmap="Reds")
    plt.colorbar(plt3, ax=axarr[0, 2])
    axarr[0, 2].set_title("Absolute difference")

    # Compare solutions
    sol1 = result1[0, :]
    sol2 = result2[0, :]
    sol3 = np.abs(sol1-sol2)
    m = np.minimum(sol1.min(), sol2.min())
    M = np.maximum(sol1.max(), sol2.max())

    levels = np.linspace(m, M)
    plt1 = axarr[1, 0].contourf(Xs, Ts, sol1.reshape(Xs.shape), levels=levels, cmap="viridis")
    plt.colorbar(plt1, ax=axarr[1, 0])
    axarr[1, 0].set_title("$y(u_{CM})$")
    axarr[1, 0].set_ylabel('Time')
    axarr[1, 0].set_xlabel('Space')
    
    plt2 = axarr[1, 1].contourf(Xs, Ts, sol2.reshape(Xs.shape), levels=levels, cmap="viridis")
    plt.colorbar(plt2, ax=axarr[1, 1])
    axarr[1, 1].set_title("$y(u_{MAP})$")
    axarr[1, 1].set_xlabel('Space')
    
    levels = np.linspace(sol3.min(), sol3.max())
    plt3 = axarr[1, 2].contourf(Xs, Ts, sol3.reshape(Xs.shape), levels=levels, cmap="Reds")
    plt.colorbar(plt3, ax=axarr[1, 2])
    axarr[1, 2].set_title("Absolute difference")
    axarr[1, 2].set_xlabel('Space')
    #axarr[1, 2].scatter(self.data_x, self.data_t, s=10, cmap="Greys")

    plt.show()


def var_plot(PDE, sou_var, sol_var):
    # Plot the point-wise variance (source and solution)
    fig, axarr = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    Xs, Ts = np.meshgrid(PDE.x, PDE.t)

    levelsu = np.linspace(sou_var.min(), sou_var.max())
    plt1 = axarr[0].contourf(Xs, Ts, sou_var.reshape(Xs.shape),
                             levels=levelsu, cmap="viridis")
    axarr[0].set_title("Source $f$ posterior variance")
    axarr[0].set_xlabel('Space')
    axarr[0].set_ylabel('Time')
    plt.colorbar(plt1, ax=axarr[0])

    levelss = np.linspace(sol_var.min(), sol_var.max())
    plt2 = axarr[1].contourf(Xs, Ts, sol_var.reshape(Xs.shape),
                             levels=levelss, cmap="viridis")
    axarr[1].set_title("Solution $z$ posterior variance")
    axarr[1].set_xlabel('Space')
    axarr[1].set_ylabel('Time')
    plt.colorbar(plt2, ax=axarr[1])
    axarr[1].scatter(PDE.data_x, PDE.data_t, s=10)
    plt.show()


#    # Functions dedicated to linear problem
#    def set_S(self, Lambda, D):
#        # Parameter to data matrix
#        param = np.zeros(2+self.nkl**2)
#        param[0:2] = np.array([Lambda, D])
#        z = np.zeros(self.data_val.size)
#        S = np.zeros((self.data_val.size, self.nkl**2))
#        for k in range(self.nkl**2):
#            # For each components
#            param[2+k] = 1
#            self.solve_Y(param)
#            
#            for (i, t) in enumerate(self.data_time):
#                idx = np.where(self.data_t == t)[0]
#                f_ij = Function(self.function_space,
#                                self.Y[self.data_time_index[i]])
#                for (j, x) in enumerate(self.data_x[idx]):
#                    z[idx[j]] = f_ij(x)
#            S[:, k] = z.copy()
#            param[2+k] = 0
#        return S
#
#    def set_linear_solution(self, Lambda, D):
#        # Compute G(lambda, D)
#        print("Computing linear solution...")
#        S = self.set_S(Lambda, D)
#        cov0 = np.diag(self.eigenv)
#        temp2 = np.matmul(cov0, S.transpose())
#        # Conditional mean and covariance for theta
#        temp = np.matmul(S, temp2) + np.eye(self.data_val.size)/self.var_noise
#        mean = np.matmul(temp2, np.linalg.solve(temp, self.data_val))
#        cov = np.linalg.solve(temp, temp2.transpose())
#        cov = cov0 - np.matmul(temp2, cov)
#        return mean, cov