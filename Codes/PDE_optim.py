import numpy as np
from scipy.optimize import minimize
import timeit


# MAP search
def MAP_opt(EDP, param0, N, lm, Dm):
    # Minimize omf with L-BFGS-B
    bnds_list = []
    bnds_list.append((0, lm))
    bnds_list.append((1e-8, Dm))
    for i in range(EDP.psi[0].shape[1]):
        bnds_list.append((None, None))
    bnds = tuple(bnds_list)
    res = minimize(fun=EDP.omf_geom,
                   jac=True,
                   x0=param0,
                   bounds=bnds,
                   method="L-BFGS-B",
                   options={'maxiter': N})
    print("Success :", res.success)
    if (res.success == 0):
        print("Message :", res.message)
    return res

# MAP search with multi-start
def MAP_mopt(EDP, N, k, lm, Dm):
    # MAP optim with multi start
    sample = np.zeros((k, EDP.psi[0].shape[1]+2))
    sample[:, 0:2] = np.random.rand(k, 2)*np.array([lm, Dm])
    sample[:, 2:] = np.random.randn(k, EDP.psi[0].shape[1])
    best = sample[0, :]
    best_omf = EDP.omf(best)
    for i in range(k):
        print("Attempt ", i+1)
        print("Start (Lambda, D, omf):",
              np.round(sample[i, 0:2], decimals=2),
              np.round(EDP.omf(sample[i, :]), decimals=2))
        tic = timeit.default_timer()
        Opt = MAP_opt(EDP, sample[i, :], N, lm, Dm)
        toc = timeit.default_timer()
        print('Run time is', np.round(toc-tic, decimals=0))
        print("Stop (Lambda, D, omf):", np.round(Opt.x[0:2], decimals=2),
              np.round(Opt.fun, decimals=2))
        if (Opt.fun < best_omf):
            best_omf = Opt.fun
            best = Opt.x
            print("New MAP Omf:", np.round(best_omf, decimals=2))
            print("New MAP Lambda,D:", np.round(best[0:2], decimals=2))
    return best


# Estimation of noise
def Estimation_noise(EDP, MAP, N, lm, Dm):
    noise = np.zeros(N)
    noise[0] = EDP.negll(MAP) * EDP.var_noise * 2 / (EDP.data_val.size-1)
    current = MAP
    for i in range(1, N):
        print("Step ", i)
        current = MAP_opt(EDP, current, 1000, 1, 2).x
        noise[i] = EDP.negll(current) * EDP.var_noise * 2 / (EDP.data_val.size-1)
        EDP.var_noise = noise[i]
        print("Var noise", np.round(noise[i], decimals=2))
    return noise, current
