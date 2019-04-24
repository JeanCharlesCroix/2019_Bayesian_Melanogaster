# Run the gpCN algorithm

import numpy as np
from PDE import *
from PDE_prior import Kernel_prior_tensor
from PDE_MCMC import MC, post_results, plot_results, plot_acf, plot_scatter
from PDE_plots import *

np.random.seed(13)

# Load MAP and noise variance
MAP = np.genfromtxt("./MAP/MAP.out", delimiter=',')
var_noise = 20.50

### PDE object
EDP = PDE(var_noise=var_noise,
          mesh_cells_number=100,
          dt_number=30,
          body_length=100,
          final_time=100)

# Get data locations and values
data = np.genfromtxt("Data.csv", delimiter=',', skip_header=1)
EDP.set_data(data)

# Add basis functions
points_bb = np.hstack((1/2, np.arange(1, 4, 2)/4, np.arange(1, 8, 2)/8,
                       np.arange(1, 16, 2)/16, np.arange(1, 32, 2)/32,
                       np.arange(1, 64, 2)/64, np.arange(1, 128, 2)/128,
                       np.arange(1, 256, 2)/256, np.arange(1, 512, 2)/1024))

def BrownianB_kernel1(t, s):
    res = 4*(np.minimum(t, s) - t*s/EDP.body_length)/EDP.body_length
    return res


def BrownianB_kernel2(t, s):
    res = 4*(np.minimum(t, s) - t*s/EDP.final_time)/EDP.final_time
    return res


kernel1 = BrownianB_kernel1
kernel2 = BrownianB_kernel2
points1 = points_bb * EDP.body_length
points2 = points_bb * EDP.final_time
Psi, eigenv = Kernel_prior_tensor(EDP, kernel1, points1, kernel2, points2, 20)
EDP.set_source(Psi, eigenv)

#    Instantiate MCMC object
MC1 = MC(negll=EDP.negll,
         negll_geom=EDP.negll_geom,
         negll_hess=EDP.negll_gn,
         lmax=2,
         Dmin=1e-8,
         Dmax=2)

# Run the algorithms
proba, results = MC1.MCMC(alg="inf_HMALA_uni", h=3e-1, n=21000, u0=MAP)
np.savetxt("./Results/Results_"+str(np.round(proba, decimals=2)), results)

plot_results(results)
CM, keep = post_results(results, 1000, 200)
plot_acf(results, 1000)
plot_scatter(keep)
temp = var_sample(EDP, keep[:, 1:])