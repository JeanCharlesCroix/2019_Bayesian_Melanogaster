import numpy as np
from PDE_prior import Kernel_prior_tensor
from PDE_optim import *
from PDE import PDE

np.random.seed(13)

# PDE object
EDP = PDE(var_noise=1e0,
          mesh_cells_number=100,
          dt_number=30,
          body_length=100,
          final_time=100)


# Get data locations and values
data = np.genfromtxt("Data.csv", delimiter=',', skip_header=1)
EDP.set_data(data)

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

#Psi, eigenv = Brownian_bridge_sheet(EDP, 10)
EDP.set_source(Psi, eigenv)

# Initial search
MAP_pre = MAP_mopt(EDP, 50, 20, 2, 2)

# Noise level bootstrap estimation from previous MAP point
sigma_noise = Estimation_noise(EDP, MAP_pre, 20, 2, 2)

# New MAP using estimated noise variance
MAP = MAP_mopt(EDP, 2000, 5, 2, 2)

np.savetxt('./MAP/MAP.out', MAP, delimiter=',')