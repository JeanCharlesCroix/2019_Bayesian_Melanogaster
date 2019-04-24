# Side file with PDE.py

import numpy as np

def current_kernel1D(kernel, points, s, t, n):
    # Successive kernel
    if (n == 0):
        S, T = np.meshgrid(s, t, indexing="ij")
        return kernel(S, T)
    else:
        mat = current_kernel1D(kernel, points, points[:n], points[:n], 0)
        mat1 = current_kernel1D(kernel, points, s, points[:n], 0)
        mat2 = current_kernel1D(kernel, points, points[:n], t, 0)
        res = current_kernel1D(kernel, points, s, t, 0)
        res -= np.matmul(mat1, np.linalg.solve(mat, mat2))
        return res


def current_basis_function1D(kernel, points, t, n):
    res = current_kernel1D(kernel, points, t, points[n], n).flatten()
    var = current_kernel1D(kernel, points, points[n], points[n], n)[:]
    return res/np.sqrt(var)


def Kernel_prior_tensor(PDE, kernel1, points1, kernel2, points2, n):
    # Assemble matrix with tensor basis
    basis = np.zeros((PDE.points.shape[0], n**2))
    eigenv = np.zeros(n**2)
    k = 0
    for i in range(n):
        for j in range(n):
            basis[:, k] = current_basis_function1D(kernel1, points1, 
                 PDE.points[:, 0], i)
            basis[:, k] = basis[:, k] * current_basis_function1D(kernel2,
                 points2, PDE.points[:, 1], j)
            eigenv1 = current_kernel1D(kernel1, points1, 
                                       points1[k], points1[k], k)
            eigenv2 = current_kernel1D(kernel2, points2, 
                                       points2[k], points2[k], k)
            eigenv[k] = eigenv1 * eigenv2
            k += 1
    
    # Sorting by decreasing variance
    arg = np.argsort(-eigenv)
    basis = basis[:, arg]
    eigenv = eigenv[arg]
    
    return basis, eigenv


def Kernel_prior(PDE, kernel, points, variances):
    # Assemble matrix with KL basis of kernel

    def current_kernel(s, t, n):
        if (t.ndim==1):
            t=t.reshape((1, 2))
        if (s.ndim==1):
            s=s.reshape((1, 2))
        if (n==0):
            Sx, Tx = np.meshgrid(s[:,0], t[:,0], indexing="ij")
            Sy, Ty = np.meshgrid(s[:,1], t[:,1], indexing="ij")
            s = np.vstack((Sx.flatten(), Sy.flatten())).transpose()
            t = np.vstack((Tx.flatten(), Ty.flatten())).transpose()
            res = kernel(s, t)
            return res.reshape(Sx.shape)
        else:
            p = points[0:n, :]
            mat = current_kernel(p, p, 0)
            mat1 = current_kernel(s, p, 0)
            mat2 = current_kernel(p, t, 0)
            res = current_kernel(s, t, 0) - np.matmul(mat1, np.linalg.solve(mat, mat2))
            return res

    def current_basis_function(t, n):
        value = current_kernel(t, points[n,:], n)/np.sqrt(variances[n])
        return value

    basis = np.zeros((PDE.points.shape[0], variances.size))
    for i in range(variances.size):
        print(i, points[i,:])
        basis[:, i] = current_basis_function(PDE.points, i).flatten()

    return basis