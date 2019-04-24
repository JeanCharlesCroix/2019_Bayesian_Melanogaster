import numpy as npfrom fenics import *from scipy.interpolate import RectBivariateSpline, interp1d# Code for Melanogaster problem# PDE: z_t + Lambda * z - D * Delta z = f,# Initial condition : z(0) = 0# Boundary conditions: z(t)[0] = z(t)[Lx] = 0# PDE solutions are stored as a list of GenericVector at each time t# z[k] = z(t_k,.) (get correct positions order using tabulate_dof_coordinates())# Implicit Euler time discretization scheme (unconditionally stable)# Finite element of order 1 in space using Fenics# Discrete adjoint methods for gradient and hessian computations# Please read implementation notes for more details.__author__ = "Jean-Charles Croix"__copyright__ = "Copyright 2019"__email__ = "j.croix@sussex.ac.uk"class PDE():    # This object contains:    # - Negative log-likelihood evaluation (+gradient, Hessian)    # - A set of data    # - Set of basis functions    def __init__(self, body_length=1, final_time=1,                 mesh_cells_number=100, dt_number=20,                 finite_element_degree=1, var_noise=0.0001):        # Mesh and geometry parameters        self.body_length = body_length  # Length of drosophilia embryo        self.final_time = final_time  # Time horizon        self.mesh_cells_number = mesh_cells_number  # Size of space mesh        self.dt_number = dt_number  # Number of time steps        self.finite_element_degree = finite_element_degree  # DoF of Lagrange        # Finite elements matrices assembly        self.set_fem()  # Initialize mesh and function space        self.set_forms()  # Set finite element matrices        # Gaussian likelihood parameter        self.var_noise = var_noise  # Noise on data        # Fake data        data = np.random.rand(30, 3)  # Fake dataset as Numpy array [t,x,val]        data[:, 0] *= self.final_time*0.9        data[:, 1] *= self.body_length*0.9        self.set_data(data)  # set Data, dt, etc...    def set_data(self, data):        # Set data from external file        # data = [t, x, val]        self.data_time = np.unique(data[:, 0])        self.data_t = data[:, 0]        self.data_x = data[:, 1]        self.data_val = data[:, 2]        self.missfit = np.zeros(data.shape[0])        self.missfit2 = self.missfit.copy()        # Create list of Points for dirac deltas        self.Points_list = list()        for (i, p) in enumerate(self.data_x):            self.Points_list.append(Point(p))        self.t = np.linspace(0, self.final_time, self.dt_number)        self.dt = np.diff(self.t)        tol = self.dt.max()        for t in self.data_time:            grid = [tk for tk in self.t if (np.abs(tk-t) > tol/3)]        self.t = np.hstack((np.array(grid), self.data_time))        self.t.sort()        self.dt = np.diff(self.t)        self.dt_number = len(self.dt)        #idx = np.isin(self.t, self.data_time)        idx = np.array([item in self.data_time for item in self.t])        self.data_time_index = np.arange(self.t.size)[idx]        # Object creation        self.function_vector = Function(self.function_space).vector()        self.Y = [] #((N+1)xNh)        self.W = [] #((N+1)xNh)        self.Rhp = [] #((N+1)xNh)        self.Qhp = [] #((N+1)xNh)        for i in range(self.dt.size+1):            self.Y.append(self.function_vector.copy())            self.W.append(self.function_vector.copy())            self.Rhp.append(self.function_vector.copy())            self.Qhp.append(self.function_vector.copy())        # Where the functions must be evaluated        X, T = np.meshgrid(self.x, self.t)        self.points = np.hstack((X.reshape(X.size, 1), T.reshape(T.size, 1)))    # Source related functions    def set_source(self, Psi, eigenv):        # Psi=[Psi^1, Psi^2, ..., Psi^N]        # Convert to rows of different times        self.eigenv = eigenv        self.psi = []        for t in self.t:            self.psi.append(Psi[np.where(self.points[:, 1] == t)[0], :])    def source(self, param):        # Compute exponential source as a list        source = []        for fun in self.psi:            source.append(np.minimum(np.exp(np.matmul(fun, param)),1e30))        return source    def source_lin(self, param):        # Compute source as a list        source = []        for fun in self.psi:            source.append(np.matmul(fun, param))        return source    # Finite element: Fenics    def set_fem(self):        # Create Mesh, FunctionSpace and Boundary conditions        self.mesh = IntervalMesh(self.mesh_cells_number, 0.0, self.body_length)        self.function_space = FunctionSpace(                self.mesh, 'Lagrange', self.finite_element_degree)        self.x = self.function_space.tabulate_dof_coordinates()[:]        def u0_boundary(x, on_boundary):            return on_boundary        self.Y_bc = DirichletBC(self.function_space,                                    Constant(0.0), u0_boundary)        self.W_bc = DirichletBC(self.function_space,                                      Constant(0.0), u0_boundary)    def set_forms(self):        # Set forms and assemble time independent matrices        u = TrialFunction(self.function_space)        v = TestFunction(self.function_space)        a = u * v * dx        b = inner(grad(u), grad(v)) * dx        self.A = assemble(a)        self.B = assemble(b)    # Negative log-likelihood related functions    def negll(self, param):        # Compute negative log-likelihood        source_param = self.source(param[2:])        return self.negll_source(param[0], param[1], source_param)    def negll_source(self, lam, D, source):        # Compute negative log-likelihood        nll = 0        self.solve_Y_source(lam, D, source)        z = np.zeros((self.x.size, self.t.size))        for (i, k) in enumerate(self.Y):            z[:, i] = np.flip(k[:])        interp = RectBivariateSpline(np.flip(self.x[:, 0]), self.t, z)        # Compute nll from Y        self.missfit = self.data_val.copy()        self.missfit -= interp(self.data_x, self.data_t, grid=False)        nll = np.sum(self.missfit**2)        nll /= (2 * self.var_noise)        return nll    # Onsager-Machlup functional    def omf(self, param):        omf = self.negll(param)        omf += np.dot(param[2:], param[2:])/2        return omf    def omf_geom(self, param):        omf, omf_grad = self.negll_geom(param)        omf += np.dot(param[2:], param[2:])/2        omf_grad[2:] += param[2:]        return omf, omf_grad    def omf_gnp(self, param, v):        omf, omf_grad = self.omf_geom(param)        temp = self.negll_hessp(param, v)        return temp+v    def negll_hess(self, param):        # Compute negative log-likelihood, Gradient and Hessian        nll, nll_grad = self.negll_geom(param)        nll_hess = np.zeros((nll_grad.size, nll_grad.size))        v = np.zeros(param.size)        for i in range(v.size):            v[i] = 1            nll_hess[:, i] = self.negll_hessp(param, v)            v[i] = 0        return nll, nll_grad, nll_hess    def negll_gn(self, param):        # Fast computation of nll, nll_grad, nll_gn        nll, nll_grad = self.negll_geom(param)        v = np.zeros(param.size)        nll_gn = np.zeros((param.size, param.size))        for i in range(param.size):            v[i] = 1            nll_gn[:, i] = self.negll_gnp(param, v)            v[i] = 0        return nll, nll_grad, nll_gn    def negll_geom(self, param):        # Compute nll and nll_grad        nll = self.negll(param) #Solve Y(u) implicitely        self.solve_W(param) #Adjoint model        source = self.source(param[2:])        # Assemble gradient        nll_grad = np.zeros(param.size)        for (i, dt) in enumerate(self.dt):            temp = float(dt) * self.A * self.W[i+1]            nll_grad[0] += self.Y[i+1].inner(temp) #Lambda            temp = float(dt) * self.B * self.W[i+1]            nll_grad[1] += self.Y[i+1].inner(temp) #D            temp = float(dt) * self.A * self.W[i+1]            temp2 = source[i+1]*self.psi[i+1].transpose() #Theta            nll_grad[2:] -= np.matmul(temp2, temp[:])        return nll, nll_grad    def negll_hessp(self, param, v):        # Compute nll_hessp        # Y(u) and W(u) are supposed already updated        self.solve_R(param, v)        self.solve_Qhp(param, v)        source = self.source(param[2:])        source2 = self.source(v[2:])        nll_hessp = np.zeros(param.size)        for (i, dt) in enumerate(self.dt):            # Hessian product w.r.t. lambda            temp = float(dt) * self.A * self.Qhp[i+1]            temp2 = float(dt) * self.A * self.W[i+1]            nll_hessp[0] += self.Y[i+1].inner(temp)            nll_hessp[0] -= self.Rhp[i+1].inner(temp2)            # Hessian product w.r.t. D            temp = float(dt) * self.B * self.Qhp[i+1]            temp2 = float(dt) * self.B * self.W[i+1]            nll_hessp[1] += self.Y[i+1].inner(temp)            nll_hessp[1] -= self.Rhp[i+1].inner(temp2)            # Hessian product w.r.t. theta            temp = float(dt) * self.A * self.Qhp[i+1]            temp3 = source[i+1]*self.psi[i+1].transpose()            nll_hessp[2:] -= np.matmul(temp3, temp[:])            temp3 = (source[i+1]*source2[i+1])*(self.psi[i+1].transpose())            nll_hessp[2:] += np.matmul(temp3, temp2[:])        return nll_hessp    def negll_gnp(self, param, v):        # Compute nll_hessp        # Y(u) is supposed already updated        self.solve_R(param, v)        self.solve_Qgnp(param, v)        source = self.source(param[2:])        nll_gnp = np.zeros(param.size)        for (i, dt) in enumerate(self.dt):            # Hessian product w.r.t. lambda            temp = float(dt) * self.A * self.Qhp[i+1]            nll_gnp[0] += self.Y[i+1].inner(temp)            # Hessian product w.r.t. D            temp = float(dt) * self.B * self.Qhp[i+1]            nll_gnp[1] += self.Y[i+1].inner(temp)            # Hessian product w.r.t. theta            temp = float(dt) * self.A * self.Qhp[i+1]            temp2 = source[i+1]*self.psi[i+1].transpose()            nll_gnp[2:] -= np.matmul(temp2, temp[:])        return nll_gnp    # Solvers (Forward, Adjoint, ...)    def solve_Y_source(self, lam, D, source):        # Update Y        matrix = float(lam) * self.A        matrix += float(D) * self.B        self.Y[0].zero()        for (i, dt) in enumerate(self.dt):            lhs = self.A + float(dt)*matrix #M_k            self.function_vector[:] = dt*source[i+1] #dt^kF^k            rhs = self.Y[i] + self.function_vector            rhs = self.A * rhs            self.Y_bc.apply(lhs, rhs)            solve(lhs, self.Y[i+1], rhs, 'gmres', 'ilu')    def solve_Y(self, param):        # Update Y        source_param = self.source(param[2:])        self.solve_Y_source(param[0], param[1], source_param)    def solve_Qhp(self, param, v):        # Solve Q(u,v) for hessian-product        # Adjoint solve with ... as source term        matrix = float(param[0]) * self.A        matrix += float(param[1]) * self.B        matrix2 = float(v[0]) * self.A        matrix2 += float(v[1]) * self.B        self.Qhp[-1].zero()  # Null final condition (no data here)        for i in range(self.dt_number-1, 0, -1):            #F_Y(Y(u),u)^TQ=Nabla_YYLR(u,v)-Nabla_Yuv            lhs = self.A + float(self.dt[i-1])*matrix            rhs = self.A*self.Qhp[i+1] - float(self.dt[i-1])*matrix2*self.W[i]            if self.t[i] in self.data_t:                idx = np.where(self.data_t == self.t[i])[0]                List = list(zip([self.Points_list[i] for i in idx],                                self.missfit2[idx]/self.var_noise))                d = PointSource(self.function_space, List)                d.apply(rhs)            self.W_bc.apply(lhs, rhs)            solve(lhs, self.Qhp[i], rhs, 'gmres', 'ilu')    def solve_Qgnp(self, param, v):        # Solve Q(u, v) for GN-product        matrix = float(param[0]) * self.A        matrix += float(param[1]) * self.B        self.Qhp[-1].zero()  # Null final condition (no data here)        for i in range(self.dt_number-1, 0, -1):            #F_Y(Y(u),u)^TQ=Nabla_YYLR(u,v)            lhs = self.A + float(self.dt[i-1])*matrix            rhs = self.A*self.Qhp[i+1]            if self.t[i] in self.data_t:                idx = np.where(self.data_t == self.t[i])[0]                List = list(zip([self.Points_list[i] for i in idx],                                self.missfit2[idx]/self.var_noise))                d = PointSource(self.function_space, List)                d.apply(rhs)            self.W_bc.apply(lhs, rhs)            solve(lhs, self.Qhp[i], rhs, 'gmres', 'ilu')    def solve_W(self, param):        # Update W        matrix = float(param[0]) * self.A        matrix += float(param[1]) * self.B        self.W[-1].zero()  # Null final condition (no data here)        for i in range(self.dt_number-1, 0, -1):            lhs = self.A + float(self.dt[i-1])*matrix            rhs = self.A * self.W[i+1]            if self.t[i] in self.data_t:                idx = np.where(self.data_t == self.t[i])[0]                List = list(zip([self.Points_list[i] for i in idx], self.missfit[idx]/self.var_noise))                d = PointSource(self.function_space, List)                d.apply(rhs)            self.W_bc.apply(lhs, rhs)            solve(lhs, self.W[i], rhs, 'gmres', 'ilu')    def solve_R(self, param, v):        # Solve R(u,v) for hessian-product        # Forward solve with Fu(Y(u),u)v as source term        source_lin = self.source_lin(v[2:])        source_exp = self.source(param[2:]) #F=(F^0,...,F^N)        matrix = float(param[0]) * self.A        matrix += float(param[1]) * self.B        matrix2 = float(v[0]) * self.A        matrix2 += float(v[1]) * self.B        self.Rhp[0].zero()        for (i, dt) in enumerate(self.dt):            # F_Y(Y(u),u)R=F_u(Y(u),u)v            lhs = self.A + float(dt)*matrix            self.function_vector[:] = dt*source_lin[i+1]*source_exp[i+1]            rhs = self.Rhp[i] - self.function_vector #F_u(Y(u),u)v            rhs = self.A*rhs + float(dt)*matrix2*self.Y[i+1]            self.Y_bc.apply(lhs, rhs)            solve(lhs, self.Rhp[i+1], rhs, 'gmres', 'ilu')                # Compute R(u,v) at data points (missfit2)        for (i, t) in enumerate(self.data_time):            idx = np.where(self.data_t == t)[0]            x = np.flip(self.x[:, 0])            y = np.flip(self.Rhp[self.data_time_index[i]][:])            interp = interp1d(x, y, kind="cubic")            self.missfit2[idx] = interp(self.data_x[idx])def func_to_vec(func):    # Convert Y solution to a numpy array    vec = np.array([])    for k in func:        vec = np.hstack((vec, k[:].copy()))    return vecdef func_to_vec2(func):    # Convert Y solution to a numpy array    vec = np.array([])    for k in func:        vec = np.hstack((vec, k.copy()))    return vecdef var_sample(PDE, sample):    # Compute point-wise sample variance    temp = func_to_vec(PDE.Y).size    sou = np.zeros((sample.shape[0], temp))    sol = np.zeros((sample.shape[0], temp))    for i in range(sample.shape[0]):        sou[i, :] = func_to_vec2(PDE.source(sample[i, 3:]))        PDE.solve_Y(sample[i, 1:])        sol[i, :] = func_to_vec(PDE.Y).copy()    sou_var = np.diag(np.cov(sou.T))    sol_var = np.diag(np.cov(sol.T))    return sou_var, sol_var