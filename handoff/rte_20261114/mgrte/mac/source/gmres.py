"""
    define a general GMRES API where user
    can define its own matrix-vector product,
    vector-vector product...

"""
import numpy as np
import torch


def _fgmres_cycle(A, b, M, x0, b_norm, tol, maxiter, verbose, log_step, iter_offset):
    device = b.device
    dtype = b.dtype
    breakdown_tol = 1e-14

    r0 = b - A(x0)
    beta = torch.norm(r0)
    rel0 = (beta / b_norm).item()
    history = [rel0]
    if rel0 < tol:
        return x0, history, 0, True, False

    V = [r0 / beta]
    Z = []
    H = torch.zeros((maxiter + 1, maxiter), dtype=dtype, device=device)
    x = x0
    converged = False
    breakdown = False

    for j in range(maxiter):
        z_j = M(V[j])
        Z.append(z_j)
        w = A(z_j)
        for i in range(j + 1):
            H[i, j] = torch.sum(w * V[i].conj())
            w = w - H[i, j] * V[i]

        H[j + 1, j] = torch.norm(w)
        h_next = torch.abs(H[j + 1, j]).item()
        if h_next > breakdown_tol:
            V.append(w / H[j + 1, j])
        else:
            breakdown = True

        H_j = H[:j + 2, :j + 1]
        e1 = torch.zeros(j + 2, dtype=dtype, device=device)
        e1[0] = beta
        y = torch.linalg.lstsq(H_j, e1).solution

        Z_tensor = torch.stack(Z, dim=0)
        y_shape = [j + 1] + [1] * (Z_tensor.ndim - 1)
        x = x0 + torch.sum(Z_tensor * y[:j + 1].reshape(y_shape), dim=0)

        relres = (torch.norm(b - A(x)) / b_norm).item()
        history.append(relres)
        global_iter = iter_offset + j + 1
        if verbose and ((global_iter % log_step == 0) or relres < tol):
            print(f"FGMRES iter {global_iter:4d}, relres = {relres:.2e}")
        if relres < tol:
            converged = True
            break
        if breakdown:
            break

    return x, history, len(history) - 1, converged, breakdown


def fgmres(A, b, M, x0=None, tol=1e-6, maxiter=100, restart=None, max_restarts=None, verbose=True, log_step=10):
    """
    Flexible right-preconditioned GMRES using true residual stopping.

    This mirrors the NPBS/Helmholtz workflow: store z_j = M(v_j), then
    reconstruct x_j from the preconditioned Arnoldi basis. It is safer for
    learned RTE preconditioners because M may be nonlinear or stateful.
    """
    if x0 is None:
        x0 = torch.zeros_like(b)
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if restart is not None:
        restart = min(int(restart), maxiter)
        if restart < 1:
            raise ValueError("restart must be >= 1")
    if max_restarts is not None and int(max_restarts) < 0:
        raise ValueError("max_restarts must be >= 0")

    b_norm = torch.clamp(torch.norm(b), min=1e-30)
    log_step = max(int(log_step), 1)
    if restart is None:
        x, history, _, _, _ = _fgmres_cycle(A, b, M, x0, b_norm, tol, maxiter, verbose, log_step, 0)
        return x, history

    x = x0
    history = []
    total_iters = 0
    restarts_used = 0
    while total_iters < maxiter:
        inner_maxiter = min(restart, maxiter - total_iters)
        x, cycle_history, cycle_iters, converged, breakdown = _fgmres_cycle(
            A, b, M, x, b_norm, tol, inner_maxiter, verbose, log_step, total_iters
        )
        history.extend(cycle_history if not history else cycle_history[1:])
        total_iters += cycle_iters
        if converged or breakdown or cycle_iters == 0:
            break
        if max_restarts is not None and restarts_used >= int(max_restarts):
            break
        restarts_used += 1
        if verbose:
            print(f"FGMRES restart {restarts_used} after {total_iters} iterations, relres = {history[-1]:.2e}")
    return x, history


class mygmres():
    def __init__(self):
        pass
    # def __init__(self, tol=1e-6, max_iter=100):
        # tol = tol
        # max_iter = max_iter
        # self.restart = restart

    def matvec(self, x):
        raise NotImplementedError
    
    def dot(self, x, y):
        raise NotImplementedError
    
    def zeros_like(self, x):
        raise NotImplementedError

    def scale(self, x, a):
        raise NotImplementedError
    
    def axby(self, a, x, b, y):
        raise NotImplementedError

    def vecnorm(self, x):
        return np.sqrt(self.dot(x, x))
    
    def solve(self, b, tol=1e-6, max_iter=100, restart=300, verbose=False):
        beta = self.vecnorm(b)
        # print("Initial residual norm: ", beta)
        if verbose:
            # print("Iteration: ", 0, "Residual norm: ", beta, "Relative residual norm: ", 1.0)
            print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (0, beta, 1.0))
        V = []
        V.append(self.scale(b, 1/beta))
        H = np.zeros((max_iter + 1, max_iter))
        # num_iter = max_iter

        # Arnoldi process
        relres_history = [1.0]
        for j in range(max_iter):
            w = self.matvec(V[j])
            for i in range(j + 1):
                H[i, j] = self.dot(V[i], w)
                w = self.axby(1, w, -H[i, j], V[i])

            H[j + 1, j] = self.vecnorm(w)
            V.append(self.scale(w, 1/H[j + 1, j]))
        # print("H: ", H)

            num_iter = j + 1
            # Solve the least squares problem
            e1 = np.zeros(num_iter + 1)
            e1[0] = beta
            y, residual_norm, _, _ = np.linalg.lstsq(H[:num_iter + 1, :num_iter], e1, rcond=None)
            residual_norm = np.sqrt(residual_norm).item()

            # Compute the approximate solution
            x = self.zeros_like(b)
            for i in range(num_iter):
                x = self.axby(1, x, y[i], V[i])

            # Compute the residual
            # r = self.axby(1, b, -1, self.matvec(x))
            # residual_norm = self.vecnorm(r)
            relres_history.append(residual_norm/beta)

            # Check for convergence
            if verbose:
                print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (num_iter, residual_norm, residual_norm/beta))
                
            if residual_norm/beta < tol:
                break

        if 1 and not verbose:
            print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (num_iter, residual_norm, residual_norm/beta))
        return x, relres_history
    
class arraygmres(mygmres):
    def __init__(self, A):#, tol=1e-6, max_iter=100):
        super().__init__()#(tol, max_iter)
        self.A = A
    
    def matvec(self, x):
        return (self.A @ x)
    
    def dot(self, x, y):
        return np.dot(x, y)
    
    def zeros_like(self, x):
        return np.zeros_like(x)
    
    def scale(self, x, a):
        return a * x
    
    def axby(self, a, x, b, y):
        return (a * x + b * y)

    def vecnorm(self, x):
        return np.linalg.norm(x)
    
class mygmres2dtorch(mygmres):
    # def __init__(self, mgcnn, myop, tol=1e-8, max_iter=3):
    def __init__(self, mgcnn, myop):
        super().__init__()
        self.mgcnn = mgcnn
        self.myop = myop

    def matvec(self, x):
        # right preconditioning
        return self.myop(self.mgcnn(x))
        # return self.mgcnn(self.myop(x))
    
    def dot(self, x, y):
        # return torch.sum(x * y)
        prod = torch.sum(x * y).item()
        # print('>>> dot product: ', prod)
        return prod
    
    def zeros_like(self, x):
        return torch.zeros_like(x)
    
    def scale(self, x, a):
        return a * x
    
    def axby(self, a, x, b, y):
        return a * x + b * y

    def vecnorm(self, x):
        # return torch.norm(x)    
        _norm = torch.norm(x).item()
        # print('>>> norm: ', _norm)
        return _norm
    
    def solve(self, b, tol, max_iter, verbose):
        # b = self.mgcnn(b) # left preconditioning
        x, relres_history = super().solve(b, tol, max_iter, verbose=verbose)
        x = self.mgcnn(x) # right preconditioning
        print('>>> residual norm: ', self.vecnorm(b - self.myop(x)))
        return x, relres_history
    
    def solve_with_restart(self, b, tol, max_iter, restart, verbose):
        print("Using restart solve with restart: ", restart)
        relres = 1
        norm_b = self.vecnorm(b)
        x = self.zeros_like(b)
        sum_iters = 0
        relres_history = [1.0]
        res = b
        res_norm = norm_b
        relres_restart_history = []
        while(relres > tol and sum_iters < max_iter):
            e, e_relres_history = super().solve(res, tol/relres, restart, verbose=verbose)
            sum_iters += len(e_relres_history) - 1
            e_relres_history = [val*res_norm for val in e_relres_history]
            relres_history += e_relres_history[1:]
            x = self.axby(1, x, 1, e)
            res = b - self.matvec(x)
            res_norm = self.vecnorm(res)
            relres = res_norm / norm_b
            relres_restart_history.append(relres)
            print(">>> Relative residual: ", relres)
            if len(e_relres_history) <= 2:
                break
            # if len(relres_restart_history) > 1 and relres_restart_history[-1] > 0.95*relres_restart_history[-2]:
            #     print("!!! reduce too small, stop")
            #     break
        x = self.mgcnn(x)
        print('>>> residual norm: ', self.vecnorm(b - self.myop(x)))
        print('ITERATION: ', sum_iters)
        return x, relres_history
    
    def setup_64(self, myop_64, mgcnn_64):
        self.myop_64 = myop_64
        self.mgcnn_64 = mgcnn_64
    
    def matvec_64(self, x):
        return self.myop_64(self.mgcnn_64(x))
        
    def matvec_mixed(self, x):
        return self.myop_64(self.mgcnn(x))
    
    # 64-bit computation is slow on GPU, therefore, we utilize mix-precision computation
    # i.e. 32-bit computation on solve, and 64-bit computation on residual update
    def solve_with_restart_mix_precision2(self, b, tol, max_iter, restart, verbose):
        print("Using restart solve with restart: ", restart)
        print("MIX PRECISION 2")
        relres = 1
        
        b_64 = b.to(torch.float64, copy=True)
        
        norm_b = self.vecnorm(b_64)
        x_64 = self.zeros_like(b_64)
        sum_iters = 0
        relres_history = [1.0]
        res_64 = b_64
        res_norm = norm_b
        relres_restart_history = []
        
        while(relres > tol and sum_iters < max_iter):
            res = res_64.to(torch.float32, copy=True)
            pc_e, e_relres_history = super().solve(res, tol/relres, restart, verbose=verbose)
            sum_iters += len(e_relres_history) - 1
            e_relres_history = [val*res_norm for val in e_relres_history]
            relres_history += e_relres_history[1:]
            e = self.mgcnn(pc_e)
            e_64 = e.to(torch.float64, copy=True)            
            x_64 = self.axby(1, x_64, 1, e_64)
            res_64 = b_64 - self.myop_64(x_64)
            res_norm = self.vecnorm(res_64)
            relres = res_norm / norm_b
            relres_restart_history.append(relres)
            print(">>> Relative residual: ", relres)
            if len(e_relres_history) <= 2:
                break
            # if len(relres_restart_history) > 1 and (relres_restart_history[-2] - relres_restart_history[-1]) < 0.97*relres_restart_history[-1]:
                # break
        # print('>>> residual norm: ', self.vecnorm(b - self.myop(x)))
        greenprint('ITERATION: %d'%sum_iters)
        return x_64, relres_history
    
def greenprint(print_str):
    print('\033[92m%s\033[0m' % print_str)
    
if __name__ == "__main__":
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
    # b = np.array([1, 2, 3])
    b = np.array([12, 3, 10])
    # ndim = 10
    # A = np.random.rand(ndim, ndim)
    # b = np.random.rand(ndim)
    # laplacian
    # A = np.diag(np.ones(ndim) * 2) - np.diag(np.ones(ndim - 1), 1) - np.diag(np.ones(ndim - 1), -1)
    # b = np.ones(ndim)
    tol = 1e-6
    max_iter = 100
    gmres = arraygmres(A)#, tol, max_iter)
    x, num_iter = gmres.solve(b, tol, max_iter, verbose=True)
    print("Solution: ", x)
    print("Number of iterations: ", num_iter)
    print("Residual: ", b - A @ x)
    print("Residual norm: ", np.linalg.norm(b - A @ x))
