import torch

def fgmres(A, b, M, x0=None, tol=1e-6, maxiter=100, verbose=True, log_step=10):
    """
    Flexible GMRES with right preconditioning.

    Works when preconditioner M is not strictly complex-linear by storing z_j = M(v_j)
    and reconstructing x_j = x0 + sum_k y_k z_k.
    Stopping criterion uses true residual ||b - A(x_j)|| / ||b||.

    Parameters:
        A      : function, linear operator A(x) returning complex-valued tensor
        b      : tensor, right-hand side
        M      : right preconditioner application z = M(v)
        x0     : initial guess (same shape as b), default is zero
        tol    : convergence tolerance on true relative residual
        maxiter: maximum number of iterations
        verbose: whether to print residual logs
        log_step: logging interval

    Returns:
        x         : approximate solution to A x ≈ b
        res_hist  : list of true relative residuals, starts from initial residual
    """
    device = b.device
    dtype = b.dtype
    eps = 1e-30
    breakdown_tol = 1e-14

    if x0 is None:
        x0 = torch.zeros_like(b)

    b_norm = torch.norm(b)
    b_norm = torch.clamp(b_norm, min=eps)

    r0 = b - A(x0)
    beta = torch.norm(r0)
    rel0 = (beta / b_norm).item()
    res_hist = [rel0]

    if rel0 < tol:
        return x0, res_hist

    V = [r0 / beta]
    Z = []
    H = torch.zeros((maxiter + 1, maxiter), dtype=dtype, device=device)

    x = x0

    for j in range(maxiter):
        z_j = M(V[j])
        Z.append(z_j)
        w = A(z_j)

        # Arnoldi orthogonalization
        for i in range(j + 1):
            H[i, j] = torch.sum(w * V[i].conj())
            w = w - H[i, j] * V[i]

        H[j + 1, j] = torch.norm(w)
        h_next = torch.abs(H[j + 1, j]).item()
        if h_next > breakdown_tol:
            V.append(w / H[j + 1, j])
        elif verbose:
            print(f"FGMRES breakdown at iteration {j+1}.")

        # Solve projected least-squares problem
        H_j = H[:j+2, :j+1]
        e1 = torch.zeros(j + 2, dtype=dtype, device=device)
        e1[0] = beta

        y = torch.linalg.lstsq(H_j, e1).solution

        Z_tensor = torch.stack(Z, dim=0)
        y_shape = [j + 1] + [1] * (Z_tensor.ndim - 1)
        x = x0 + torch.sum(Z_tensor * y[:j + 1].reshape(y_shape), dim=0)

        # True residual stopping criterion
        true_relres = torch.norm(b - A(x)) / b_norm
        true_relres_item = true_relres.item()
        res_hist.append(true_relres_item)

        if verbose and (((j + 1) % log_step == 0) or (true_relres_item < tol)):
            print(f"FGMRES iter {j+1:4d}, relres = {true_relres_item:.2e}")

        if true_relres_item < tol:
            break

        if h_next <= breakdown_tol:
            break

    return x, res_hist
