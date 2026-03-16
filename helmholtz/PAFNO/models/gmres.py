import torch

def gmres(A, b, M, x0=None, tol=1e-6, maxiter=100, verbose=True, log_step=10):
    """
    GMRES with right preconditioning: solving A M^{-1} y = b, then u = M^{-1} y

    Parameters:
        A      : function, linear operator A(x) returning complex-valued tensor
        b      : tensor, shape (batch, 2, N, N) — complex format: (real, imag)
        M      : preconditioner, M(x)
        x0     : initial guess (same shape as b), default is zero
        tol    : convergence tolerance on relative residual
        maxiter: maximum number of iterations
        verbose: whether to print residual per iteration

    Returns:
        u         : approximate solution to A u ≈ b
        res_hist  : list of relative residuals per iteration
    """
    device = b.device
    cdtype = b.dtype

    if x0 is None:
        x0 = torch.zeros_like(b)

    r0 = b - A(x0)
    beta = torch.norm(r0)
    res_hist = [beta.item()]

    if beta < tol:
        return x0, res_hist

    V = [r0 / beta]
    H = torch.zeros((maxiter + 1, maxiter), dtype=cdtype, device=device)

    for j in range(maxiter):
        z = M(V[j])         # Apply right preconditioner
        w = A(z)            # A * M^{-1} * v_j

        # Arnoldi orthogonalization
        for i in range(j + 1):
            H[i, j] = torch.sum(w * V[i].conj())
            w = w - H[i, j] * V[i]

        H[j + 1, j] = torch.norm(w)

        # Breakdown check
        if torch.abs(H[j + 1, j]) < 1e-14:
            if verbose:
                print(f"GMRES lucky breakdown at iteration {j+1}.")
            break

        V.append(w / H[j + 1, j])

        # Solve least squares problem
        H_j = H[:j+2, :j+1]
        e1 = torch.zeros(j + 2, dtype=cdtype, device=device)
        e1[0] = beta

        lstsq_result = torch.linalg.lstsq(H_j, e1)
        y = lstsq_result.solution

        residual_norm = torch.norm(e1 - H_j @ y)
        relres = residual_norm / beta
        res_hist.append(relres.item())

        if verbose and ((j+1)%log_step==0):
            print(f"GMRES iter {j+1:4d}, relres = {relres.item():.2e}")

        if relres < tol:
            break

    # Final solution
    num_iters = len(V)
    H_final = H[:num_iters, :num_iters - 1]
    e1_final = torch.zeros(num_iters, dtype=cdtype, device=device)
    e1_final[0] = beta
    y_final = torch.linalg.lstsq(H_final, e1_final).solution

    # Linear combination of V with y
    V_tensor = torch.stack(V[:num_iters - 1], dim=0)           # (num_iters-1, B, 2, N, N)
    y_reshaped = y_final.view(-1, 1, 1, 1, 1)                   # (num_iters-1, 1, 1, 1, 1)
    update_vec = torch.sum(V_tensor * y_reshaped, dim=0)        # (B, 2, N, N)

    u = x0 + M(update_vec)
    return u, res_hist