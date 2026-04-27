import torch


def _fgmres_cycle(A, b, M, x0, b_norm, tol, maxiter, verbose, log_step, iter_offset):
    device = b.device
    dtype = b.dtype
    breakdown_tol = 1e-14

    r0 = b - A(x0)
    beta = torch.norm(r0)
    rel0 = (beta / b_norm).item()
    res_hist = [rel0]

    if rel0 < tol:
        return x0, res_hist, 0, True, False

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

        # Arnoldi orthogonalization
        for i in range(j + 1):
            H[i, j] = torch.sum(w * V[i].conj())
            w = w - H[i, j] * V[i]

        H[j + 1, j] = torch.norm(w)
        h_next = torch.abs(H[j + 1, j]).item()
        if h_next > breakdown_tol:
            V.append(w / H[j + 1, j])
        elif verbose:
            print(f"FGMRES breakdown at iteration {iter_offset + j + 1}.")

        # Solve projected least-squares problem
        H_j = H[:j + 2, :j + 1]
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

        global_iter = iter_offset + j + 1
        if verbose and ((global_iter % log_step == 0) or (true_relres_item < tol)):
            print(f"FGMRES iter {global_iter:4d}, relres = {true_relres_item:.2e}")

        if true_relres_item < tol:
            converged = True
            break

        if h_next <= breakdown_tol:
            breakdown = True
            break

    return x, res_hist, len(res_hist) - 1, converged, breakdown


def fgmres(
    A,
    b,
    M,
    x0=None,
    tol=1e-6,
    maxiter=100,
    restart=None,
    max_restarts=None,
    verbose=True,
    log_step=10,
):
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
        restart: Arnoldi subspace size per cycle. If None, run unrestarted GMRES.
        max_restarts: maximum number of restart cycles after the first cycle.
        verbose: whether to print residual logs
        log_step: logging interval

    Returns:
        x         : approximate solution to A x ≈ b
        res_hist  : list of true relative residuals, starts from initial residual
    """
    eps = 1e-30

    if x0 is None:
        x0 = torch.zeros_like(b)

    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")

    if restart is not None:
        restart = int(restart)
        if restart < 1:
            raise ValueError("restart must be >= 1 when provided")
        restart = min(restart, maxiter)

    if max_restarts is not None:
        max_restarts = int(max_restarts)
        if max_restarts < 0:
            raise ValueError("max_restarts must be >= 0 when provided")

    log_step = max(int(log_step), 1)
    b_norm = torch.norm(b)
    b_norm = torch.clamp(b_norm, min=eps)

    if restart is None:
        x, res_hist, _, _, _ = _fgmres_cycle(
            A=A,
            b=b,
            M=M,
            x0=x0,
            b_norm=b_norm,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            log_step=log_step,
            iter_offset=0,
        )
        return x, res_hist

    x = x0
    res_hist = []
    total_iters = 0
    restarts_used = 0

    while total_iters < maxiter:
        inner_maxiter = min(restart, maxiter - total_iters)
        x, cycle_hist, cycle_iters, converged, breakdown = _fgmres_cycle(
            A=A,
            b=b,
            M=M,
            x0=x,
            b_norm=b_norm,
            tol=tol,
            maxiter=inner_maxiter,
            verbose=verbose,
            log_step=log_step,
            iter_offset=total_iters,
        )

        if not res_hist:
            res_hist.extend(cycle_hist)
        else:
            res_hist.extend(cycle_hist[1:])

        total_iters += cycle_iters

        if converged or breakdown or cycle_iters == 0:
            break

        if max_restarts is not None and restarts_used >= max_restarts:
            break

        restarts_used += 1
        if verbose:
            restart_limit = "inf" if max_restarts is None else str(max_restarts)
            print(
                f"FGMRES restart {restarts_used}/{restart_limit} "
                f"after {total_iters} iterations, relres = {res_hist[-1]:.2e}"
            )

    return x, res_hist
