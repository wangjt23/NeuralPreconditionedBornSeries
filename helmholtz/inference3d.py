# Resolve repository imports when this file is run directly.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helmholtz.runtime import add_runtime_arguments, configure_runtime, configure_data
import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from helmholtz.dataloader import unspvd_sos_Dataset
from helmholtz.utils import Logger, Timer, plot_result_one_method_3d
from helmholtz.models.cbs import ConvergentBornSeries3d
from helmholtz.models.model3d import MGNO_BS_3D
from helmholtz.models.gmres import fgmres



def main(args):
    # load configs
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = configure_runtime(cfg, args)

    if args.finetune:
        train_cfg = cfg['finetune']
        model_cfg = cfg['model']
        weights_suffix = "weights_finetune"
    else:
        train_cfg = cfg['train']
        model_cfg = cfg['model']
        weights_suffix = "weights"

    infer_cfg = cfg['inference']
    data_cfg = configure_data(cfg, model_cfg, args)

    # choose preconditioner and paths
    save_name = model_cfg['name']
    dir_path = train_cfg['save_dir']
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist, creating...")
        os.makedirs(dir_path, exist_ok=True)

    lprint = Logger(save_path=os.path.join(dir_path, "inference.log"))
    lprint(f"running 3D inference with {args.config}...")
    iter_his_path = os.path.join(dir_path, "iter_history")
    wavefield_path = os.path.join(dir_path, "wavefield")
    out_plots_path = os.path.join(dir_path, "out_plots")
    os.makedirs(iter_his_path, exist_ok=True)
    os.makedirs(wavefield_path, exist_ok=True)
    os.makedirs(out_plots_path, exist_ok=True)

    fdtype = torch.float64 if infer_cfg['fdtype'] == 'float64' else torch.float32
    cdtype = torch.complex128 if fdtype == torch.float64 else torch.complex64

    if save_name == "cbs":
        model = ConvergentBornSeries3d(freq=data_cfg.freq,
                                       N=[data_cfg.size] * 3,
                                       dx=data_cfg.dx,
                                       boundary_width=[model_cfg['boundary_width']] * 3,
                                       boundary_type=model_cfg['boundary_type'],
                                       fdtype=fdtype,
                                       device=device)
    elif save_name == "mgno_bs_3d":
        model = MGNO_BS_3D(loss_type=model_cfg['loss_type'],
                           num_channel_u=model_cfg.get('num_channel_u', 32),
                           num_iteration=model_cfg.get('num_iteration', None),
                           activation=model_cfg.get('activation', 'gelu'),
                           freq=data_cfg.freq,
                           sos_mean=data_cfg.sos_mean,
                           sos_max=data_cfg.sos_max,
                           sos_min=data_cfg.sos_min,
                           N=data_cfg.size,
                           dx=data_cfg.dx,
                           boundary_width=model_cfg['boundary_width'],
                           boundary_type=model_cfg['boundary_type'],
                           fdtype=fdtype,
                           device=device)
        mg_path = args.checkpoint or os.path.join(dir_path, f"{save_name}_{weights_suffix}.pth")
        ckpoint = torch.load(mg_path, weights_only=True, map_location=device)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    else:
        raise ValueError(f"model '{save_name}' not supported")

    # prepare dataset & loader (B,1,D,H,W)
    test_point = infer_cfg['test_point']
    dataset = unspvd_sos_Dataset(
        sos_path=data_cfg.dir_path,
        data_size=data_cfg.size,
        data_type=data_cfg.data_type,
        data_num=infer_cfg['num_test'],
        test_point=test_point,
        test=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # build 3D point source
    src = torch.zeros(1, 1, data_cfg.size, data_cfg.size, data_cfg.size, device=device, dtype=fdtype)
    src_point_loc = [data_cfg.size // 10, data_cfg.size // 2, data_cfg.size // 2]
    src[..., src_point_loc[0], src_point_loc[1], src_point_loc[2]] = 1.0

    timer = Timer(device=device)

    num_iters = int(infer_cfg['max_num_iters'])
    if num_iters < 1:
        raise ValueError("inference.max_num_iters must be positive")
    tol = float(infer_cfg['tol'])
    print_step = int(infer_cfg.get('print_step', 10))
    solver_type = str(infer_cfg.get('solver_type', 'fixed_point')).lower()
    use_krylov = (solver_type == 'krylov')
    krylov_maxiter = int(infer_cfg.get('krylov_maxiter', num_iters))
    krylov_restart = infer_cfg.get('krylov_restart')
    krylov_restart = None if krylov_restart is None else int(krylov_restart)
    krylov_max_restarts = infer_cfg.get('krylov_max_restarts')
    krylov_max_restarts = None if krylov_max_restarts is None else int(krylov_max_restarts)
    krylov_log_step = int(infer_cfg.get('krylov_log_step', print_step))
    krylov_verbose = bool(infer_cfg.get('krylov_verbose', True))
    loss_type = model_cfg.get('loss_type', None)
    has_precond = hasattr(model, 'precond') and getattr(model, 'precond') is not None
    has_cbs_kernel_precond = (save_name == "cbs")

    if use_krylov:
        if loss_type in (2, 3) and (not has_precond or not hasattr(model, 'G') or not hasattr(model, 'dx')):
            raise ValueError("krylov solver with loss_type 2/3 requires model.G and model.dx")
        if has_cbs_kernel_precond and (not hasattr(model, 'G') or not hasattr(model, 'dx')):
            raise ValueError("cbs krylov kernel requires model.G and model.dx")
        if has_precond:
            precond_name = "model.precond"
        elif has_cbs_kernel_precond:
            precond_name = "cbs_kernel"
        else:
            precond_name = "identity"
        lprint(
            "krylov config: "
            f"preconditioner={precond_name}, "
            f"loss_type={loss_type}, "
            f"maxiter={krylov_maxiter}, restart={krylov_restart}, "
            f"max_restarts={krylov_max_restarts}, log_step={krylov_log_step}, "
            f"verbose={krylov_verbose}"
        )

        def _precondition_residual(r):
            if has_cbs_kernel_precond:
                if not hasattr(model, 'gamma') or not hasattr(model, 'g0_tilde'):
                    raise RuntimeError("cbs kernel preconditioner is not ready; call model.setup(...) first")
                z = model.gamma * model.G(r / (model.dx ** 2), model.g0_tilde)
                return z.to(cdtype)

            if not has_precond:
                return r

            # Keep the Krylov preconditioner input aligned with the residual
            # representation used during training for loss_type 2/3.
            if loss_type in (2, 3):
                r = model.G(r / (model.dx ** 2))

            r_ri = torch.cat([r.real, r.imag], dim=1).to(torch.float32)
            z = model.precond(r_ri, model.a_list, model.s_list)
            if torch.is_complex(z):
                if z.dim() == 5:
                    z = z.unsqueeze(1)
                return z.to(cdtype)
            z = torch.view_as_complex(z.permute(0, 2, 3, 4, 1).contiguous()).unsqueeze(1)
            return z.to(cdtype)

    iter_num_list = []
    iter_time_list = []
    total_tested = 0
    success_count = 0
    # warm up
    lprint(">>>start warm up(the first run will be slow if use torch.compile)... ")
    with torch.inference_mode():
        sos = dataset[:1].to(device=device, dtype=fdtype)
        model.setup(sos=sos, src=src)
        if use_krylov:
            b = model.b.to(cdtype)

            def _A_warmup(x):
                return model.A_op(x)

            def _M_warmup(r):
                return _precondition_residual(r)

            _ = fgmres(
                A=_A_warmup,
                b=b,
                M=_M_warmup,
                x0=None,
                tol=tol,
                maxiter=min(5, krylov_maxiter),
                restart=krylov_restart,
                max_restarts=krylov_max_restarts,
                verbose=False,
                log_step=krylov_log_step,
            )
        else:
            u = torch.zeros_like(model.b, dtype=cdtype)
            for j in range(10):
                u = model(u)
    lprint(">>>start inference... ")
    # inference loop
    with torch.inference_mode():
        for i, sos in enumerate(dataloader):
            idx = test_point + i
            lprint(f"------------------- Solving {idx}th SOS (3D) -------------------")
            total_tested += 1
            sos = sos.to(device=device, dtype=fdtype)
            model.setup(sos=sos, src=src)
            norm_src = torch.norm(model.b)
            u = torch.zeros_like(model.b, dtype=cdtype)

            iter_history = []
            converged = False
            if use_krylov:
                b = model.b.to(cdtype)

                def A_func(x):
                    return model.A_op(x)

                def M_func(r):
                    return _precondition_residual(r)

                t1 = timer.record()
                u, gmres_hist = fgmres(
                    A=A_func,
                    b=b,
                    M=M_func,
                    x0=None,
                    tol=tol,
                    maxiter=krylov_maxiter,
                    restart=krylov_restart,
                    max_restarts=krylov_max_restarts,
                    verbose=krylov_verbose,
                    log_step=krylov_log_step,
                )
                t2 = timer.record()
                total_time = timer.elapsed_time(t1, t2)

                if len(gmres_hist) > 0:
                    iter_history = gmres_hist

                rel_res = torch.norm(model.b - model.A_op(u)) / norm_src
                gmres_iters = max(len(iter_history) - 1, 0)
                current_iter_num = gmres_iters
                lprint(f"FGMRES iters: {gmres_iters:5}, Rel Residual: {rel_res.item():.4e}")
                if rel_res.item() < tol:
                    success_count += 1
                    converged = True
                elif rel_res.item() > 99:
                    lprint(f"FGMRES iters: {gmres_iters:5}, Rel Residual: {rel_res.item():.4e} > 99, diverged! Skipping to next case.")
            else:
                for j in range(num_iters):
                    u = model(u)
                    rel_res = torch.norm(model.b - model.A_op(u)) / norm_src
                    iter_history.append(rel_res.item())
                    if (j + 1) % print_step == 0:
                        lprint(f"Iter: {j+1:5}, Rel Residual: {rel_res.item():.4e}")
                    if rel_res.item() < tol:
                        lprint(f"Iter: {j+1:5}, Rel Residual: {rel_res.item():.4e}")
                        success_count += 1
                        converged = True
                        break
                    if rel_res.item() > 99:
                        lprint(f"Iter: {j+1:5}, Rel Residual: {rel_res.item():.4e} > 99, diverged! Skipping to next case.")
                        break
                if not converged:
                    lprint(f"Did not converge; final residual: {rel_res.item():.4e}")
                    torch.save({"iter_log": iter_history, "converged": False,
                                "final_residual": rel_res.item()},
                               os.path.join(iter_his_path, f"iter_history_{save_name}_{idx}.pth"))
                    continue
                # Since residual computing nearly double cbs's solve time,
                # In order to fairly compare, we need to disable residual computing and record time again
                current_iter_times = j + 1
                t1 = timer.record()
                model.setup(sos=sos, src=src)
                norm_src = torch.norm(model.b)
                u = torch.zeros_like(model.b, dtype=cdtype)
                for j in range(current_iter_times):
                    u = model(u)
                t2 = timer.record()
                total_time = timer.elapsed_time(t1, t2)
                current_iter_num = len(iter_history)

            if not converged:
                lprint(f"Did not converge; final residual: {rel_res.item():.4e}")
                torch.save({"iter_log": iter_history, "converged": False,
                            "final_residual": rel_res.item()},
                           os.path.join(iter_his_path, f"iter_history_{save_name}_{idx}.pth"))
                continue

            iter_num_list.append(current_iter_num)
            iter_time_list.append(total_time)
            lprint(f"Total time: {total_time:.4f} s" if total_time else "")
            if not args.no_plots:
                plot_result_one_method_3d(u,
                                          sos,
                                          iter_history,
                                          id=idx,
                                          boundary_width=model_cfg['boundary_width'],
                                          plot_pml=True,
                                          solve_time=total_time,
                                          save_dir=out_plots_path,
                                          save_name=save_name)
            iter_history = {"iter_log": iter_history, "iter_time": total_time, "converged": True, "final_residual": rel_res.item()}
            torch.save(iter_history, f=os.path.join(iter_his_path, f"iter_history_{save_name}_{idx}.pth"))
            torch.save(u.cpu(), f=os.path.join(wavefield_path, f"u_{save_name}_{idx}.pth"))
        lprint('-------------------------------------')
        mean_iters = np.mean(iter_num_list) if iter_num_list else float('nan')
        lprint(f'Average iterations to reach tol: {mean_iters:.1f}')
        mean_time = np.mean(iter_time_list[1:] or iter_time_list) if iter_time_list else float('nan')
        lprint(f'Average time to reach tol: {mean_time:.2e}s')
        lprint(f"Convergence rate: {success_count}/{total_tested} ({100*success_count/max(total_tested, 1):.1f}%)")
        lprint(f"save inference results to {dir_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D inference")
    parser.add_argument(
        "--config", "-c",
        default='helmholtz/configs/3d/mgno_bs_pre.yaml',
        type=str,
        help="Path to the YAML config file."
    )
    parser.add_argument(
        "--finetune", "-f",
        action='store_true',
        help="If set, use finetune configuration."
    )
    parser.add_argument(
        "--optimize", "-o",
        action='store_true',
        help="If set, use torch.compile")
    add_runtime_arguments(parser, inference=True)
    args = parser.parse_args()
    main(args)
