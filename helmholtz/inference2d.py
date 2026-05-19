import os
import argparse
import yaml
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import unspvd_sos_Dataset, Dataconfig
from utils import Logger, Timer, plot_result_one_method
from models.cbs import ConvergentBornSeries2d
from models.shifted_laplacian import (
    FiniteDifferenceShiftedLaplacianVcycle2d,
    MatrixFreeShiftedLaplacian2d,
    ShiftedLaplacianVcycle2d,
    SparseShiftedLaplacian2d,
)
from models.model2d import MPMG_BS,PAFNO_BS,MGCFNN_BS,DeepONet_BS,EncoderSolver_BS,MGNO_BS,UNet_BS
from models.gmres import fgmres
torch.backends.cudnn.benchmark = True    
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # load configs
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if llpf finetune
    if args.llpf:
        train_cfg = cfg['finetune']
        model_cfg = cfg['model_coarser']
        weights_suffix = "weights_llpf"
    elif args.finetune:
        train_cfg = cfg['finetune']
        model_cfg = cfg['model']
        weights_suffix = "weights_finetune"
    else:
        train_cfg = cfg['train']
        model_cfg = cfg['model']
        weights_suffix = "weights"

    infer_cfg = cfg['inference']
    set_seed(infer_cfg.get("seed", cfg.get("seed")))
    data_cfg  = Dataconfig(user_data_type=model_cfg['data_type'])

    # choose preconditioner and paths
    save_name = model_cfg['name']
    dir_path = train_cfg['save_dir']
    checkpoint_override = getattr(args, "checkpoint", None)
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist, creating...")
        os.makedirs(dir_path, exist_ok=True)
    
    lprint = Logger(save_path=os.path.join(dir_path, "inference.log"))
    lprint(f"running inference with {args.config}...")
    iter_his_path = os.path.join(dir_path,f"iter_history")
    wavefield_path = os.path.join(dir_path,f"wavefield")
    out_plots_path = os.path.join(dir_path,f"out_plots")
    os.makedirs(iter_his_path, exist_ok=True)
    os.makedirs(wavefield_path, exist_ok=True)
    os.makedirs(out_plots_path, exist_ok=True)
    fdtype = torch.float64 if infer_cfg['fdtype'] == 'float64' else torch.float32
    cdtype = torch.complex128 if fdtype==torch.float64 else torch.complex64

    def checkpoint_path(model_name):
        return checkpoint_override or os.path.join(dir_path, f"{model_name}_{weights_suffix}.pth")

    if save_name == "cbs":
        model = ConvergentBornSeries2d(freq=data_cfg.freq,
                                        N=[data_cfg.size]*2,
                                        dx=data_cfg.dx,
                                        boundary_width=[model_cfg['boundary_width']]*2,
                                        boundary_type=model_cfg['boundary_layer'],
                                        fdtype=fdtype,
                                        device=device
                                        )
    elif save_name in ("sl_vcycle", "shifted_laplacian_vcycle"):
        boundary_type = model_cfg.get("boundary_layer", model_cfg.get("boundary_type", "pbl"))
        model = ShiftedLaplacianVcycle2d(
            freq=data_cfg.freq,
            N=[data_cfg.size] * 2,
            dx=data_cfg.dx,
            boundary_width=[model_cfg["boundary_width"]] * 2,
            boundary_type=boundary_type,
            shift=model_cfg.get("shift", 0.5),
            precond_discretization=model_cfg.get("precond_discretization", "fft"),
            num_levels=model_cfg.get("num_levels", None),
            min_coarse_size=model_cfg.get("min_coarse_size", 10),
            pre_smooth=model_cfg.get("pre_smooth", 2),
            post_smooth=model_cfg.get("post_smooth", 2),
            coarse_smooth=model_cfg.get("coarse_smooth", 30),
            direct_coarse=model_cfg.get("direct_coarse", True),
            smoother=model_cfg.get("smoother", "jacobi"),
            vanka_patch_size=model_cfg.get("vanka_patch_size", 2),
            vanka_stride=model_cfg.get("vanka_stride", 1),
            vanka_weight=model_cfg.get("vanka_weight", 0.7),
            jacobi_weight=model_cfg.get("jacobi_weight", 0.6),
            num_cycles=model_cfg.get("num_cycles", 1),
            fdtype=fdtype,
            device=device,
        )
    elif save_name in ("fd_sl_vcycle", "finite_difference_shifted_laplacian_vcycle"):
        boundary_type = model_cfg.get("boundary_layer", model_cfg.get("boundary_type", "pbl"))
        model = FiniteDifferenceShiftedLaplacianVcycle2d(
            freq=data_cfg.freq,
            N=[data_cfg.size] * 2,
            dx=data_cfg.dx,
            boundary_width=[model_cfg["boundary_width"]] * 2,
            boundary_type=boundary_type,
            shift=model_cfg.get("shift", 1.0),
            num_levels=model_cfg.get("num_levels", None),
            min_coarse_size=model_cfg.get("min_coarse_size", 7),
            pre_smooth=model_cfg.get("pre_smooth", 4),
            post_smooth=model_cfg.get("post_smooth", 4),
            coarse_smooth=model_cfg.get("coarse_smooth", 100),
            direct_coarse=model_cfg.get("direct_coarse", True),
            smoother=model_cfg.get("smoother", "jacobi"),
            vanka_patch_size=model_cfg.get("vanka_patch_size", 2),
            vanka_stride=model_cfg.get("vanka_stride", 2),
            vanka_weight=model_cfg.get("vanka_weight", 0.8),
            jacobi_weight=model_cfg.get("jacobi_weight", 0.6),
            num_cycles=model_cfg.get("num_cycles", 3),
            fdtype=fdtype,
            device=device,
        )
    elif save_name in ("fd_sl_matrix_free", "finite_difference_shifted_laplacian_matrix_free"):
        boundary_type = model_cfg.get("boundary_layer", model_cfg.get("boundary_type", "pbl"))
        model = MatrixFreeShiftedLaplacian2d(
            freq=data_cfg.freq,
            N=[data_cfg.size] * 2,
            dx=data_cfg.dx,
            boundary_width=[model_cfg["boundary_width"]] * 2,
            boundary_type=boundary_type,
            shift=model_cfg.get("shift", 0.25),
            num_levels=model_cfg.get("num_levels", None),
            min_coarse_size=model_cfg.get("min_coarse_size", 7),
            pre_smooth=model_cfg.get("pre_smooth", 4),
            post_smooth=model_cfg.get("post_smooth", 4),
            coarse_smooth=model_cfg.get("coarse_smooth", 100),
            direct_coarse=model_cfg.get("direct_coarse", False),
            smoother=model_cfg.get("smoother", "jacobi"),
            vanka_patch_size=model_cfg.get("vanka_patch_size", 2),
            vanka_stride=model_cfg.get("vanka_stride", 2),
            vanka_weight=model_cfg.get("vanka_weight", 0.8),
            jacobi_weight=model_cfg.get("jacobi_weight", 0.6),
            num_cycles=model_cfg.get("num_cycles", 3),
            inner_maxiter=model_cfg.get("inner_maxiter", 12),
            inner_preconditioner=model_cfg.get("inner_preconditioner", "jacobi"),
            inner_jacobi_sweeps=model_cfg.get("inner_jacobi_sweeps", 1),
            inner_vcycles=model_cfg.get("inner_vcycles", 1),
            fdtype=fdtype,
            device=device,
        )
    elif save_name in ("fd_sl_sparse", "finite_difference_shifted_laplacian_sparse"):
        boundary_type = model_cfg.get("boundary_layer", model_cfg.get("boundary_type", "pbl"))
        model = SparseShiftedLaplacian2d(
            freq=data_cfg.freq,
            N=[data_cfg.size] * 2,
            dx=data_cfg.dx,
            boundary_width=[model_cfg["boundary_width"]] * 2,
            boundary_type=boundary_type,
            shift=model_cfg.get("shift", 0.25),
            fdtype=fdtype,
            device=device,
        )
    elif save_name == 'mpmg_bs':
        model = MPMG_BS(freq=data_cfg.freq,
                        sos_mean=data_cfg.sos_mean,
                        sos_max=data_cfg.sos_max,
                        sos_min=data_cfg.sos_min,
                        N=data_cfg.size,
                        dx=data_cfg.dx,
                        boundary_width=model_cfg['boundary_width'],
                        boundary_type=model_cfg['boundary_type'],
                        u_channels_list=model_cfg['u_channels_list'],
                        a_channels_list=model_cfg['a_channels_list'],
                        mg_levels=model_cfg['mg_sweeps_lvls'],
                        fdtype=fdtype,
                        device=device)
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    elif save_name == "pafno_bs":
        model = PAFNO_BS(loss_type=model_cfg['loss_type'],
                        freq=data_cfg.freq,
                        sos_mean=data_cfg.sos_mean,
                        sos_max=data_cfg.sos_max,
                        sos_min=data_cfg.sos_min,
                        N=data_cfg.size,
                        dx=data_cfg.dx,
                        boundary_width=model_cfg['boundary_width'],
                        boundary_type=model_cfg['boundary_type'],
                        u_channels_list=model_cfg['u_channels_list'],
                        a_channels_list=model_cfg['a_channels_list'],
                        mg_levels=model_cfg['mg_sweeps_lvls'],
                        fdtype=torch.float32,
                        device=device)
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    elif save_name == "mgcfnn_bs":
        model = MGCFNN_BS(loss_type=model_cfg['loss_type'],
                        freq=data_cfg.freq,
                        sos_mean=data_cfg.sos_mean,
                        sos_max=data_cfg.sos_max,
                        sos_min=data_cfg.sos_min,
                        N=data_cfg.size,
                        dx=data_cfg.dx,
                        boundary_width=model_cfg['boundary_width'],
                        boundary_type=model_cfg['boundary_type'],
                        fdtype=torch.float32,
                        device=device)
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    elif save_name == "deeponet_bs":
        model = DeepONet_BS(loss_type=model_cfg['loss_type'],
                            freq=data_cfg.freq,
                            sos_mean=data_cfg.sos_mean,
                            sos_max=data_cfg.sos_max,
                            sos_min=data_cfg.sos_min,
                            N=data_cfg.size,
                            dx=data_cfg.dx,
                            boundary_width=model_cfg['boundary_width'],
                            boundary_type=model_cfg['boundary_type'],
                            num_basis=model_cfg.get('num_basis', 32),
                            trunk_channels=model_cfg.get('trunk_channels', 64),
                            fdtype=torch.float32,
                            device=device)
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    elif save_name == "encodersolver_bs":
        model = EncoderSolver_BS(loss_type=model_cfg['loss_type'],
                                 num_channel_u=model_cfg.get('num_channel_u', 32),
                                 num_levels=model_cfg.get('num_levels', 4),
                                 implicit=model_cfg.get('implicit', True),
                                 freq=data_cfg.freq,
                                 sos_mean=data_cfg.sos_mean,
                                 sos_max=data_cfg.sos_max,
                                 sos_min=data_cfg.sos_min,
                                 N=data_cfg.size,
                                 dx=data_cfg.dx,
                                 boundary_width=model_cfg['boundary_width'],
                                 boundary_type=model_cfg['boundary_type'],
                                 fdtype=torch.float32,
                                 device=device)
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    elif save_name == "mgno_bs":
        model = MGNO_BS(loss_type=model_cfg['loss_type'],
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
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    elif save_name == "unet_bs":
        model = UNet_BS(loss_type=model_cfg['loss_type'],
                        num_channel_u=model_cfg.get('num_channel_u', 32),
                        num_levels=model_cfg.get('num_levels', 4),
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
        mg_path = checkpoint_path(save_name)
        ckpoint = torch.load(mg_path, weights_only=False)
        model.load_state_dict(ckpoint)
        model.eval()
        if args.optimize:
            model.precond = torch.compile(model.precond, mode="max-autotune")
    else:
        raise ValueError(f"model '{save_name}' not supported")
    model = model#.to(memory_format=torch.channels_last)
    # prepare dataset & loader
    test_point = infer_cfg['test_point']
    dataset = unspvd_sos_Dataset(
                                sos_path=data_cfg.dir_path,
                                data_size=data_cfg.size,
                                data_type=data_cfg.data_type,
                                data_num=infer_cfg['num_test'],
                                test_point = test_point,
                                test=True
                                )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # build point source

    src = torch.zeros(1, 1, data_cfg.size, data_cfg.size, device=device, dtype=fdtype)
    src = src#.to(memory_format=torch.channels_last)
    src_point_loc = [data_cfg.size//10, data_cfg.size//2]
    src[..., src_point_loc[0], src_point_loc[1]] = 1.0
    # src[..., src_point_loc[0], src_point_loc[1]//2] = 1.0
    # src[..., src_point_loc[0], src_point_loc[1]*3//2] = 1.0

    # src[..., src_point_loc[0]:src_point_loc[0]+20, src_point_loc[1]-:src_point_loc[1]+128] = 1.0
    # src = torch.randn(1,1,480,480, device=device, dtype=torch.float64)
    # timing events
    timer = Timer(device=device)


    num_iters = infer_cfg['max_num_iters']
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
    drop_first_timing = bool(infer_cfg.get("drop_first_timing", False))
    cbs_krylov_preconditioner = str(
        infer_cfg.get("cbs_krylov_preconditioner", model_cfg.get("cbs_krylov_preconditioner", "gamma_green"))
    ).lower()
    if cbs_krylov_preconditioner in ("gamma", "cbs"):
        cbs_krylov_preconditioner = "gamma_green"
    if cbs_krylov_preconditioner in ("born", "green", "g_eta", "shifted_laplacian_reference"):
        cbs_krylov_preconditioner = "green"
    if cbs_krylov_preconditioner not in ("gamma_green", "green"):
        raise ValueError("cbs_krylov_preconditioner must be 'gamma_green' or 'green'")
    loss_type = model_cfg.get('loss_type', None)
    has_precond = hasattr(model, 'precond')
    has_cbs_kernel_precond = (save_name == "cbs")
    has_sl_vcycle_precond = hasattr(model, "apply_preconditioner")

    if use_krylov:
        if loss_type in (2, 3) and (not has_precond or not hasattr(model, 'G') or not hasattr(model, 'dx')):
            raise ValueError("krylov solver with loss_type 2/3 requires model.G and model.dx")
        if has_cbs_kernel_precond and (not hasattr(model, 'G') or not hasattr(model, 'dx')):
            raise ValueError("cbs krylov kernel requires model.G and model.dx")
        if has_sl_vcycle_precond:
            precond_name = getattr(model, "preconditioner_kind", "shifted_laplacian_vcycle")
        elif has_precond:
            precond_name = "model.precond"
        elif has_cbs_kernel_precond:
            precond_name = f"cbs_kernel:{cbs_krylov_preconditioner}"
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
            if has_sl_vcycle_precond:
                return model.apply_preconditioner(r).to(cdtype)

            if has_cbs_kernel_precond:
                if not hasattr(model, 'gamma') or not hasattr(model, 'g0_tilde'):
                    raise RuntimeError("cbs kernel preconditioner is not ready; call model.setup(...) first")
                z = model.G(r / (model.dx ** 2), model.g0_tilde)
                if cbs_krylov_preconditioner == "gamma_green":
                    z = model.gamma * z
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
                if z.dim() == 4:
                    z = z.unsqueeze(1)
                return z.to(cdtype)
            z = torch.view_as_complex(z.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
            return z.to(cdtype)
    elif has_sl_vcycle_precond:
        raise ValueError("sl_vcycle is a Krylov preconditioner; set inference.solver_type: krylov")

    iter_num_list = []
    iter_time_list = []
    total_tested = 0
    success_count = 0
    # warm up
    lprint(">>>start warm up(the first run will be slow if use torch.compile)... ")
    with torch.inference_mode():
        sos = dataset[:1].to(device)#.to(memory_format=torch.channels_last)
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
            u = torch.zeros_like(model.b,dtype=cdtype)#.to(memory_format=torch.channels_last)
            for j in range(10):
                u = model(u)
    lprint(">>>start inference... ")    
    # inference loop
    with torch.inference_mode():
    # with torch.no_grad():
        for i, sos in enumerate(dataloader):
            idx = test_point + i
            lprint(f"------------------- Solving {idx}th SOS -------------------")
            total_tested += 1
            sos = sos.to(device)#.to(memory_format=torch.channels_last)
            # t1 = timer.record()
            model.setup(sos=sos, src=src)
            norm_src = torch.norm(model.b)
            u = torch.zeros_like(model.b,dtype=cdtype)#.to(memory_format=torch.channels_last)
            
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
                    continue
                # Since residual computing nearly double cbs's solve time,
                # In order to fairly compare, we need to disable residual computing and record time again
                current_iter_times = j+1
                t1 = timer.record()
                model.setup(sos=sos, src=src)
                norm_src = torch.norm(model.b)
                u = torch.zeros_like(model.b,dtype=cdtype)#.to(memory_format=torch.channels_last)
                for j in range(current_iter_times):
                    u = model(u)
                t2 = timer.record()
                total_time = timer.elapsed_time(t1, t2)
                current_iter_num = len(iter_history)

            if not converged:
                continue

            iter_num_list.append(current_iter_num)
            iter_time_list.append(total_time)
            lprint(f"Total time: {total_time:.4f} s" if total_time else "")
            plot_result_one_method( u,
                                    sos,
                                    iter_history,
                                    src_position=src_point_loc,
                                    id=idx,
                                    boundary_width=model_cfg['boundary_width'],
                                    plot_pml=True,
                                    solve_time=total_time,
                                    save_dir=out_plots_path,
                                    save_name=save_name)
            iter_history = {"iter_log":iter_history,'iter_time':total_time}
            torch.save(iter_history,f=os.path.join(iter_his_path,f"iter_history_{save_name}_{idx}.pth"))
            torch.save(u.cpu(),f=os.path.join(wavefield_path,f"u_{save_name}_{idx}.pth"))
        lprint('-------------------------------------')
        avg_iters = np.mean(iter_num_list) if iter_num_list else float("nan")
        timing_samples = iter_time_list[1:] if drop_first_timing and len(iter_time_list) > 1 else iter_time_list
        avg_time = np.mean(timing_samples) if timing_samples else float("nan")
        lprint(f'Average iterations to reach tol:{avg_iters:.1f}')
        lprint(f'Average time to reach tol:{avg_time:.2e}s (drop_first_timing={drop_first_timing})')
        lprint(f"Convergence rate: {success_count}/{total_tested} ({100*success_count/total_tested:.1f}%)")
        lprint(f"save inference results to {dir_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference inference")

    parser.add_argument(
        "--config","-c",
        default='./configs/2d/cbs.yaml',
        type=str,
        help="Path to the YAML config file."
    )
    parser.add_argument(
        "--llpf","-llpf",
        action='store_true',
        help="If set, use model_llpf configuration."
    )
    parser.add_argument(
        "--finetune","-f",
        action='store_true',
        help="If set, use finetune configuration."
    )
    parser.add_argument(
        "--optimize","-o",
        action='store_true',
        help="If set, use torch.compile")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit model checkpoint path for learned preconditioners.")
    args = parser.parse_args()
    main(args)
