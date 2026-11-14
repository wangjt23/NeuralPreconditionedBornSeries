"""Warm-started solver-aware training, with FP64 outer and FP32 neural kernels.

Train through a short real-inner-product FGMRES cycle. This is a new objective
and reset optimizer schedule, never an exact continuation of a previous run.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.mixed_precision_transport import MixedPrecisionPreconditioner, high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, capture_rng, cpu_tree, dataset_fingerprint, make_operator, restore_rng, save_checkpoint, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import measured_solver, write_csv
from rte.train_collision_transport_ensemble import iteration_score, material_split
from rte.train_neural_transport import residual_sample
from rte.spectral_collision_transport import make_resolution_model


def short_fgmres(operator, model, residual, steps=3):
    """Differentiable real-stacked Arnoldi; two-pass MGS and tiny relative ridge.

    All outer vectors/inner products are FP64; only model applications are FP32.
    The m-by-m normal solve has ridge 1e-12 * mean(diag(H^T H)); no teacher needed.
    """
    residual = residual.to(torch.complex128)
    b = len(residual)
    norm = residual.abs().square().flatten(1).sum(1).sqrt().clamp_min(1e-30)
    normalized = residual/norm[:, None, None, None]
    vectors, corrections, columns = [normalized], [], []
    shape = (b, 1, 1, 1)
    for j in range(steps):
        z = model(vectors[j].to(torch.complex64)).to(torch.complex128)
        corrections.append(z)
        w = operator(z)
        coefficients = [torch.zeros(b, dtype=torch.float64, device=w.device) for _ in range(j+1)]
        for _ in range(2):
            for i, v in enumerate(vectors):
                h = (v.conj()*w).real.flatten(1).sum(1)
                coefficients[i] = coefficients[i]+h
                w = w-h.reshape(shape)*v
        next_norm = w.abs().square().flatten(1).sum(1).clamp_min(1e-40).sqrt()
        vectors.append(w/next_norm.reshape(shape))
        column = torch.stack(coefficients+[next_norm], -1)
        columns.append(torch.nn.functional.pad(column, (0, steps-j-1)))
    hessenberg = torch.stack(columns, -1)
    normal = hessenberg.transpose(-2, -1)@hessenberg
    ridge = normal.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)*1e-12
    identity = torch.eye(steps, dtype=normal.dtype, device=normal.device)
    y = torch.linalg.solve(normal+ridge[:, None, None]*identity, hessenberg[:, 0, :].unsqueeze(-1)).squeeze(-1)
    solution = sum(z*y[:, j].reshape(shape) for j, z in enumerate(corrections))
    relative_squared = (normalized-operator(solution)).abs().square().flatten(1).sum(1)
    return relative_squared, solution*norm.reshape(shape)


def training_loss(high, model, residual, steps):
    # Coefficient-network outputs are reused within the short solve, with their
    # graph intact; inference still uses the original model's tested cache.
    original_fields = model.fields
    fields = original_fields()
    model.fields = lambda: fields
    try:
        squared, _ = short_fgmres(high.A_op, model, residual, steps)
        return squared.clamp_min(1e-12).log().mean(), squared.mean()
    finally:
        model.fields = original_fields


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    initial = torch.load(args.warm_start, map_location='cpu', weights_only=True)
    saved = initial['args']
    for key in ('model', 'depth', 'width', 'base', 'no_mixing', 'no_spectral', 'n_angles', 'g'):
        setattr(args, key, saved[key])
    upgrade = args.spectral_upgrade or getattr(args, 'characteristic_upgrade', False)
    if upgrade:
        if args.model != 'ct':
            raise ValueError('spectral upgrade requires a CT warm start')
        args.model = 'cct' if getattr(args, 'characteristic_upgrade', False) else 'sct'
    args.train_size, args.val_size, args.test_size = args.train_size, args.val_size, 0
    args.dataset_dir, args.regimes = '', 'mixed'
    cfg = ExperimentConfig(size=args.size, n_angles=args.n_angles, anisotropy_g=args.g,
                           regimes='mixed', batch_size=args.batch_size, eta_scale=0.)
    train, validation = [material_split(args, cfg, name) for name in ('train', 'validation')]
    fingerprints = {name: dataset_fingerprint(data) for name, data in [('train', train), ('validation', validation)]}
    ids = torch.arange(min(args.batch_size, args.train_size))
    q = source_from_data(cfg, train, ids, device)
    low = make_operator(cfg, train['sigma_t'][ids], train['sigma_s'][ids], q, device)
    high = high_precision_copy(low)
    model = make_resolution_model(args, low).to(device)
    if upgrade:
        model.backbone.load_state_dict(initial['model'], strict=True)
    else:
        model.load_state_dict(initial['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.01)
    generator = torch.Generator().manual_seed(args.seed+9300)
    history, best, best_state, best_step, start_step = [], None, None, 0, 0
    permutation, offset, elapsed = torch.empty(0, dtype=torch.long), 0, 0.
    out = Path(args.output)
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
        clean = lambda a: {k:v for k,v in {'spectral_upgrade': False, 'characteristic_upgrade': False, **a}.items()
            if k not in ('resume', 'stop_after', 'output')}
        if clean(vars(args)) != clean(resumed['args']) or fingerprints != resumed['fingerprints']:
            raise ValueError('resume configuration/data mismatch')
        model.load_state_dict(resumed['model']); optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler']); restore_rng(resumed['rng'], generator, device)
        history, best, best_state, best_step = [resumed[k] for k in ('history', 'best', 'best_state', 'best_step')]
        permutation, offset, elapsed, start_step = [resumed[k] for k in ('permutation', 'offset', 'elapsed_seconds', 'step')]
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out/'warm_start.pt', initial)
    for name, data in [('train', train), ('validation', validation)]:
        save_checkpoint(out/f'{name}.pt', dict(schema=1, data=data, fingerprint=fingerprints[name], g=args.g))
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), fingerprints=fingerprints,
        warm_start_sha256=hashlib.sha256(Path(args.warm_start).read_bytes()).hexdigest(),
        optimizer_scheduler='reset for new objective; exact subsequent resume supported',
        loss='mean log squared true relative residual after short FGMRES',
        precision='FP64 physical/Arnoldi, FP32 neural kernels',
        parameters=sum(p.numel() for p in model.parameters()), torch=str(torch.__version__)), indent=2)+'\n')
    start = time.perf_counter()

    @torch.no_grad()
    def assess(step, loss=None):
        nonlocal best, best_state, best_step
        rows, short_losses = [], []
        model.eval()
        for index in range(min(args.selection_count, len(validation['sigma_t']))):
            setup_batch(cfg, low, validation, torch.tensor([index]), device)
            high.setup(low.sigma_t.double(), low.sigma_s.double(), low.source.to(torch.complex128))
            short_losses.append(float(short_fgmres(high.A_op, model, high.source, args.krylov_steps)[0].mean()))
            method = MixedPrecisionPreconditioner(high, low, model)
            rows.append(measured_solver(high, method, high.source, args))
        score = iteration_score(rows, args.tol)
        if best is None or score < best:
            best, best_state, best_step = score, cpu_tree(model.state_dict()), step
            save_checkpoint(out/f'selected_step_{step}.pt', dict(args=vars(args), model=best_state,
                selected_step=step, fingerprints=fingerprints))
        history.append(dict(step=step, train_log_loss=loss, validation_short_squared=sum(short_losses)/len(short_losses),
            validation_failures=score[0], validation_iterations=score[1], validation_final=score[2], selected_step=best_step))
        write_csv(out/'history.csv', history)
        print(f'{args.model}{args.depth} KRYLOV step={step} short={history[-1]["validation_short_squared"]:.5g} iters={score[1]} failures={score[0]} selected={best_step}', flush=True)

    if not history:
        assess(0)
    for step in range(start_step+1, args.steps+1):
        model.train()
        if offset >= len(permutation):
            permutation, offset = torch.randperm(args.train_size, generator=generator), 0
        ids = permutation[offset:offset+args.batch_size]
        offset += len(ids)
        source = setup_batch(cfg, low, train, ids, device)
        high.setup(low.sigma_t.double(), low.sigma_s.double(), source.to(torch.complex128))
        residual = residual_sample(low, source, generator, .2)
        choose = (torch.rand(len(source), generator=generator) < args.source_fraction).to(device)
        residual = torch.where(choose[:, None, None, None], source, residual)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_loss(high, model, residual, args.krylov_steps)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        stop = bool(args.stop_after and step >= args.stop_after and step < args.steps)
        if step % args.eval_every == 0 or step == args.steps or stop:
            assess(step, float(loss.detach()))
            save_checkpoint(out/'best.pt', dict(args=vars(args), model=best_state, selected_step=best_step, fingerprints=fingerprints))
            save_checkpoint(out/'last.pt', dict(args=vars(args), model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                scheduler=scheduler.state_dict(), rng=capture_rng(generator, device), history=history, best=best, best_state=best_state,
                best_step=best_step, step=step, permutation=permutation, offset=offset, fingerprints=fingerprints,
                elapsed_seconds=elapsed+time.perf_counter()-start))
        if stop:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-start', required=True)
    parser.add_argument('--spectral-upgrade', action='store_true')
    parser.add_argument('--characteristic-upgrade', action='store_true')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=402)
    parser.add_argument('--train-size', type=int, default=512)
    parser.add_argument('--val-size', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--krylov-steps', type=int, default=3)
    parser.add_argument('--source-fraction', type=float, default=.75)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--selection-count', type=int, default=8)
    parser.add_argument('--solver-steps', type=int, default=60)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', default='')
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.spectral_upgrade and args.characteristic_upgrade:
        parser.error('choose only one upgrade')
    if min(args.size, args.train_size, args.val_size, args.batch_size, args.steps, args.krylov_steps, args.selection_count, args.eval_every) < 1:
        parser.error('positive counts required')
    if not 0 <= args.source_fraction <= 1:
        parser.error('source fraction must be in [0,1]')
    run(args)
