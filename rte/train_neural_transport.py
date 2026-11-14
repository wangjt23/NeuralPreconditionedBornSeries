"""Bounded architecture pilots on fresh coefficient data; validation only.

NTO and CT controls share the sampler, data, loss, update horizon and solver.
No test split is generated. Checkpoints include exact update-boundary state.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.neural_transport_operator import NeuralTransportOperator
from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import (
    ExperimentConfig, capture_rng, cpu_tree, dataset_fingerprint, make_operator,
    restore_rng, save_checkpoint, setup_batch, source_from_data,
)
from rte.run_residual_subspace_experiment import measured_solver, synchronized, write_csv
from rte.train_collision_transport_ensemble import iteration_score, material_split, sample_residual


def make_model(args, op):
    if args.model == 'ct':
        return CollisionTransportPreconditioner(op, depth=args.depth, learned=True, width=12)
    return NeuralTransportOperator(op, depth=args.depth, width=args.width,
                                   mixing=not args.no_mixing, spectral=not args.no_spectral,
                                   base=getattr(args, 'base', 'reference'))


def residual_sample(op, source, generator, broad_fraction):
    residual = sample_residual(op, source, generator, 1/3)
    with torch.no_grad():
        error = torch.randn(source.shape, dtype=source.dtype, generator=generator).to(source.device)
        broad = op.A_op(error)
        choose = (torch.rand(len(source), generator=generator) < broad_fraction).to(source.device)
    return torch.where(choose[:, None, None, None], broad, residual)


def objective(op, model, residual, rollout_weight):
    correction = model(residual)
    loss = physical_residual_loss(op.A_op, residual, correction)
    if rollout_weight:
        error = residual - op.A_op(correction)
        error2 = error - op.A_op(model(error))
        numerator = error2.abs().square().flatten(1).sum(1)
        denominator = residual.abs().square().flatten(1).sum(1).clamp_min(1e-30)
        loss = loss + rollout_weight * (numerator / denominator).mean()
    return loss


@torch.no_grad()
def validate(args, cfg, data, op, model, device):
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 9400)
    total = 0.
    for ids in torch.arange(len(data['sigma_t'])).split(args.batch_size):
        source = setup_batch(cfg, op, data, ids, device)
        residual = residual_sample(op, source, generator, args.broad_fraction)
        total += len(ids)*float(objective(op, model, residual, args.rollout_weight))
    rows = []
    for index in range(min(args.selection_count, len(data['sigma_t']))):
        source = setup_batch(cfg, op, data, torch.tensor([index]), device)
        rows.append(measured_solver(op, model, source, args))
    return total/len(data['sigma_t']), iteration_score(rows, args.tol)


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = ExperimentConfig(size=args.size, n_angles=args.n_angles, anisotropy_g=args.g,
                           regimes='mixed', batch_size=args.batch_size, eta_scale=0.)
    args.regimes, args.dataset_dir = 'mixed', ''
    train = material_split(args, cfg, 'train')
    validation = material_split(args, cfg, 'validation')
    fingerprints = {name: dataset_fingerprint(data) for name, data in [('train', train), ('validation', validation)]}
    ids = torch.arange(min(args.batch_size, args.train_size))
    q = source_from_data(cfg, train, ids, device)
    op = make_operator(cfg, train['sigma_t'][ids], train['sigma_s'][ids], q, device)
    model = make_model(args, op).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.01)
    generator = torch.Generator().manual_seed(args.seed+9300)
    out = Path(args.output)
    history, best, best_state, best_step, start_step, elapsed = [], None, None, 0, 0, 0.
    permutation, offset = torch.empty(0, dtype=torch.long), 0
    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=True)
        ignored = {'output', 'resume', 'stop_after'}
        clean = lambda values: {k: v for k, v in values.items() if k not in ignored}
        if clean(vars(args)) != clean(payload['args']) or fingerprints != payload['fingerprints']:
            raise ValueError('resume configuration/data changed')
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        restore_rng(payload['rng'], generator, device)
        history, best, best_state = payload['history'], payload['best'], payload['best_state']
        best_step, start_step, elapsed = payload['best_step'], payload['step'], payload['elapsed_seconds']
        permutation, offset = payload['permutation'], payload['offset']
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(args=vars(args), fingerprints=fingerprints, parameters=sum(p.numel() for p in model.parameters()),
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    scope='architecture pilot: train/validation only, no confirmatory test',
                    loss='L1 + rollout_weight * ||E(E(r))||^2/||r||^2',
                    sampler='broad_fraction full-spectrum A e; remainder TSA trajectory/smooth A e (2:1)',
                    checkpoint_selection='validation failures, iterations, true residual; timing separately')
    (out/'settings.json').write_text(json.dumps(metadata, indent=2)+'\n')
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def assess(step, train_loss=None):
        nonlocal best, best_state, best_step
        loss, score = validate(args, cfg, validation, op, model, device)
        if best is None or score < best:
            best, best_state, best_step = score, cpu_tree(model.state_dict()), step
            save_checkpoint(out/f'selected_step_{step}.pt', dict(args=vars(args), model=best_state,
                selected_step=step, fingerprints=fingerprints))
        history.append(dict(step=step, train_batch_objective=train_loss, validation_objective=loss,
                            validation_failures=score[0], validation_iterations=score[1],
                            validation_final=score[2], selected_step=best_step))
        write_csv(out/'history.csv', history)
        print(f'{args.model}{args.depth} step={step} val={loss:.5g} iters={score[1]:.4f} failures={score[0]} selected={best_step}', flush=True)

    if not history:
        assess(0)
    for step in range(start_step+1, args.steps+1):
        model.train()
        if offset >= len(permutation):
            permutation, offset = torch.randperm(args.train_size, generator=generator), 0
        ids = permutation[offset:offset+args.batch_size]
        offset += len(ids)
        q = setup_batch(cfg, op, train, ids, device)
        residual = residual_sample(op, q, generator, args.broad_fraction)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(op, model, residual, args.rollout_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite training loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); scheduler.step()
        stop = bool(args.stop_after and step >= args.stop_after and step < args.steps)
        if step % args.eval_every == 0 or step == args.steps or stop:
            assess(step, float(loss.detach()))
            synchronized(device)
            save_checkpoint(out/'last.pt', dict(args=vars(args), model=cpu_tree(model.state_dict()),
                optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                rng=capture_rng(generator, device), step=step, history=history, best=best,
                best_state=best_state, best_step=best_step, permutation=permutation, offset=offset,
                fingerprints=fingerprints, elapsed_seconds=elapsed+time.perf_counter()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
            save_checkpoint(out/'best.pt', dict(args=vars(args), model=best_state, selected_step=best_step, fingerprints=fingerprints))
        if stop:
            return dict(status='paused', step=step, selected_step=best_step)
    return dict(status='complete', step=args.steps, selected_step=best_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['nto', 'ct'], default='nto')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--base', choices=['reference', 'hss'], default='hss')
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--no-mixing', action='store_true')
    parser.add_argument('--no-spectral', action='store_true')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--n-angles', type=int, default=12)
    parser.add_argument('--g', type=float, default=.65)
    parser.add_argument('--train-size', type=int, default=256)
    parser.add_argument('--val-size', type=int, default=64)
    parser.add_argument('--test-size', type=int, default=0, help='Must be zero: validation-only pilot')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--selection-count', type=int, default=16)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--broad-fraction', type=float, default=.2)
    parser.add_argument('--rollout-weight', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--solver-steps', type=int, default=30)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', default='')
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.test_size != 0 or min(args.train_size, args.val_size, args.batch_size, args.steps, args.eval_every, args.selection_count) < 1:
        parser.error('positive train/validation/counts required; test-size must be zero')
    if not 0 <= args.broad_fraction <= 1 or args.rollout_weight < 0:
        parser.error('invalid loss/sampling weights')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
