"""Train one tail direction on actual classical-prefix states, no test access."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from rte.augmented_fgmres import augmented_fgmres, projected_direction_loss
from rte.audit_tail_injection import operator_from_bank
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.run_residual_subspace_experiment import write_csv
from rte.tail_response_network import TailResponseNetwork
from rte.tail_moment_space import combine_moment_space, moment_injected_solve
from rte.optical_tail_network import OpticalTailNetwork


def normalized_bank(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    bank = {k: v.to(device) for k, v in saved['bank'].items()}
    scale = bank['rhs_exact'].abs().square().mean((1, 2, 3), keepdim=True).sqrt()
    for key in ('q', 'rhs_exact', 'target', 'reference'):
        bank[key] = bank[key]/scale.to(bank[key].real.dtype)
    return bank


def teacher_direction_loss(proposal, teacher):
    rms = lambda x: x.abs().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-30)
    return ((proposal/rms(proposal))-(teacher/rms(teacher))).abs().square().flatten(1).mean(1)


@torch.no_grad()
def validate(model, parent, device, moments=False):
    model.eval()
    records = []
    for index in range(len(parent['bank']['q'])):
        low, high, base, initial = operator_from_bank(parent['bank'], index, device)
        def inject(x, r):
            r = r.to(torch.complex64)
            reference = base.reference.reference_transport_solve(r)
            return model(low.sigma_t, low.sigma_s, r, reference).to(torch.complex128)
        if moments:
            _, status = moment_injected_solve(high, low, base, initial, model)
        else:
            _, status = augmented_fgmres(high.A_op, high.source,
                lambda r: base(r.to(torch.complex64)).to(torch.complex128), initial, inject=inject)
        status.pop('trace')
        records.append(dict(index=index, **status))
    # Precommitted work proxy: measured neural cost ~half one TSA8 call.
    score = (sum(row['failed'] for row in records),
        sum(row['classical_calls']+.5*row['injection_calls'] for row in records)/len(records),
        sum(row['true_residual'] for row in records)/len(records))
    return score, records


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(args=vars(args), complete=False, scope='tail-direction training and validation only; zero test access',
        selection='failures, mean classical calls + 0.5*neural calls, then physical residual; all 16 validation cases',
        architecture='TailResponseNetwork, global/local plus characteristic filters, reference-RMS scaling',
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['rte/augmented_fgmres.py', 'rte/tail_response_network.py',
             'rte/optical_tail_network.py', 'rte/tail_moment_space.py',
             'rte/train_tail_direction.py']})
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    bank = normalized_bank(args.context, device)
    parent = torch.load(args.validation_parent, map_location='cpu', weights_only=True)
    model_type = OpticalTailNetwork if args.optical else TailResponseNetwork
    model = model_type(width=args.width, blocks=args.blocks, modes=args.modes).to(device)
    settings['architecture'] = model_type.__name__
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location='cpu', weights_only=True)
        for key in ('width', 'blocks', 'modes'):
            if warm['args'][key] != getattr(args, key):
                raise ValueError('warm-start architecture mismatch')
        model.load_state_dict(warm['model'])
        settings['warm_start_sha256'] = hashlib.sha256(Path(args.warm_start).read_bytes()).hexdigest()
        settings['warm_start_mode'] = 'weights only; optimizer/schedule/sampling reset'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.05)
    n = bank['q'].shape[-1]
    cfg = ExperimentConfig(size=n, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
    low = make_operator(cfg, bank['t'][:args.batch], bank['s'][:args.batch], bank['q'][:args.batch], device)
    high = high_precision_copy(low)
    generator = torch.Generator().manual_seed(args.seed+72)
    history, validations, best, best_step, best_state = [], [], None, 0, None
    started = time.perf_counter()
    for step in range(args.steps+1):
        if step % args.eval_every == 0 or step == args.steps:
            score, records = validate(model, parent, device, args.moments)
            if best is None or score < best:
                best, best_step, best_state = score, step, copy.deepcopy(model.state_dict())
                torch.save(dict(args=vars(args), model={k:v.cpu() for k,v in best_state.items()},
                    step=step, score=score), out/'best.pt')
            validations.extend(dict(step=step, **row) for row in records)
            write_csv(out/'validation.csv', validations)
            torch.save(dict(args=vars(args), step=step, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                generator_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), best_step=best_step, best_score=best,
                best_model=best_state, history=history), out/'last.pt')
            print(f'tail step {step}: failures={score[0]}, equivalent calls={score[1]:.4f}, best={best_step}', flush=True)
        if step == args.steps:
            break
        model.train()
        ids = torch.randint(len(bank['q']), (args.batch,), generator=generator).to(device)
        t, s, r, reference, target, basis = [bank[k][ids] for k in
            ('t', 's', 'rhs_exact', 'reference', 'target', 'image_basis')]
        # Complex phase augmentation preserves the true operator and real
        # orthogonality when residual, teacher and search images rotate together.
        phase = torch.polar(torch.ones(args.batch), torch.rand(args.batch, generator=generator)*6.283185307179586).to(device)
        r = r*phase[:, None, None, None]
        reference = reference*phase[:, None, None, None]
        target = target*phase[:, None, None, None]
        basis = basis*phase[:, None, None, None, None]
        low.setup(t, s, r.to(torch.complex64))
        high.setup(t.double(), s.double(), r)
        z = model(t, s, r.to(torch.complex64), reference).to(torch.complex128)
        teacher_loss = teacher_direction_loss(z, target).mean()
        if args.moments:
            _, projected = combine_moment_space(high.A_op, z, r, basis)
        else:
            projected = projected_direction_loss(high.A_op, z, r, basis)
        loss = teacher_loss if step < args.supervised_steps else projected.clamp_min(1e-8).log().mean()+.05*teacher_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite tail loss; last checkpoint preserved')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if (step+1) % 20 == 0:
            history.append(dict(step=step+1, loss=float(loss), teacher_loss=float(teacher_loss),
                projected_ratio=float(projected.mean()), grad=float(grad), seconds=time.perf_counter()-started))
            write_csv(out/'history.csv', history)
    settings.update(complete=True, selected_step=best_step, validation_score=best,
        parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
        training_seconds=time.perf_counter()-started,
        context_sha256=hashlib.sha256(Path(args.context).read_bytes()).hexdigest())
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--context', default='rte/workstation_runs/tail_injection_gate_20260926/train_context.pt')
    parser.add_argument('--validation-parent', default='rte/workstation_runs/source_tail_validation_20260926/validation.pt')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--supervised-steps', type=int, default=800)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=26927)
    parser.add_argument('--moments', action='store_true')
    parser.add_argument('--warm-start', default='')
    parser.add_argument('--optical', action='store_true')
    run(parser.parse_args())
