"""Bounded tail-network fine tuning through two real classical continuation steps."""

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from rte.benchmark_tail_solver import digest, load_model
from rte.global_source_predictor import FrozenTSA
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.run_residual_subspace_experiment import write_csv
from rte.tail_lookahead import continuation_loss
from rte.train_source_predictor import reference_cache
from rte.train_tail_direction import normalized_bank, teacher_direction_loss, validate
from rte.transport_baselines import ClassicalTransportStack


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    bank = normalized_bank(args.context, 'cuda')
    parent = torch.load(args.validation_parent, map_location='cpu', weights_only=True)
    warm = torch.load(args.warm_start, map_location='cpu', weights_only=True)
    # Preserve the standard checkpoint schema for downstream evaluation.
    for key in ('width', 'blocks', 'modes'):
        setattr(args, key, warm['args'][key])
    model = load_model(args.warm_start, False, 'cuda')
    cfg = ExperimentConfig(size=bank['q'].shape[-1], n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
    settings = dict(args=vars(args), complete=False,
        scope='training and validation only; two-step actual flexible-Arnoldi continuation loss',
        selection='validation failures, classical calls +0.5*injection calls, then true residual; same proxy as base network',
        warm_start_sha256=digest(args.warm_start), context_sha256=digest(args.context),
        source_sha256={p:digest(p) for p in ('rte/train_tail_lookahead.py', 'rte/tail_lookahead.py',
            'rte/train_tail_direction.py', 'rte/augmented_fgmres.py', 'rte/global_source_predictor.py')})
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    started = time.perf_counter()
    inverses, streams = [], []
    with torch.no_grad():
        for start in range(0, len(bank['q']), args.batch):
            sl = slice(start, start+args.batch)
            low = make_operator(cfg, bank['t'][sl], bank['s'][sl], bank['q'][sl], 'cuda')
            base = ClassicalTransportStack(low, 'tsa', 8, 0.)
            inv, stream = reference_cache(base)
            inverses.append(inv)
            streams.append(stream)
        bank['inverse'], bank['stream'] = torch.cat(inverses), torch.cat(streams)
    del inverses, streams, base
    settings['reference_cache_seconds'] = time.perf_counter()-started
    low = make_operator(cfg, bank['t'][:args.batch], bank['s'][:args.batch], bank['q'][:args.batch], 'cuda')
    high = high_precision_copy(low)
    generator = torch.Generator().manual_seed(args.seed+72)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.05)
    history, validations, best, best_step, best_state = [], [], None, 0, None
    started = time.perf_counter()
    for step in range(args.steps+1):
        if step % args.eval_every == 0 or step == args.steps:
            score, records = validate(model, parent, 'cuda')
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
            print(f'lookahead step {step}: failures={score[0]}, equivalent calls={score[1]:.4f}, best={best_step}', flush=True)
        if step == args.steps:
            break
        model.train()
        ids = torch.randint(len(bank['q']), (args.batch,), generator=generator).to('cuda')
        t, s, r, ref, target, basis = [bank[k][ids] for k in
            ('t', 's', 'rhs_exact', 'reference', 'target', 'image_basis')]
        angle = torch.rand(args.batch, generator=generator, dtype=torch.float64)*6.283185307179586
        phase = torch.polar(torch.ones_like(angle), angle).to('cuda')
        r, target = r*phase[:, None, None, None], target*phase[:, None, None, None]
        ref = (ref*phase[:, None, None, None]).to(torch.complex64)
        basis = basis*phase[:, None, None, None, None]
        low.setup(t, s, r.to(torch.complex64))
        high.setup(t.double(), s.double(), r)
        preconditioner = FrozenTSA(low, bank['inverse'][ids], bank['stream'][ids], 8)
        precondition = lambda value: preconditioner(value.to(torch.complex64)).to(torch.complex128)
        proposal = model(t, s, r.to(torch.complex64), ref).to(torch.complex128)
        ratios = continuation_loss(high.A_op, precondition, proposal, r, basis, args.lookahead)
        supervised = teacher_direction_loss(proposal, target).mean()
        loss = ratios.clamp_min(1e-12).log().mean()+.05*supervised
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite lookahead loss; checkpoints retained')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if (step+1) % 20 == 0:
            history.append(dict(step=step+1, loss=float(loss), teacher_loss=float(supervised),
                injection_ratio=float(ratios[:,0].mean()), final_ratio=float(ratios[:,-1].mean()),
                grad=float(grad), seconds=time.perf_counter()-started))
            write_csv(out/'history.csv', history)
    settings.update(complete=True, selected_step=best_step, validation_score=best,
        training_seconds=time.perf_counter()-started)
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--context', default='rte/workstation_runs/tail_injection_gate_20260926/train_context.pt')
    parser.add_argument('--validation-parent', default='rte/workstation_runs/source_tail_validation_20260926/validation.pt')
    parser.add_argument('--warm-start', default='rte/workstation_runs/tail_direction_pilot_20260926/best.pt')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lookahead', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=26929)
    run(parser.parse_args())
