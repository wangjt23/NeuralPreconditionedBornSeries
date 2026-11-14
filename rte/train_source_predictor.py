"""Bounded source-manifold learning: independent materials, solver-aware loss.

No fine inverse matrices. Accurate fine-grid solutions are offline labels.
The final solver remains classical and checks the original physical residual.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from rte.compiled_hss import CompiledHSS
from rte.global_source_predictor import FrozenTSA, GlobalSourcePredictor, balance_integrated_source
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_medium_dataset, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import solve_from_initial
from rte.train_krylov_transport import short_fgmres
from rte.transport_baselines import ClassicalTransportStack


def source_family(size, count, seed):
    if size % 64:
        raise ValueError('this pilot uses grids divisible by 64')
    cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
    data = make_medium_dataset(cfg, count, seed)
    rng = torch.Generator().manual_seed(seed+913)
    geometry = torch.stack((torch.randint(4, (count,), generator=rng),
        torch.randint(64, (count,), generator=rng), torch.randint(64, (count,), generator=rng)), 1)
    offsets = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for key in ('sigma_t', 'sigma_s'):
        values = []
        for value, code in zip(data[key], geometry):
            angle = int(code[0])
            # Rotate about the periodic origin, rather than about a grid-cell
            # centre, so nested N64/N128 media agree exactly.
            value = torch.roll(torch.rot90(value, angle, (-2, -1)), offsets[angle], (-2, -1))
            values.append(torch.roll(value, (int(code[1])*size//64, int(code[2])*size//64), (-2, -1)))
        data[key] = torch.stack(values)
    data['geometry_base64'] = geometry
    return cfg, data


def material_operator(cfg, data, ids, epsilon, device):
    q = source_from_data(cfg, data, ids, device)
    t = data['sigma_t'][ids]/epsilon
    s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
    low = make_operator(cfg, t, s, q, device)
    return low, high_precision_copy(low)


def reference_cache(base):
    ref = base.reference
    stream = 1/(ref._stream_symbol(len(ref.sigma_t), ref.complex_dtype, ref.sigma_t.device)+ref.sigma0)
    return ref.reference_transport_inverse, stream


@torch.no_grad()
def acquire(cfg, data, epsilon, device, out, split):
    entries, logs = {k: [] for k in ('t', 's', 'q', 'target', 'reference', 'inverse', 'stream')}, []
    for index in range(len(data['sigma_t'])):
        low, high = material_operator(cfg, data, torch.tensor([index]), epsilon, device)
        base = ClassicalTransportStack(low, 'tsa', 8, 0.)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        target, status = solve_from_initial(high.A_op, high.source,
            lambda r: base(r.to(torch.complex64)).to(torch.complex128), tolerance=1e-9, maxiter=250)
        used = 'tsa8'
        if status['failed']:
            fallback = CompiledHSS(low, 16, 2.).prepare()
            target, status = solve_from_initial(high.A_op, high.source,
                lambda r: fallback(r.to(torch.complex64)).to(torch.complex128), tolerance=1e-9, maxiter=350)
            used = 'hss16s2_fallback'
        torch.cuda.synchronize(device)
        logs.append(dict(index=index, seconds=time.perf_counter()-start, method=used, **status))
        if status['failed']:
            write_csv(out/f'{split}_teachers.csv', logs)
            raise RuntimeError(f'{split} teacher {index} failed; do not discard this medium')
        inverse, stream = reference_cache(base)
        reference = base.reference.reference_transport_solve(low.source)
        for key, value in [('t', low.sigma_t), ('s', low.sigma_s), ('q', low.source),
                           ('target', target), ('reference', reference), ('inverse', inverse), ('stream', stream)]:
            entries[key].append(value.detach())
        if (index+1) % 16 == 0 or index+1 == len(data['sigma_t']):
            write_csv(out/f'{split}_teachers.csv', logs)
            print(f'{split}: {index+1}/{len(data["sigma_t"])} accurate teachers', flush=True)
    bank = {k: torch.cat(v) for k, v in entries.items()}
    # Homogeneous symbols are cheap to regenerate, so do not duplicate them on disk.
    torch.save(dict(data=data, bank={k: v.cpu() for k, v in bank.items() if k not in ('inverse', 'stream')}), out/f'{split}.pt')
    return bank


@torch.no_grad()
def reuse_training_bank(path, cfg, data, epsilon, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('sigma_t', 'sigma_s', 'centers', 'phases', 'amplitudes', 'angular_codes', 'geometry_base64'):
        if not torch.equal(saved['data'][key], data[key]):
            raise ValueError('teacher-bank material/source identity mismatch: '+key)
    bank = {key: value.to(device) for key, value in saved['bank'].items()}
    expected_t = data['sigma_t']/epsilon
    expected_s = expected_t-epsilon*(data['sigma_t']-data['sigma_s'])
    if not torch.equal(bank['t'].cpu(), expected_t) or not torch.equal(bank['s'].cpu(), expected_s):
        raise ValueError('teacher-bank stiffness mismatch')
    inverses, streams = [], []
    for start in range(0, len(expected_t), 8):
        low = make_operator(cfg, bank['t'][start:start+8], bank['s'][start:start+8], bank['q'][start:start+8], device)
        base = ClassicalTransportStack(low, 'tsa', 8, 0.)
        inverse, stream = reference_cache(base)
        inverses.append(inverse)
        streams.append(stream)
    bank['inverse'], bank['stream'] = torch.cat(inverses), torch.cat(streams)
    return bank


def build_base(low, name):
    if name.startswith('hss'):
        depth, shift = name[3:].split('s')
        return CompiledHSS(low, int(depth), float(shift)).prepare()
    for kind in ('tsa', 'reference'):
        if name.startswith(kind):
            return ClassicalTransportStack(low, kind, int(name[len(kind):]), 0.)
    raise ValueError(name)


@torch.no_grad()
def validation(model, cfg, data, epsilon, device, limit):
    model.eval()
    records = []
    for index in range(min(limit, len(data['sigma_t']))):
        low, high = material_operator(cfg, data, torch.tensor([index]), epsilon, device)
        base = ClassicalTransportStack(low, 'tsa', 8, 0.)
        reference = base.reference.reference_transport_solve(low.source)
        prediction = model(low.sigma_t, low.sigma_s, low.source, reference).to(torch.complex128)
        prediction = balance_integrated_source(high, prediction, high.source)
        _, status = solve_from_initial(high.A_op, high.source,
            lambda r: base(r.to(torch.complex64)).to(torch.complex128), prediction)
        records.append(status)
    return (sum(r['failed'] for r in records), sum(r['iterations'] for r in records)/len(records),
            sum(r['true_residual'] for r in records)/len(records))


@torch.no_grad()
def evaluate(model, cfg, data, epsilon, device, candidates, repeats, output, label,
             kinds=('zero', 'reference_balanced', 'neural')):
    rows = []
    randomizer = random.Random(8251)
    model.eval()
    for index in range(len(data['sigma_t'])):
        low, high = material_operator(cfg, data, torch.tensor([index]), epsilon, device)
        bases, setups = {}, {}
        for name in candidates:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            bases[name] = build_base(low, name)
            torch.cuda.synchronize(device)
            setups[name] = time.perf_counter()-start
        # Reference conditioning is part of initialization and is timed below.
        ref_base = bases.get('tsa8')
        if ref_base is None:
            ref_base = ClassicalTransportStack(low, 'tsa', 8, 0.)
        methods = [(name, kind) for name in candidates for kind in kinds]
        for repeat in range(-1, repeats):
            randomizer.shuffle(methods)
            for order, (name, kind) in enumerate(methods):
                base = bases[name]
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                initial = None
                if kind != 'zero':
                    conditioned = base if hasattr(base, 'reference') else ref_base
                    reference = conditioned.reference.reference_transport_solve(low.source)
                    initial = reference if kind == 'reference_balanced' else model(low.sigma_t, low.sigma_s, low.source, reference)
                    initial = balance_integrated_source(high, initial.to(torch.complex128), high.source)
                torch.cuda.synchronize(device)
                initialization_seconds = time.perf_counter()-start
                _, status = solve_from_initial(high.A_op, high.source,
                    lambda r: base(r.to(torch.complex64)).to(torch.complex128), initial)
                torch.cuda.synchronize(device)
                seconds = time.perf_counter()-start
                setup = setups[name]
                if kind != 'zero' and not hasattr(base, 'reference'):
                    setup += setups['tsa8']
                if repeat >= 0:
                    rows.append(dict(split=label, size=cfg.size, index=index, repeat=repeat, order=order,
                        method=name+'_'+kind, seconds=seconds, initialization_seconds=initialization_seconds, **status))
                    rows[-1].update(setup_seconds=setup, setup_plus_seconds=setup+seconds)
        write_csv(output, rows)
        if (index+1) % 8 == 0:
            print(f'{label} N{cfg.size}: {index+1}/{len(data["sigma_t"])} evaluated', flush=True)
    return rows


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('this bounded training pilot requires the available GPU')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(args=vars(args), device=torch.cuda.get_device_name(),
        gpu_before=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True),
        scope='source-conditioned one-shot initializer; independent material/source splits; no arbitrary-residual neural preconditioner',
        accuracy='unchanged FP32 geometry promoted to FP64; final residual relative to original source',
        selection='validation TSA8 failures then iterations then residual; classical candidates selected on validation timing',
        timing='synchronized, shuffled warm solves; neural/reference initialization included; common physical/preconditioner setup excluded')
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    cfg, train = source_family(args.size, args.train_count, args.seed+1000)
    _, val = source_family(args.size, args.val_count, args.seed+2000)
    if args.reuse_train:
        bank = reuse_training_bank(args.reuse_train, cfg, train, args.epsilon, device)
        settings['teacher_bank_sha256'] = hashlib.sha256(Path(args.reuse_train).read_bytes()).hexdigest()
    else:
        bank = acquire(cfg, train, args.epsilon, device, out, 'train')
    torch.save(dict(data=val), out/'validation_data.pt')
    model = GlobalSourcePredictor(width=args.width, blocks=args.blocks, modes=args.modes).to(device)
    if args.warm_start:
        initial = torch.load(args.warm_start, map_location='cpu', weights_only=True)
        for key in ('width', 'blocks', 'modes', 'size', 'epsilon', 'train_count', 'val_count', 'seed'):
            if initial['args'][key] != getattr(args, key):
                raise ValueError('warm-start configuration mismatch: '+key)
        model.load_state_dict(initial['model'])
        settings['warm_start_sha256'] = hashlib.sha256(Path(args.warm_start).read_bytes()).hexdigest()
        settings['warm_start_mode'] = 'weights only, optimizer/scheduler/RNG deliberately reset'
    settings['parameters'] = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.05)
    generator = torch.Generator().manual_seed(args.seed+811)
    low, high = material_operator(cfg, train, torch.arange(args.batch_size), args.epsilon, device)
    best, best_step, best_state, history = None, 0, None, []
    started = time.perf_counter()
    for step in range(args.steps+1):
        if step % args.eval_every == 0 or step == args.steps:
            score = validation(model, cfg, val, args.epsilon, device, args.validation_limit)
            if best is None or score < best:
                best, best_step, best_state = score, step, copy.deepcopy(model.state_dict())
                torch.save(dict(args=vars(args), model={k: v.cpu() for k, v in best_state.items()},
                    selected_step=best_step, validation_score=best), out/'best.pt')
            torch.save(dict(args=vars(args), step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                generator_rng=generator.get_state(), best_step=best_step, best_score=best,
                history=history), out/'last.pt')
            print(f'step {step}: validation failures={score[0]}, iterations={score[1]:.3f}, best={best_step}', flush=True)
        if step == args.steps:
            break
        model.train()
        ids = torch.randint(args.train_count, (args.batch_size,), generator=generator).to(device)
        t, s, q, reference, target = [bank[k][ids] for k in ('t', 's', 'q', 'reference', 'target')]
        low.setup(t, s, q)
        high.setup(t.double(), s.double(), q.to(torch.complex128))
        prediction = model(t, s, q, reference).to(torch.complex128)
        prediction = balance_integrated_source(high, prediction, high.source)
        scale = model.intensity_scale(t, s, q).to(torch.float64)
        mse = ((prediction-target)/scale).abs().square().mean()
        if step < args.supervised_steps:
            loss, terminal = mse, torch.tensor(float('nan'), device=device)
        else:
            base = FrozenTSA(low, bank['inverse'][ids], bank['stream'][ids], depth=8)
            residual = high.source-high.A_op(prediction)
            relative_squared, _ = short_fgmres(high.A_op, base, residual, steps=args.krylov_steps)
            weight = residual.abs().square().flatten(1).sum(1)/high.source.abs().square().flatten(1).sum(1)
            terminal = relative_squared*weight
            if args.terminal_target > 0:
                loss = torch.log1p(terminal/args.terminal_target**2).mean()+.01*mse.clamp_min(1e-14).log()
            else:
                loss = terminal.clamp_min(1e-14).log().mean()+.1*mse.clamp_min(1e-14).log()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite training loss; last checkpoint retained')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        if (step+1) % 10 == 0:
            history.append(dict(step=step+1, loss=float(loss), mse=float(mse),
                terminal=float(terminal.mean()), grad=float(grad), seconds=time.perf_counter()-started))
            write_csv(out/'history.csv', history)
    settings['training_seconds'] = time.perf_counter()-started
    settings['selected_step'], settings['validation_score'] = best_step, best
    model.load_state_dict(best_state)
    candidates = ['tsa2', 'tsa4', 'tsa8', 'tsa16', 'tsa32', 'reference4', 'reference8', 'reference16', 'reference32',
                  'hss8s1', 'hss16s1', 'hss16s2', 'hss32s2', 'hss64s2']
    validation_rows = evaluate(model, cfg, val, args.epsilon, device, candidates, args.validation_repeats,
        out/'validation_classical_cases.csv', 'validation_classical', kinds=('zero', 'reference_balanced'))
    import pandas as pd
    frame = pd.DataFrame(validation_rows)
    pooled = frame.groupby(['index', 'method']).agg(seconds=('seconds', 'median'), failed=('failed', 'max')).reset_index().groupby('method').agg(seconds=('seconds', 'mean'), failed=('failed', 'sum'))
    classical = pooled.sort_values(['failed', 'seconds'])
    shortlist = list(dict.fromkeys(name.split('_')[0] for name in classical.index))[:3]
    shortlist = list(dict.fromkeys(shortlist+['tsa8']))
    neural_rows = evaluate(model, cfg, val, args.epsilon, device, shortlist, args.validation_repeats,
        out/'validation_neural_cases.csv', 'validation_neural', kinds=('neural',))
    neural = pd.DataFrame(neural_rows).groupby(['index', 'method']).agg(seconds=('seconds', 'median'),
        failed=('failed', 'max')).reset_index().groupby('method').agg(seconds=('seconds', 'mean'),
        failed=('failed', 'sum')).sort_values(['failed', 'seconds'])
    selected = classical.index[0].split('_')[0]
    selected_neural = neural.index[0].split('_')[0]
    settings['selected_classical'] = classical.index[0]
    settings['selected_neural'] = neural.index[0]
    # Test materials are created only after both selections are frozen.
    for size in args.test_sizes:
        test_cfg, test = source_family(size, args.test_count, args.seed+args.test_offset)
        torch.save(dict(data=test), out/f'test_data_n{size}.pt')
        evaluate(model, test_cfg, test, args.epsilon, device, list(dict.fromkeys([selected, selected_neural, 'tsa8'])),
            args.test_repeats, out/f'test_cases_n{size}.csv', 'test')
    settings['complete'] = True
    settings['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    print(f'Complete; selected step {best_step}; classical validation choice {selected}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--test-sizes', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--train-count', type=int, default=128)
    parser.add_argument('--val-count', type=int, default=16)
    parser.add_argument('--test-count', type=int, default=24)
    parser.add_argument('--validation-limit', type=int, default=8)
    parser.add_argument('--test-repeats', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=2626)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--supervised-steps', type=int, default=400)
    parser.add_argument('--krylov-steps', type=int, default=3)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--warm-start', default='')
    parser.add_argument('--reuse-train', default='')
    parser.add_argument('--terminal-target', type=float, default=0.)
    parser.add_argument('--test-offset', type=int, default=3000)
    parser.add_argument('--validation-repeats', type=int, default=1)
    run(parser.parse_args())
