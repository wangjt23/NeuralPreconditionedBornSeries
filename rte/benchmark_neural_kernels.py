"""Compare real-two-RHS and complex cached applications at identical weights."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import synchronized, write_csv
from rte.train_collision_transport_ensemble import material_split
from rte.train_neural_transport import make_model


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    saved = SimpleNamespace(**payload['args'])
    cfg = ExperimentConfig(size=saved.size, n_angles=saved.n_angles, anisotropy_g=saved.g,
                           regimes='mixed', batch_size=saved.batch_size, eta_scale=0.)
    data = material_split(saved, cfg, 'validation')
    device = torch.device(args.device)
    ids = torch.tensor([0]); q = source_from_data(cfg, data, ids, device)
    op = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, device)
    model = make_model(saved, op).to(device).eval()
    model.load_state_dict(payload['model'])
    rows = []
    generator = torch.Generator().manual_seed(9721)
    with torch.no_grad():
        for index in range(4):
            q = setup_batch(cfg, op, data, torch.tensor([index]), device)
            maps, symbol = model.prepare()
            kernels = [('real_two_rhs', maps.real), ('complex', maps.to(q.dtype))]
            model._prepared = kernels[1][1], symbol
            reference = model(q)
            # Interleave warmed variants; conversion and preparation excluded.
            for repeat in range(args.repeats):
                for position in torch.randperm(2, generator=generator).tolist():
                    label, matrix = kernels[position]
                    model._prepared = matrix, symbol
                    model(q)
                    synchronized(device); start = time.perf_counter()
                    result = model(q)
                    synchronized(device)
                    seconds = time.perf_counter()-start
                    error = float((result-reference).norm()/reference.norm())
                    rows.append(dict(index=index, variant=label, repeat=repeat, seconds=seconds,
                                     relative_to_complex=error))
    write_csv(out/'samples.csv', rows)
    summary = [dict(variant=name, median_seconds=statistics.median(r['seconds'] for r in rows if r['variant']==name),
                    max_relative_difference=max(r['relative_to_complex'] for r in rows if r['variant']==name))
               for name, _ in kernels]
    write_csv(out/'summary.csv', summary)
    (out/'settings.json').write_text(json.dumps(dict(checkpoint=args.checkpoint, selected_step=payload['selected_step'],
        device=args.device, repeats=args.repeats, scope='same cached map; application only on first four validation media'), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
