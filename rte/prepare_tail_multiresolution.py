"""Acquire paired N128 training/validation tails for the next mixed-grid study.

Uses the exact same material/source draws as the existing N64 bank, not test
media. Paired resolutions count as one draw, not independent extra examples.
No neural training or test evaluation is performed by this preparation step.
"""

import argparse
import json
from pathlib import Path

import torch

from rte.audit_tail_injection import prepare_contexts
from rte.benchmark_tail_solver import digest
from rte.train_source_predictor import acquire, source_family


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(args=vars(args), complete=False,
        scope='paired N128 training and validation teachers and actual TSA8 four-step contexts; no test access or training',
        source_sha256={p:digest(p) for p in ('rte/prepare_tail_multiresolution.py',
            'rte/train_source_predictor.py', 'rte/audit_tail_injection.py', 'rte/augmented_fgmres.py')})
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    for split, count, seed, coarse_path in [
        ('train', 128, 3626, 'rte/workstation_runs/source_predictor_pilot_20260926/train.pt'),
        ('validation', 16, 4626, 'rte/workstation_runs/source_tail_validation_20260926/validation.pt')]:
        cfg, data = source_family(128, count, seed)
        coarse = torch.load(coarse_path, map_location='cpu', weights_only=True)['data']
        for key in ('sigma_t', 'sigma_s'):
            if not torch.equal(data[key][..., ::2, ::2], coarse[key]):
                raise RuntimeError('paired material identity failed: '+key)
        for key in ('centers', 'phases', 'amplitudes', 'angular_codes', 'geometry_base64'):
            if not torch.equal(data[key], coarse[key]):
                raise RuntimeError('paired source/geometry parameters differ: '+key)
        acquire(cfg, data, .1, torch.device('cuda'), out, split)
        prepare_contexts(out/f'{split}.pt', out/f'{split}_context.pt', torch.device('cuda'))
        settings[split] = dict(count=count, seed=seed, coarse_parent=coarse_path,
            coarse_parent_sha256=digest(coarse_path), nested_material_and_source_parameters_exact=True,
            bank_sha256=digest(out/f'{split}.pt'), context_sha256=digest(out/f'{split}_context.pt'))
    settings['complete'] = True
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
