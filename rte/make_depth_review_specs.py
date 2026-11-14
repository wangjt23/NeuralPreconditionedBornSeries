"""Expand classical depth controls without changing the frozen learned arms."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    specs = []
    for mode, family in [('compiled_hss', 'compiled_hss'), ('classical', 'hss')]:
        for depth in (2, 4, 8, 16, 32):
            for scale in (.5, 1., 2.):
                specs.append(dict(label=f'{family}_d{depth}_s{scale}', group=family, mode=mode,
                    family=family, depth=depth, scale=scale))
    for family in ('tsa', 'dsa', 'dsa_transport', 'reference'):
        for depth in ((2, 4, 8, 16) if family in ('tsa', 'reference') else (2, 4, 8)):
            for scale in (0., .03):
                specs.append(dict(label=f'{family}_d{depth}_s{scale}', group=family, mode='classical',
                    family=family, depth=depth, scale=scale))
    for kind in ('ct', 'sct'):
        path = f'rte/workstation_runs/{kind}4_krylov_n128_seed402_20260912/best.pt'
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        for repeats in (1, 2):
            specs.append(dict(label=f'{kind}128_r{repeats}', group=kind, checkpoint=path, repeats=repeats))
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specs, indent=2)+'\n')


if __name__ == '__main__':
    main()
