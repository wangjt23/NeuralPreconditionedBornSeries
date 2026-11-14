"""Write explicit bounded classical candidate grids and frozen checkpoint arms."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='*', default=[], help='label=checkpoint.pt')
    parser.add_argument('--selected-classical', default='')
    parser.add_argument('--learned-repeats', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.selected_classical:
        specs = json.loads(Path(args.selected_classical).read_text())
    else:
        specs = []
        for mode, family in [('compiled_hss', 'compiled_hss'), ('classical', 'hss')]:
            for depth in (2, 4, 8):
                for scale in (.5, 1., 2.):
                    specs.append(dict(label=f'{family}_d{depth}_s{scale}', group=family, mode=mode,
                                      family=family, depth=depth, scale=scale))
        for family in ('tsa', 'dsa', 'dsa_transport', 'reference'):
            for depth in (2, 4, 8):
                for scale in (0., .03):
                    specs.append(dict(label=f'{family}_d{depth}_s{scale}', group=family, mode='classical',
                                      family=family, depth=depth, scale=scale))
    for item in args.checkpoints:
        label, path = item.split('=', 1)
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        for repeats in args.learned_repeats:
            specs.append(dict(label=f'{label}_r{repeats}', group=label, checkpoint=path, repeats=repeats))
    if len({s['label'] for s in specs}) != len(specs):
        raise ValueError('duplicate candidate labels')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2)+'\n')


if __name__ == '__main__':
    main()
