"""Bounded directional-decoder continuation using the same audited protocol.

Equivalent to the train_source_predictor CLI with a characteristic decoder.
The architecture identity is recorded as the executable module and in every
checkpoint's characteristic.* keys. All data, loss and evaluation code are
the same; the base training source remains unchanged.
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys

import rte.global_source_predictor as global_model
from rte.characteristic_source_predictor import CharacteristicSourcePredictor


if __name__ == '__main__':
    global_model.GlobalSourcePredictor = CharacteristicSourcePredictor
    provenance = dict(architecture='CharacteristicSourcePredictor, three fine-grid directional scales',
        command=sys.argv, source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('rte/train_source_predictor.py', 'rte/global_source_predictor.py',
                         'rte/characteristic_source_predictor.py', 'rte/train_characteristic_source.py')},
        upgrade='base weights accepted only with missing characteristic.* keys; zero branch output preserves base prediction')
    try:
        runpy.run_module('rte.train_source_predictor', run_name='__main__')
    finally:
        if '--output' in sys.argv:
            out = Path(sys.argv[sys.argv.index('--output')+1])
            if out.is_dir():
                (out/'architecture.json').write_text(json.dumps(provenance, indent=2)+'\n')
