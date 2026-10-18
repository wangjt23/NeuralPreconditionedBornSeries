# Convection–diffusion–reaction

We solve `−∇·(κ∇u) + v·∇u + u = f` on a periodic unit square.
Run the commands below from the repository root.

## Data

Generate 4000 training and 400 test samples of 128 × 128 coefficient fields and sources:

```bash
python cdr/generate_data.py \
  --output data/cdr --train-samples 4000 --test-samples 400 \
  --resolution 128 --low-resolution 16 --ill-level 0.4 --seed 42
```

Each `sample_*.pt` contains float32 tensors:

| Key | Shape | Field |
| --- | --- | --- |
| `kappa` | `(N, N)` | Positive diffusion coefficient |
| `v` | `(N, N, 2)` | Velocity components |
| `f` | `(N, N)` | Source |

The reaction coefficient is fixed at one. Training uses random Gaussian
right-hand sides; the stored sources are used for inference.

## Training

```bash
python cdr/train.py \
  --train-path data/cdr/train --test-path data/cdr/test \
  --train-samples 4000 --test-samples 400 \
  --output outputs/cdr/npbs --mode npbs \
  --resolution 128 --features 25 --modes 32 32 32 --padding 8 \
  --epochs 60 --batch-size 32 --lr 1e-3 --seed 42
```

`--test-path` supplies the validation data used to select `best.pt`.
Generation and training default to 4000/400 samples; inference defaults to
400 test samples. A requested count must be available in the input directory.
Use explicit smaller counts for a small test run.
Use separate output directories for the three variants:

| `--mode` | Network input | Loss |
| --- | --- | --- |
| `plain_l2` | `r` | `‖A M(r) − r‖ / ‖r‖` |
| `preconditioned_l2` | `G r` | `‖A M(G r) − r‖ / ‖r‖` |
| `npbs` | `G r` | `‖G(A M(G r) − r)‖ / ‖G r‖` |

The loss averages per-sample relative norms. The reference inverse uses
`Ĝ(ξ) = 1 / (mean(κ)|ξ|² + i mean(v)·ξ + 2)` for each sample.
The constant `2` combines reaction one and damping one.

## Inference

```bash
python cdr/inference.py \
  --checkpoint outputs/cdr/npbs/best.pt --data-path data/cdr/test \
  --output outputs/cdr/npbs/results --samples 400 \
  --iterations 600 --tolerance 1e-6
```

Results contain the solution, residual history, and convergence status.
All scripts accept `--device cpu`, `cuda`, or `cuda:N`; the default selects
CUDA when available. Inference uses float64 physics and a float32 network.

The FFT grid has spacing `1/N`. The network is specific to its grid size,
and the padded grid must be even.
