# Nonlinear Poisson / Newton systems

We solve `−Δu − u² = −s sin(πx) sin(πy)` on the unit square with zero
Dirichlet boundary conditions. Each Newton step solves
`J(u) δu = −F(u)`, where `J(u)v = −Δv − 2uv`.
Run the commands below from the repository root.

## Data

The experiments use two sets of Newton states, with 5000 training and 500
held-out states per set (10000 training and 1000 test states in total). See
[Newton-Informed-Neural-Operator](https://github.com/xlliu2017/Newton-Informed-Neural-Operator#datasets)
for the original data source. A data file contains `{"u": states}` with shape
`(samples, 1, N, N)` or `(samples, N, N)`; `du` is optional because training
only needs the states. Tensor tuples beginning with `u` are also accepted.

For a legacy file containing NumPy/pickle objects, convert a trusted file once:

```bash
python nonlinear/convert_data.py \
  --input path/to/newton_method_iterations_7_solution.pt \
  --output data/nonlinear/newton_7.pt --trust-pickle
```

Convert the other trajectory file to `data/nonlinear/newton_5.pt` in the same way.
For a small synthetic dataset instead, use:

```bash
python nonlinear/generate_data.py \
  --output data/nonlinear/small.pt --samples 32 --grid-size 12 --source-scale 10
```

For this small dataset, train with `--source-scale 10 --grid-size 12
--width 4 --modes 4 4 --padding 2 --train-count 25 --test-count 7`.
For inference on its held-out states, use `--start 25 --count 7`.

## Training

```bash
python nonlinear/train.py \
  --data data/nonlinear/newton_5.pt data/nonlinear/newton_7.pt \
  --train-count 5000 --test-count 500 --grid-size 64 \
  --source-scale 1600 --source-grid legacy-endpoints \
  --width 25 --modes 20 20 --padding 8 \
  --variant npbs-weighted --epochs 200 --batch-size 50 \
  --lr 1e-4 --weight-decay 1e-6 --seed 0 --device cuda \
  --output outputs/nonlinear/npbs_weighted.pt
```

Each file contributes its first 5000 states to training and its next 500 to
validation and subsequent evaluation. These are also the defaults for `--data`;
each file must contain at least 5500 states. The in-memory
`--synthetic-samples` mode retains an 80%/20% split unless counts are specified.
Training uses Gaussian right-hand sides; validation uses `−F(u)`.

| `--variant` | Network input | Loss |
| --- | --- | --- |
| `baseline` | `r` | `‖J Q(u,r) − r‖ / ‖r‖` |
| `npbs-l2` | `B r` | `‖J Q(u,B r) − r‖ / ‖r‖` |
| `npbs-weighted` | `B r` | `‖B(J Q(u,B r) − r)‖ / ‖B r‖` |

Here `B` is the discrete Dirichlet Poisson inverse. Training norms are over
the whole batch. Use a separate checkpoint path for each variant.

## Inference

```bash
python nonlinear/inference.py \
  --checkpoint outputs/nonlinear/npbs_weighted.pt \
  --data data/nonlinear/newton_5.pt data/nonlinear/newton_7.pt \
  --start 5000 --count 500 \
  --mode inner --max-iterations 100 --tolerance 1e-4 \
  --output outputs/nonlinear/inner.pt --device cuda
```

This evaluates 500 held-out states from each file, for 1000 tests in one run.
`--start` and `--count` apply to each file and default to 5000 and 500.
Use `--mode newton --newton-iterations 10 --newton-tolerance 1e-4` for the
outer Newton loop. Results include solutions, source-file identifiers, and
per-sample residual histories.
Use `--device cpu` for CPU execution.

The five-point stencil uses interior spacing `h = 1/(N+1)`. The default
`legacy-endpoints` source uses `linspace(0,1,N)`, as in the original code;
`--source-grid interior` samples at the stencil coordinates instead.
`--grid-size` resamples states bilinearly. Inference reads the grid size,
source scale, and coordinate convention from the checkpoint.
