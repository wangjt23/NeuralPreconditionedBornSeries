# NPBS

Code for **Born-Preconditioned Learning for Iterative Neural PDE Solvers**.

## Requirements

Python 3.10+ and PyTorch with CUDA support for training.

```bash
pip install -r requirements.txt
```

## Experiments

| Directory | Experiment |
| --- | --- |
| [helmholtz](helmholtz/README.md) | 2D/3D Helmholtz equations, CBS and FGMRES |
| [cdr](cdr/README.md) | Convection–diffusion–reaction equations |
| [nonlinear](nonlinear/README.md) | Newton linear systems for nonlinear Poisson equations |

Run commands from the repository root. Data and output paths can be set through the command-line arguments; the defaults use `data/` and `outputs/`.

## Datasets

The experiment commands use the following training and test counts:

| Dataset | Training samples | Test samples |
| --- | ---: | ---: |
| CurveVel-A / CurveFault-B (each) | 3000 | 50 |
| 3D Kimberlina-V1 | 600 | 50 |
| Convection–diffusion–reaction | 4000 | 400 |
| Newton states (two trajectory sets combined) | 10000 | 1000 |

### Helmholtz

Download the CurveVel-A, CurveFault-B, and 3D Kimberlina-V1 velocity models from [OpenFWI](https://openfwi-lanl.github.io/) and place them under `data/openfwi/`. See [helmholtz/README.md](helmholtz/README.md#data) for the directory layout and file formats.

### Convection–diffusion–reaction

Generate the diffusion coefficients, velocity fields, and sources using the provided script:

```bash
python cdr/generate_data.py \
  --output data/cdr --train-samples 4000 --test-samples 400 \
  --resolution 128 --low-resolution 16 --ill-level 0.4 --seed 42
```

The script writes samples to `data/cdr/train/` and `data/cdr/test/`. See [cdr/README.md](cdr/README.md#data) for the tensor format.

### Nonlinear Poisson

Obtain the Newton-state datasets from the data source linked in [Newton-Informed-Neural-Operator](https://github.com/xlliu2017/Newton-Informed-Neural-Operator#datasets). Convert trusted legacy files to tensor-only files under `data/nonlinear/`:

```bash
python nonlinear/convert_data.py \
  --input path/to/newton_method_iterations_7_solution.pt \
  --output data/nonlinear/newton_7.pt --trust-pickle
```

Convert the other trajectory file to `data/nonlinear/newton_5.pt` in the same way. A small synthetic dataset for testing the workflow can also be generated locally:

```bash
python nonlinear/generate_data.py \
  --output data/nonlinear/small.pt --samples 32 --grid-size 12 --source-scale 10
```

See [nonlinear/README.md](nonlinear/README.md#data) for accepted formats and training parameters for the synthetic dataset.

## Training and evaluation

Training commands, loss settings, and inference examples for each equation are documented in [helmholtz/README.md](helmholtz/README.md#training-and-inference), [cdr/README.md](cdr/README.md#training), and [nonlinear/README.md](nonlinear/README.md#training).

## License

[MIT](LICENSE). The license also includes the notices for adapted MgNO and CompactImplicitHelmholtz components.
