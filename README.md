# Preconditioned Neural PDE Solvers

This repository contains implementations of Born-Series-Inspired Residual Metric for Learning-based Preconditioners.

## Structure

```
├── helmholtz/    # Helmholtz equation
├── cdr/          # Convection-Diffusion-Reaction equation
└── nolinear/     # Nonlinear Poisson equation
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, Matplotlib

## Usage

```bash
# Train Helmholtz solver
python helmholtz/PAFNO/train2d.py --config config.yaml

# Train CDR solver
python cdr/train_solver.py

# Train nonlinear solver
python nolinear/newton.py --model_type FNO_solver --epochs 500
```
