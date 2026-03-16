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

# Train linearized newton system solver

python nolinear/newton.py --model_type FNO_solver_precond_bs_R_eta \
                          --final_div_factor 50 \
                          --weight_decay 1e-6 \
                          --lr 1e-4   \
                          --batch_size 50 \
                          --epochs 200 \
                          --loss_type solver
```
