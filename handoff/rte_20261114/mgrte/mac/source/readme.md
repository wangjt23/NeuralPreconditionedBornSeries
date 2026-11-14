# MgNet for Steady-State Radiative Transfer Equation

This repository provides the Python implementation of **MgNet** for solving the **steady-state radiative transfer equation (RTE)**. In this project, MgNet is trained and used as a **preconditioner** for the fully discretized linear system arising from the steady-state RTE.

The numerical discretization is based on:

- **DOM** (Discrete Ordinates Method) for angular discretization
- **ATFPS** for the discretization scheme used in the loss construction and linear system formulation

---

## Project Overview

The repository contains:

- data generation scripts for different physical regimes,
- model definitions for **CoeffNet** and **MgNet**,
- training code for MgNet,
- testing scripts for evaluating preconditioning performance,
- GMRES implementation in PyTorch,
- plotting utilities for visualization.

The main goal is to compare the performance of the following preconditioners for solving the fully discretized steady-state RTE system:

1. **Classical block Jacobi preconditioner**
2. **MgNet preconditioner trained with 10,000 samples**
3. **MgNet preconditioner trained with 20,000 samples**

---

## File Folder Description

### `discretized_parameters`

The dictionary `discretized_parameters` stores the following information:

1. **`quadrature`**  
   Discretized angular/velocity directions and their corresponding quadrature weights from DOM.

2. **`Kappa`**  
   The discretized kernel function.

3. **Dataset information for different physical regimes**, including:
   - `Coef`
   - `fsmLRBTC`
   - `I2A`
   - `MLRBT`
   - `VecSize`

   The last four items are generated from the **ATFPS discretization scheme**. They are derived from `Coef` and are prepared for the design of the loss function.

   > The dataset can be generated using `RTE_datagenerator.py`.

---

### `model_precond`, `train_loss`, and `val_loss`

These dictionaries store:

- the trained **MgNet model** (viewed as a preconditioner),
- the **training loss**,
- the **validation loss**

for each dataset/regime.

---

## File Description

### `train.py`
Training script for MgNet.

### `test.ipynb`
This notebook:

1. generates the test dataset,
2. visualizes the generated data,
3. applies **GMRES** with three different preconditioners to solve the fully discretized linear system from the steady-state RTE.

The three preconditioners are:

- classical block Jacobi preconditioner,
- MgNet trained with **10,000** samples,
- MgNet trained with **20,000** samples.

---

### `RTE_datagenerator.py`
Generates datasets for different physical regimes. The generated data include:

- `Coef`
- `fsmLRBTC`
- `I2A`
- `MLRBT`
- `VecSize`

The last four quantities are generated from `Coef` through the **ATFPS discretization scheme** and are used in the design of the loss function.

---

### `RTE_dataloader.py`
Wraps the generated data into a custom dataset so that it can be used conveniently with tools such as `DataLoader`.

---

### `RTE_mgmodel.py`
Defines and constructs **CoeffNet** and **MgNet**.

---

### `RTE_ATFPS_func.py`
Implements the construction of the loss function.

Key points:

- The loss is **physics-informed**.
- It uses the residual of the fully discretized system based on **TFPS/ATFPS**.
- The degree of compression in the loss function is automatically controlled by the threshold `tol_exp`.
- If `tol_exp` is very large, then no compression is applied.
- The compression behavior also depends on the optical properties of the background medium.

---

### `gmres.py`
PyTorch implementation of the **GMRES** algorithm.

---

### `plot.py`
Provides utilities for plotting 2D images of:

- angular flux,
- scalar flux.

---

### `utils.py`
Contains common utility functions, including:

- saving models,
- saving training/validation losses,
- checking Chinese font support,
- and other helper functions.

---

### `someplot.ipynb`
Contains additional plotting code for producing cleaner and more visually appealing training/validation loss figures.

---

## Usage

### 1. Install

Use either the Python project metadata:

```bash
python -m pip install -e ".[dev]"
```

or the Conda environment:

```bash
conda env create -f environment.yml
conda activate mgRTE
python -m pip install -e ".[dev]"
```

### 2. Generate Data

Generated `.pt` datasets are intentionally ignored by git. Generate the dataset that matches the training config:

```bash
python RTE_datagenerator.py --config configs/default.yaml
```

You can override individual fields:

```bash
python RTE_datagenerator.py --data_type diffusion --num_coef 1000 --mesh_size 16 --N 1 --tol_exp 5 --kernel_g 0
```

### 3. Train MgNet

```bash
python train.py --config configs/default.yaml
```

The default config preserves the original raw-residual objective, but now uses
the NPBS-style training mechanics: fresh random residuals per batch and relative
residual normalization. The older fixed-RHS/MSE behavior is still available with
`--rhs_mode fixed --loss_normalization mse`.

The NPBS-style RTE objective trains in the block-Jacobi/Born coordinate
`funcDinv(rhs)`:

```bash
python train.py --config configs/born_precond.yaml
```

If the Born-coordinate MgNet model does not improve FGMRES, try the
NPBS-style gated U-Net architecture. It is linear in the residual for fixed
coefficients and is intended to learn a small residual correction over
block-Jacobi:

```bash
python train.py --config configs/born_unet_precond.yaml
```

Training saves model weights and train/validation losses under `model_precond/`, `train_loss/`, and `val_loss/`.

### 4. Evaluate Preconditioners

```bash
python inference.py --config configs/default.yaml --checkpoint model_precond/CosineAnnealingLR_T_max100/lr0.001/<checkpoint>.pt
```

To compare the original raw MgNet preconditioner with the Born-coordinate variant:

```bash
python inference.py --config configs/born_precond.yaml --checkpoint model_precond/CosineAnnealingLR_T_max100/lr0.001/<checkpoint>.pt --preconditioners block_jacobi mgnet_raw mgnet_born
```

For a U-Net checkpoint, pass the U-Net config or `--model_type unet` during
inference so the checkpoint is loaded into the same architecture.

The inference script compares right preconditioners with FGMRES and writes a JSON summary under `outputs/`. It reports both true relative residual and the final `funcDinv`-metric residual.

### Experimental Pseudo-Spectral Operator

`RTE_pseudospectral.py` adds a separate periodic DOM pseudo-spectral backend with state shape `[batch, 4*M, I, J]`. It is intended for smooth/reference experiments and FFT Green preconditioning, not as a replacement for ATFPS on interface or diffusion-limit cases.

### 5. Test

```bash
python -m pytest
```

The current tests cover importability, deterministic RHS sampling, MgNet forward shape, variable channel MgNet shape, ATFPS masks, the `all` regime epsilon branch, the independent y-direction eigenbasis, FGMRES, raw-vs-Born preconditioning comparison, and pseudo-spectral reference identities.
