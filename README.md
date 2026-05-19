# Neural Preconditioned Born Series (NPBS)

Code for the paper:

> **Neural Preconditioned Born Series: A Metric-Matched Framework for Learning-based Preconditioners**

NPBS is a learned iterative preconditioning framework that uses the Convergent Born Series (CBS) as a preconditioned residual coordinate system. The key idea is to replace the scalar Born correction with a learned residual-to-correction map in Born-preconditioned coordinates, and to use the Green-operator-induced residual metric as the training objective — aligning the geometry used during training with the geometry used during inference.

## Overview

The framework is validated on three PDE families:

| Module | PDE | Key feature |
|--------|-----|-------------|
| `helmholtz/` | High-frequency Helmholtz in heterogeneous media | CBS preconditioner + learned correction; stationary and FGMRES modes |
| `rte/` | Steady-state radiative transfer prototype | Transport-resolvent NPBS/Born coordinates for discrete ordinates |
| `cdr/` | Convection–diffusion–reaction (steady-state) | CBS-style preconditioned Richardson iteration with periodic BCs |
| `nolinear/` | Newton linearizations of nonlinear Poisson | Preconditioned inner-loop solver for Newton's method |

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, Matplotlib
- PyYAML
- h5py (for HDF5 datasets)

## Helmholtz Module

### Dataset

The Helmholtz experiments use two datasets(CurveVel_A, CurveFault_B) from [OpenFWI](https://smileunc.github.io/projects/openfwi), a benchmark suite for seismic full waveform inversion. We only need the velocity map here.

### Models

| Model | Preconditioner backbone |
|-------|------------------------|
| `PAFNO_BS` | Physics-aware FNO (`Fourier_precond`) |
| `MGCFNN_BS` | Multigrid compact-FNN (`MGCFNN_precond`) |
| `MGNO_BS` | MgNO (`MGNO_precond`) |
| `UNet_BS` | U-Net (`UNet_precond`) |
| `EncoderSolver_BS` | Encoder-solver (`EncoderSolverPrecond`) |
| `ShiftedLaplacianVcycle2d` | Classical shifted-Laplacian geometric V-cycle |

The reference solver `ConvergentBornSeries2d` (and `3d`) in `models/cbs.py` implements the classical CBS iteration:

```
u ← u − γ (u − G(V·u + b/dx²))
```

where `G` is the reference Green operator applied via FFT, `V` is the scattering potential, and `γ` is the CBS preconditioner coefficient.

When CBS is used inside FGMRES, `inference.cbs_krylov_preconditioner` controls the right-preconditioner map:

| Value | Map applied to residual `r` | Interpretation |
|-------|------------------------------|----------------|
| `gamma_green` (default) | `γ G(r/dx²)` | CBS correction map used by the stationary CBS iteration |
| `green` | `G(r/dx²)` | pure Born / constant-reference shifted-Laplacian inverse |

The `green` mode is the direct code-level counterpart of the paper identity `I - G_eta V_eta = G_eta A`; the default preserves existing CBS-FGMRES runs.

The baseline shifted-Laplacian preconditioner in `models/shifted_laplacian.py` uses the same padded Helmholtz discretization and ABC/PBL setup as CBS, then applies a standard geometric V-cycle to the complex-shifted operator. Example FGMRES configs are available at:

```bash
cd helmholtz
python inference2d.py --config configs/2d/openfwi/CurveFaultB/sl_vcycle_gmres.yaml
python inference2d.py --config configs/2d/openfwi/CurveVelA/sl_vcycle_gmres.yaml
python inference2d.py --config configs/2d/openfwi/CurveFaultB/cbs_gmres_green.yaml
```

The V-cycle shifted-Laplacian classes are classical variable-coefficient shifted Helmholtz preconditioners. They are useful baselines, but they are not identical to the constant-coefficient FFT reference operator `G` used by CBS/NPBS in the paper's Born-coordinate equivalence.

To verify the discrete identity used by the CBS/NPBS implementation:

```bash
cd helmholtz
python verify_operator_identities.py
```

For a dataset-independent check, `verify_shifted_laplacian.py` generates a smooth moderate-contrast squared-slowness field `κ = 1/c²` and verifies FGMRES convergence with the V-cycle preconditioner:

```bash
cd helmholtz
python verify_shifted_laplacian.py
```

To reproduce a small CBS-vs-shifted-Laplacian wall-time and convergence comparison with plots:

```bash
cd helmholtz
python compare_cbs_shifted_laplacian.py
```

### Loss types (`loss_type` in YAML)

All three modes train the preconditioner `P` to approximate `A⁻¹b` given a right-hand side `b`. They differ in the coordinate system used for the preconditioner input and the training objective.

| Value | Input to `P` | Training loss | Description |
|-------|-------------|---------------|-------------|
| `1` | `b` (physical) | `‖b − A·P(b)‖ / ‖b‖` | Physical-coordinate input, physical residual loss |
| `2` | `G(b/dx²)` (Born) | `‖b − A·P(G(b/dx²))‖ / ‖b‖` | Born-coordinate input, physical residual loss |
| `3` | `G(b/dx²)` (Born) | `‖G(b/dx²) − G(A·P(G(b/dx²))/dx²)‖ / ‖G(b/dx²)‖` | Born-coordinate input, Born-metric loss (metric-matched) |


### Training

Training is fully config-driven. Each YAML file specifies the model, dataset, and training hyperparameters.

```bash
cd helmholtz
python train2d.py --config configs/2d/openfwi/CurveVelA/ppw14/pafno_bs_pre.yaml
```

Key YAML fields:

```yaml
train:
  seed: 1234              # optional reproducibility seed
  num_epochs: 400
  batch_size: 30
  lr: 5e-4
  lr_gamma: 0.8
  lr_step_size: 20
  num_train: 3000
  num_test: 100
  save_dir: "./outputs/..."

model:
  name: pafno_bs          # model type
  boundary_width: 32
  boundary_type: pbl      # pbl or abc
  loss_type: 3            # 1=L2, 2=precond-L2, 3=precond-R_eta
  data_type: openfwi_CurveVel_A14

inference:
  fdtype: float64
  max_num_iters: 2000
  tol: 1e-6
  num_test: 50
  drop_first_timing: false # optionally exclude first solved case from average time
```

Available dataset types:  `openfwi_CurveVel_A`, `openfwi_CurveVel_B`, `openfwi_FlatVel_A`, `openfwi_CurveFault_B`, and their `_pad` / `_inter` / `_ppwXX` variants.

### Inference

```bash
cd helmholtz
python inference2d.py --config configs/2d/openfwi/CurveVelA/ppw14/pafno_bs_pre.yaml \
                      --checkpoint ./outputs/.../best_model.pt
```


## CDR Module

The CDR module applies the same metric-matched preconditioning principle to the steady-state convection–diffusion–reaction equation:

```
−∇·(κ∇u) + v·∇u + σu = f
```

The reference preconditioner `ConvectionDiffusionCBS` in `cdr/cd_cbs.py` uses a constant-coefficient reference operator `L₀` inverted via FFT (periodic BCs).

### Dataset

The CDR dataset is synthetically generated by `cdr/data_generate.py`. Each sample contains:

| Field | Description | Shape |
|-------|-------------|-------|
| `kappa` | Diffusion coefficient (GRF, strictly positive) | `(128, 128)` |
| `v` | Incompressible convection field (derived from a stream-function GRF) | `(128, 128, 2)` |
| `f` | Source term (localized Gaussian blobs) | `(128, 128)` |

An `ill_level ∈ [0, 1]` parameter controls problem difficulty: higher values produce stronger coefficient contrast and convection dominance. Default splits: 5000 training samples and 500 test samples at `ill_level=0.4`.

```bash
cd cdr
python data_generate.py   # generates data_ill/train/ and data_ill/test/
```

### Training

```bash
cd cdr
python train_solver.py
```

The training script loads coefficient fields `(κ, v_x, v_y)` from `ConvectionDiffusionDataset`, samples random right-hand sides, and minimizes the preconditioned residual loss.

### Inference

```bash
cd cdr
bash run_infer.sh
# or directly:
python inference.py --model_path <checkpoint> --max_iter 200 --tol 1e-6
python inference_cmp.py --model_path <checkpoint>   # compare against baselines
```

### Baselines

`cdr/baseline/` provides: `FNO_solver`, `FNO2d`, `UNet`, `ResNet`, `MgNO`, `MPMG`.

## Nonlinear Module

The nonlinear module trains a preconditioned inner-loop solver for Newton linearizations of a nonlinear Poisson equation. At each Newton step the linearized system `J(u^k) δu = −F(u^k)` is solved with the learned preconditioner.

### Dataset

Data and experimental setup follow [Newton-Informed-Neural-Operator](https://github.com/xlliu2017/Newton-Informed-Neural-Operator). Download the data files from that repository and place them under `nolinear/data/`:

- `newton_method_iterations_5.pt` 
- `newton_method_iterations_7_solution.pt`

### Model variants

| Model type | Loss | Description |
|------------|------|-------------|
| `FNO_solver` | direct | FNO solving `Aδu = rhs` directly |
| `FNO_solver_precond_bs_l_2` | precond-L2 | FNO with preconditioned L2 loss |
| `FNO_solver_precond_bs_R_eta` | precond-R_η | FNO with Born-metric loss (recommended) |

### Training

```bash
cd nolinear

# Direct loss (baseline)
python newton.py --model_type FNO_solver \
                 --final_div_factor 50 --weight_decay 1e-6 \
                 --lr 1e-4 --batch_size 50 --epochs 200 --loss_type solver

# Preconditioned L2 loss
python newton.py --model_type FNO_solver_precond_bs_l_2 \
                 --final_div_factor 50 --weight_decay 1e-6 \
                 --lr 1e-4 --batch_size 50 --epochs 200 --loss_type solver

# Born-metric loss (metric-matched, recommended)
python newton.py --model_type FNO_solver_precond_bs_R_eta \
                 --final_div_factor 50 --weight_decay 1e-6 \
                 --lr 1e-4 --batch_size 50 --epochs 200 --loss_type solver
```

Data is loaded from `nolinear/data/newton_method_iterations_5.pt` and `newton_method_iterations_7_solution.pt` (Reference [Newton-Informed-Neural-Operator](https://github.com/xlliu2017/Newton-Informed-Neural-Operator)).

### Inference

```bash
cd nolinear

# Evaluate the linearized inner loop (fixed Jacobian)
python inference_innerloop.py \
    --model_path "model/FNO_solver_precondns_merge<timestamp>.pt" \
    --batch_size 1 --max_iter 100 --tol 1e-4

# Evaluate the full Newton iteration
python inference_newton.py \
    --model_path "model/FNO_solver_precondns_merge<timestamp>.pt" \
    --sample_index 5001 \
    --max_newton_iter 10 --newton_tol 1e-4 \
    --linear_solver_max_iter 100 --linear_solver_tol 1e-4

# Compare against DST-preconditioned baseline
python inference_dst.py --log_dir ./dst.log --batch_size 1 --max_iter 100 --tol 1e-4
