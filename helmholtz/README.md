# Helmholtz

Run the commands below from the repository root.

## Data

The experiments use velocity models from [OpenFWI](https://openfwi-lanl.github.io/):

```text
data/openfwi/
├── CurveVel_A/model/       # model1.npy, model2.npy, ...
├── CurveFault_B/           # vel6_1_0.npy, vel6_1_1.npy, ...
└── 3DKimberlina-V1/        # year0_cut1.bin, ..., year100_cut63.bin
```

The 2D arrays have shape `(500, 1, H, W)` and are interpolated to the configured grid. CurveFault-B files follow `vel{file_index // 36 + 6}_1_{file_index % 36}.npy`. The learned-model presets use 3,000 training samples, and every inference preset evaluates 50 samples. Training starts at sample 0; `val_point` and `test_point` are zero-based offsets across files.

The 3D files contain float32 volumes of shape `(400, 400, 350)`. Training takes the first 600 volumes in year/cut order from years `[0, 1, 2, 5, 10, 15, 20, 25, 30, 35]`, cuts 1–63; validation and testing both use cuts 1–50 from year 100. Volumes are loaded and resized individually.

In the YAML files, `train.num_test` is the validation count: 100 in 2D and 50 in 3D. `inference.num_test` is the evaluation count: 50 in both dimensions. The current validation and test samples overlap; in 2D, each test set is the first 50 samples of its 100-sample validation set. MGCFNN retains offset 4,000, while the other 2D presets use offset 6,000.

Use `--data-dir` to specify the velocity directory directly, or set `NPBS_DATA_ROOT` to replace the default `data/` root. Physical parameters and grid sizes are defined in [dataloader.py](dataloader.py) and can be overridden in the YAML `data` section.

## Training and inference

2D FNO on CurveFault-B:

```bash
python helmholtz/train2d.py \
  --config helmholtz/configs/2d/curvefault_b/pafno_bs_pre.yaml --device cuda:0
python helmholtz/inference2d.py \
  --config helmholtz/configs/2d/curvefault_b/pafno_bs_pre.yaml --device cuda:0
```

3D MgNO on Kimberlina-V1:

```bash
python helmholtz/train3d.py \
  --config helmholtz/configs/3d/mgno_bs_pre.yaml --device cuda:0
python helmholtz/inference3d.py \
  --config helmholtz/configs/3d/mgno_bs_pre.yaml --device cuda:0
```

Weights are saved as `<model.name>_weights.pth` under `train.save_dir`. Inference reads this file by default; `--checkpoint` overrides it. Use `--no-plots` to skip plots and `--help` for other arguments.

## Loss settings

| `loss_type` | Config suffix | Network input | Residual used in the loss |
| --- | --- | --- | --- |
| `1` | `nopre` | `r` | `r - A M(r)` |
| `2` | `nopreloss` | `G(r / dx²)` | `r - A M(G(r / dx²))` |
| `3` | `pre` | `G(r / dx²)` | `G((r - A M(G(r / dx²))) / dx²)` |

Each loss is normalized by the corresponding input residual norm over the batch. Train each variant separately. The available backbones are FNO, MgNO, U-Net, Encoder–solver, and an external MGCFNN adapter.

The intended 2D setting uses 32 absorbing cells on each side (`boundary_type: pbl`), giving a padded grid of 320 × 320; the 3D presets use 16 cells on each side. Training uses AdamW. The FNO presets for all three losses on CurveVel-A and CurveFault-B use 400 epochs with a factor of 0.8 decay every 20 epochs. The other 2D learned-model presets use 200 epochs with a factor of 0.8 decay every 10 epochs, including the training settings in the MgNO-FGMRES preset. All 2D learned-model presets use an initial learning rate of `1e-3`. The 3D presets use 200 epochs and retain a factor of 0.8 decay every 20 epochs. All supplied FGMRES presets use a restart length of 50. The manuscript's Adam optimizer and its 400-epoch/every-20-epoch schedule for the other 2D backbones still differ from these settings.

## FGMRES

CBS with FGMRES requires no checkpoint:

```bash
python helmholtz/inference2d.py \
  --config helmholtz/configs/2d/curvefault_b/cbs_gmres.yaml --device cuda:0
```

For NPBS with FGMRES, first train with `mgno_bs_pre.yaml`, then run:

```bash
python helmholtz/inference2d.py \
  --config helmholtz/configs/2d/curvefault_b/mgno_bs_pre_gmres.yaml \
  --checkpoint outputs/helmholtz/2d/openfwi/CurveFault_B/ppw12/mgno_bs_pre/mgno_bs_weights.pth \
  --device cuda:0
```

`krylov_maxiter` limits total iterations; `krylov_restart` sets the restart length. Stopping uses the physical relative residual. Failed solves retain their residual histories; reported averages cover converged samples. Fixed-point timings exclude residual checks, while FGMRES timings include them.

## MGCFNN

The [MGCFNN implementation](https://gitee.com/xiehuohuo77/mgcfnn) is not included because its upstream repository does not specify a license. To use a separately obtained compatible implementation, set `NPBS_MGCFNN_MODULE` to its module name. The adapter expects `MyModel(in_channels, out_channels, coeff_in_channels, flow_param)` with `setup(coeff)` and `forward(rhs)` methods; Fourier weights must be recomputed during training.
