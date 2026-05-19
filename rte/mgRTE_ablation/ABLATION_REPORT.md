# mgRTE Defect-CNN Revision Ablation

This folder is a self-contained NPBS-local copy of the small mgRTE components
needed to audit the coefficient-conditioned defect-CNN revisions.  The source
snapshot came from `/Users/hxchen/mgRTE`, but all ablation runs below were
executed from this NPBS repository folder.

## Revisions Tested

1. `MG_precond` implementation fixes:
   - replaced coefficient-path `BatchNorm2d` with `GroupNorm`;
   - resized prolongated coarse corrections before addition;
   - recomputed the up-sweep residual after prolongation;
   - detached setup features in eval mode.

2. `RTEUNetPrecond` defect-CNN changes:
   - kept the residual path linear in the RHS for fixed coefficients;
   - zero-initialized the output projection, so the learned correction starts
     exactly at zero;
   - added `preconditioner_form="defect2"`:
     \[
       z_0=B_h r,\qquad z_1=B_h(r-A_hz_0),\qquad
       M_\theta r=z_0+z_1+C_\theta(z_0,z_1,\kappa).
     \]

## Commands

```bash
cd /Users/hxchen/NeuralPreconditionedBornSeries/rte/mgRTE_ablation

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python -m py_compile \
  RTE_mgmodel.py rte_preconditioning.py ablate_defect_revisions.py

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python ablate_defect_revisions.py \
  --data-types diffusion transport interface \
  --num-coef 2 --mesh-size 4 --channels 4 --levels 2 \
  --train-steps 80 --lr 0.003
```

The CSV output is `outputs/defect_revision_ablation.csv`.

## Sanity Checks

The copied model passes the audit failure checks:

| Check | Result |
|---|---:|
| `MG_precond` odd grid `5x5` | finite output, correct shape |
| default `MG_precond(1)` on batch-1 `16x16` | finite output, correct shape |
| `RTEUNetPrecond` defect2 input, zero initialization | output norm `0.0` |
| `RTEUNetPrecond` fixed-coefficient RHS linearity | defect `0.0` |

## Ablation Results

`true_rel` is \(\|A_h u-r\|/\|r\|\).  `d_metric` is the repository's
`funcDinv`-metric residual.  `fallback` is the relative difference from the
intended classical fallback: `born1` for `unet_residual_zero`, and `born2` for
`unet_defect2_zero`.

| Case | Method | true_rel | d_metric | fallback | loss0 | loss1 |
|---|---|---:|---:|---:|---:|---:|
| diffusion | identity | 2.1806 | 1.1334 | 0.0000 |  |  |
| diffusion | born1 | 1.0787 | 1.5308 | 0.0000 |  |  |
| diffusion | born2 | 2.4463 | 5.5009 | 0.0000 |  |  |
| diffusion | unet_residual_zero | 1.0787 | 1.5308 | 0.0000 |  |  |
| diffusion | unet_defect2_zero | 2.4463 | 5.5009 | 0.0000 |  |  |
| diffusion | unet_residual_trained | 0.9615 | 0.9999 | 1.3608 | 0.5812 | 0.0699 |
| diffusion | unet_defect2_trained | 1.0443 | 1.6048 | 0.8718 | 5.9845 | 1.1085 |
| transport | identity | 1.3579 | 1.0359 | 0.0000 |  |  |
| transport | born1 | 1.3093 | 1.6595 | 0.0000 |  |  |
| transport | born2 | 3.5997 | 4.6898 | 0.0000 |  |  |
| transport | unet_residual_zero | 1.3093 | 1.6595 | 0.0000 |  |  |
| transport | unet_defect2_zero | 3.5997 | 4.6898 | 0.0000 |  |  |
| transport | unet_residual_trained | 0.9838 | 1.2267 | 1.2527 | 1.5294 | 0.5546 |
| transport | unet_defect2_trained | 2.6409 | 3.4277 | 0.5958 | 12.9576 | 7.1071 |
| interface | identity | 1.4948 | 1.1355 | 0.0000 |  |  |
| interface | born1 | 1.1236 | 1.5518 | 0.0000 |  |  |
| interface | born2 | 2.5869 | 4.5634 | 0.0000 |  |  |
| interface | unet_residual_zero | 1.1236 | 1.5518 | 0.0000 |  |  |
| interface | unet_defect2_zero | 2.5869 | 4.5634 | 0.0000 |  |  |
| interface | unet_residual_trained | 1.0537 | 1.4128 | 0.5961 | 0.7646 | 0.4941 |
| interface | unet_defect2_trained | 1.5942 | 2.4540 | 0.7384 | 6.6920 | 2.6093 |

## Interpretation

The zero-initialization revision is verified: the residual-form U-Net starts
exactly as `born1`, and the defect2 U-Net starts exactly as `born2`.

The `defect2` wiring is mathematically correct, but this ATFPS block-Jacobi
base is not a stable two-step Richardson preconditioner on these tiny cases:
`born2` is worse than `born1` in all three regimes.  Consequently, the defect2
U-Net improves substantially over its own unstable `born2` fallback after 80
steps, but it still does not beat the trained residual-form U-Net.

The strongest positive result is the residual-form coefficient-conditioned U-Net:
after 80 small CPU training steps it improves the true residual in all tested
regimes and beats `born1` on the true residual.  For example, in the transport
case it reduces `true_rel` from `1.3093` to `0.9838`.

The practical conclusion is:

- use `RTEUNetPrecond` as the coefficient-conditioned correction module;
- keep exact zero initialization;
- for the current ATFPS `funcDinv` base, prefer `preconditioner_form="residual"`
  as the first laptop-scale experiment;
- use `defect2` only after adding damping/relaxation or a stronger base
  preconditioner, because the raw two-step base can amplify the true residual.

## Larger-Scale Ablation

The larger ablation keeps the study laptop-safe but moves beyond the initial
\(4\times4\) toy run.  It uses five data regimes, more coefficient samples, and
larger grids:

```bash
cd /Users/hxchen/NeuralPreconditionedBornSeries/rte/mgRTE_ablation

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python ablate_defect_revisions.py \
  --data-types diffusion transport interface bufferzone all \
  --num-coef 4 --mesh-size 8 --channels 8 --levels 3 \
  --train-steps 120 --lr 0.003 --seed 20260519 \
  --output outputs/large_mesh8_ablation_seed20260519.csv

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python ablate_defect_revisions.py \
  --data-types diffusion transport interface bufferzone all \
  --num-coef 4 --mesh-size 16 --channels 8 --levels 3 \
  --train-steps 80 --lr 0.003 --seed 20260519 \
  --output outputs/large_mesh16_ablation_seed20260519.csv
```

The mesh-8 run took about 28 seconds.  The mesh-16 run took about 60 seconds.

During the larger ablation, the first diagnostic script used a double-corrected
residual-form training path.  The corrected files are
`outputs/large_mesh8_ablation_corrected_seed20260519.csv`,
`outputs/large_mesh16_ablation_corrected_seed20260519.csv`, and
`outputs/large_mesh16_ablation_relax025_corrected_seed20260519.csv`.

### Corrected Mesh \(8\times8\), Four Coefficients

| Case | Born1 | Born2 | Residual U-Net | Defect2 U-Net | Residual gain | Defect2 gain | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| diffusion | 1.1880 | 3.8190 | 0.4504 | 0.5177 | +62.1% | +56.4% | residual |
| transport | 1.4337 | 6.1143 | 0.5360 | 2.8300 | +62.6% | -97.4% | residual |
| interface | 1.3414 | 5.3886 | 0.4913 | 1.5095 | +63.4% | -12.5% | residual |
| bufferzone | 1.3109 | 4.5212 | 0.4706 | 1.4747 | +64.1% | -12.5% | residual |
| all | 1.4001 | 6.3858 | 0.5596 | 2.1604 | +60.0% | -54.3% | residual |

### Corrected Mesh \(16\times16\), Four Coefficients

| Case | Born1 | Born2 | Residual U-Net | Defect2 U-Net | Residual gain | Defect2 gain | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| diffusion | 1.2733 | 4.9726 | 0.8961 | 1.5877 | +29.6% | -24.7% | residual |
| transport | 1.7033 | 10.3759 | 1.1836 | 7.8402 | +30.5% | -360.3% | residual |
| interface | 1.3935 | 6.7655 | 0.9539 | 4.3541 | +31.5% | -212.4% | residual |
| bufferzone | 1.6757 | 9.2547 | 1.4115 | 5.7973 | +15.8% | -246.0% | residual |
| all | 1.5914 | 8.8937 | 1.1622 | 5.6810 | +27.0% | -257.0% | residual |

### Corrected Mesh \(16\times16\), Relaxed Defect2 with \(\omega=0.25\)

Here Born2 means \(z_0+0.25z_1\).

| Case | Born1 | Born2 | Residual U-Net | Defect2 U-Net | Residual gain | Defect2 gain | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| diffusion | 1.2733 | 2.3127 | 0.8961 | 0.7413 | +29.6% | +41.8% | defect2 |
| transport | 1.7033 | 4.6190 | 1.1836 | 3.4342 | +30.5% | -101.6% | residual |
| interface | 1.3935 | 3.0777 | 0.9539 | 1.9435 | +31.5% | -39.5% | residual |
| bufferzone | 1.6757 | 4.1856 | 1.4115 | 2.5603 | +15.8% | -52.8% | residual |
| all | 1.5914 | 4.0044 | 1.1622 | 2.2109 | +27.0% | -38.9% | residual |

### Larger-Scale Conclusion

The corrected larger ablation strengthens the main conclusion.  The residual
form
\[
  M_\theta r = B_h r + C_\theta(B_h r,\kappa)
\]
is the stable choice with the current ATFPS `funcDinv` base.  It improves true
residual over Born1 in every mesh-8 and mesh-16 regime tested.  The gain is
about \(60\%\)--\(64\%\) at mesh 8 and \(16\%\)--\(32\%\) at mesh 16.

The raw `defect2` architecture is not reliably competitive with this base.  The
underlying two-step Born fallback \(z_0+z_1\) is much worse than Born1 in every
tested regime.  Damping the second correction to \(z_0+0.25z_1\) helps and makes
defect2 best for mesh-16 diffusion, but it still fails in transport,
interface, bufferzone, and mixed cases.  This does not invalidate defect CNNs
in general; it says that the current ATFPS/Born base is too aggressive for an
undamped two-step defect wrapper.  A DSA/TSA or optimized relaxed base is still
needed before `defect2` should be the default.

\[
\boxed{\text{Use residual-form U-Net now; revisit defect2 only with a better or
damped }B_h.}
\]

## Reassessment of the Earlier Audit Issues

The ablations and sanity checks change the priority of the earlier issues:

1. \textbf{RTEUNetPrecond is the right current model.}  This is now supported
empirically.  The corrected mesh-8 and mesh-16 studies show consistent true
residual gains for the residual-form U-Net.

2. \textbf{The exact-zero output initialization is important and valid.}  The
zero U-Net reproduces the classical fallback exactly: residual-form zero equals
Born1, and defect2 zero equals the chosen \(z_0+\omega z_1\).

3. \textbf{The original `defect2` recommendation needs qualification.}  The
mathematical wiring is useful, but the current base makes \(z_0+z_1\)
unstable.  Defect2 should not be the default until \(B_h\) is damped, optimized,
or replaced by DSA/TSA.

4. \textbf{The MgNet implementation issues remain real.}  The wrong up-sweep
residual, size mismatch, and BatchNorm fragility were actual implementation
problems.  The NPBS-local copy fixes them, and the sanity checks pass on odd
grids and batch-one \(16\times16\) inputs.

5. \textbf{The coefficient channel issue remains real.}  The code expects
three channels \((\sigma_t,\sigma_a,\log\varepsilon)\).  Any two-channel
interface should be rejected or explicitly expanded.

6. \textbf{The setup/autograd issue remains real for inference.}  Setup should
be done under `torch.no_grad()` with `model.eval()` before FGMRES.

7. \textbf{Boundary padding is still unresolved.}  These ablations did not
isolate boundary effects.  Zero padding may be acceptable for the learned
correction, but production comparisons should add boundary masks or
boundary-aware padding.

8. \textbf{The selected/Born-coordinate loss needs caution.}  On tiny ATFPS
cases the selected metric loss can be uninformative while the true residual is
large.  The diagnostic ablation therefore trained against true residual.  Future
training should report both true residual and the selected/Born metric.
