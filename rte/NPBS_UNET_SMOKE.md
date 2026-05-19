# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive']`
- `sizes`: `[16]`
- `n_angles`: `8`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['residual', 'defect2']`
- `defect_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `8`
- `levels`: `2`
- `steps`: `20`
- `lr`: `0.002`
- `maxiter`: `20`
- `fgmres_maxiter`: `10`
- `tol`: `1e-06`
- `seed`: `20260519`
- `output_csv`: `rte/npbs_unet_smoke.csv`
- `output_md`: `rte/NPBS_UNET_SMOKE.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 16 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 4.723e-06 | 8.734e-06 |  |  |
| 16 | transport_dsa | diffusive | train | dsa2 | 4.220e-01 | 4.252e-01 | 4.788e-06 | 5.411e-06 |  |  |
| 16 | transport_dsa | diffusive | train | UNet-residual-w1 | 9.523e-01 | 9.396e-01 | 4.630e-06 | 6.105e-06 | 0.3384934663772583 | 0.09031021595001221 |
| 16 | transport_dsa | diffusive | train | UNet-defect2-w1 | 1.874e-01 | 1.879e-01 | 4.874e-06 | 5.908e-06 | 0.01322499755769968 | 0.004582657478749752 |
| 16 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 4.551e-06 | 6.466e-06 |  |  |
| 16 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 4.041e-06 | 5.300e-06 |  |  |
| 16 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 8.547e-01 | 8.430e-01 | 4.281e-06 | 5.483e-06 | 0.3384934663772583 | 0.09031021595001221 |
| 16 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 1.735e-01 | 1.746e-01 | 4.283e-06 | 5.251e-06 | 0.01322499755769968 | 0.004582657478749752 |
