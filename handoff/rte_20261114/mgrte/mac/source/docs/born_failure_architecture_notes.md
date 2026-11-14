# Why the Born-Coordinate Direction Underperformed

## Observed Failure Mode

The small diffusion experiments show the same pattern:

- the Born-coordinate objective can become small;
- the true ATFPS physical residual stays much larger;
- FGMRES with `mgnet_born` does not beat `block_jacobi`, and can be worse than the raw model.

This means the issue is not only syntax or train-loop plumbing. The learned map is improving the metric it sees during training, but it is not producing a correction that helps the non-normal ATFPS operator under true-residual FGMRES.

## Likely Reasons

1. `funcDinv` is already a strong local preconditioner.

   In diffusion tests, block Jacobi is often hard to beat. A direct neural output can easily damage this reference inverse. The safe formulation is residual-over-reference:

   ```text
   M(r) = q + N_theta(q),  q = funcDinv(r)
   ```

   rather than asking the network to relearn the whole correction.

2. The Born metric is not equivalent to the true residual metric.

   Training minimizes roughly

   ```text
   ||funcDinv(A z) - funcDinv(r)|| / ||funcDinv(r)||
   ```

   but FGMRES stops on

   ```text
   ||A z - r|| / ||r||.
   ```

   For a non-normal transport/interface operator, those can disagree. Increasing the physical loss weight is therefore important.

3. Random residuals do not match Krylov residuals.

   Uniform random RHS tensors are useful for operator learning, but FGMRES produces structured residuals. Replay residuals from block-Jacobi or FGMRES are the right next data source.

4. The old MgNet architecture is too constrained for the correction it needs to learn.

   After `funcDinv`, the remaining error is mostly nonlocal neighbor/interface propagation. The old V-cycle can be effective, but it has limited skip-path capacity and no explicit safe reference skip.

## Architecture Change

`RTEUNetPrecond` was added in `RTE_mgmodel.py`, following the NPBS design principles:

- nonlinear coefficient encoder;
- residual path is exactly linear for fixed coefficients;
- coefficient features enter multiplicatively as gates;
- U-Net skip connections provide larger nonlocal correction capacity;
- small output scale keeps the learned correction near block Jacobi at initialization.

Use it with:

```bash
python train.py --config configs/born_unet_precond.yaml
```

The intended comparison is:

```text
block_jacobi
mgnet_born with model_type=mgnet
mgnet_born with model_type=unet
```

The naming `mgnet_born` in inference means "learned Born preconditioner"; with `--model_type unet`, the loaded model is the U-Net preconditioner.

## Recommended Experiment

Run the same diffusion slice with:

```bash
python train.py --config configs/born_unet_precond.yaml \
  --num_coef 32 --num_data 512 --batch_size 32 --num_epochs 100 \
  --u_channels_list 8 16 32 --a_channels_list 8 16 32 --mg_levels 2 2 2 \
  --unet_channels 16 --unet_levels 3
```

Then evaluate with:

```bash
python inference.py --config configs/born_unet_precond.yaml \
  --checkpoint <checkpoint> --num_test 8 --max_iter 30 --restart 30
```

If this still fails to beat block Jacobi, the architecture is probably not the main bottleneck; prioritize FGMRES replay residuals and stronger physical-residual weighting.
