# Training Scheme Audit

## Conclusion

The current training direction is valid after two corrections:

- Use fresh random residuals per batch, matching the NPBS Helmholtz training loop.
- Use relative residual losses, not unnormalized MSE, so the objective is invariant to residual amplitude and selected-mode count.

The raw ATFPS loss is still available as a baseline, but the recommended production experiment is:

```bash
python train.py --config configs/born_precond.yaml
python inference.py --config configs/born_precond.yaml --checkpoint <checkpoint> --preconditioners block_jacobi mgnet_raw mgnet_born
```

## NPBS Pattern

The Helmholtz NPBS training loop samples a fresh random right-hand side each batch:

```python
rhs = torch.randn(..., dtype=torch.complex64, device=device)
```

It then uses three loss families:

- `loss_type=1`: raw physical residual, `||rhs - A(out)|| / ||rhs||`.
- `loss_type=2`: network input is the Green-coordinate residual `G(rhs)`, but the loss remains physical residual.
- `loss_type=3`: both input and loss are in the Green coordinate, `||G(rhs) - G(A(out))|| / ||G(rhs)||`.

The important lesson is not the Helmholtz scalar formula itself. The important lesson is the training geometry: the network must see the same residual representation during training that it will receive as a Krylov preconditioner.

## mgRTE Mapping

For ATFPS:

```text
A  = func_inflow_Dirichlet
D  = funcD
G0 = D^{-1} = funcDinv
```

The recommended Born-coordinate training path is:

```text
q      = funcDinv(rhs)
z      = MG_precond(q)
z_corr = correction(z, rhs_unselected)
loss   = ||funcDinv(A z_corr) - q|| / ||q||
       + lambda * ||A z_corr - rhs|| / ||rhs||
```

The corresponding inference preconditioner is:

```text
M(r) = correction(MG_precond(funcDinv(r)), r_unselected)
```

This is now implemented in `rte_preconditioning.py`, exposed in `train.py` as `loss_type=born_mixed`, and exposed in `inference.py` as `mgnet_born`.

## Fixes Applied

- `train.py` now supports `--rhs_mode random|fixed`.
- `train.py` now supports `--loss_normalization relative|mse`.
- Defaults use `rhs_mode=random` and `loss_normalization=relative`.
- `UnsuperviseDataset` skips storing the full fixed RHS tensor when `rhs_mode=random`.
- `rte_config.run_name` now includes loss type, RHS mode, and normalization to avoid checkpoint collisions.
- Tests include a synthetic ATFPS block-diagonal case where the Born-coordinate objective is exact and raw residual input is not.
- Tests include an optimizer check showing the Born-coordinate loss has an effective gradient on a tiny trainable model.

## Remaining Validity Checks For Real Experiments

Before trusting a large run, check:

- The first few batches have finite `raw`, `born`, and `born_mixed` losses.
- Gradient norm is finite and nonzero.
- `||funcDinv(rhs)|| / ||rhs||` is not exploding for most samples.
- Validation uses the same residual coordinate as inference.
- FGMRES compares `block_jacobi`, `mgnet_raw`, and `mgnet_born` on the same coefficients and RHS seeds.
- Report both true residual and `funcDinv`-metric residual.

If random residual validation looks good but FGMRES does not improve, add replay residuals from partial block-Jacobi or FGMRES solves. That is the next likely improvement, because Krylov residuals are not exactly distributed like uniform random RHS tensors.
