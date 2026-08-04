# Large Angular-Spectral RTE Experiments

This note defines the first large-scale experiments for deciding whether the
weak RTE results are caused by insufficient coefficient data, insufficient
optimization, or an inadequate learned operator class.

## Why scale the data

The previous large run used 5,400 optimizer updates, 256 residuals per update,
24 rollout media, and 8 auxiliary sources. Its rollout bank therefore had only
216 coefficient/source environments, and the coefficient medium was still
fixed during training. The selected checkpoint was step zero. This is evidence
against treating more epochs as the only missing ingredient: the learned map
was not being asked to represent a broad coefficient-conditioned operator.

The new driver samples thousands of distinct material fields and source
parameters. It varies total attenuation, albedo, inclusion geometry, layered
interfaces, source location, source amplitude, and angular source profile.
The anisotropy parameter `g` is held fixed within an ensemble because the
current `PeriodicRTE2D` implementation constructs one angular phase matrix per
operator instance. This is a controlled coefficient-conditioning experiment,
not yet a full phase-function generalization experiment.

## Priority order

The runner is `rte/run_large_angular_spectral_experiments.py`.

1. `fixed_mixed_sample_scale_4096_2000` uses 4,096 randomized samples from the
   mixed regime with the angular-spectral defect model. It isolates sample
   count and optimizer effort while keeping the physical regime comparable.
2. `coefficient_ensemble_4096_2000` uses the same sample and update budget over
   diffusive, heterogeneous, anisotropic, and mixed regimes. It tests whether
   coefficient diversity is the missing signal.
3. `coefficient_ensemble_unet_control_4096_2000` repeats the ensemble study
   with the earlier `UNet-defect2` architecture. This is the matched control
   needed to distinguish data/training limitations from angular representation.
4. `coefficient_ensemble_capacity_8192_3000` increases both data and model
   capacity. It should be run only if the first three runs show a meaningful
   held-out trend, since it is the most expensive configuration.

The first three runs use 2,000 epochs and two updates per epoch. The number of
minibatch sample presentations is

\[
  N_{\rm presentations}
  = N_{\rm epochs}\,N_{\rm steps/epoch}\,N_{\rm batch}
  = 256{,}000
\]

for the default batch size 64. The fourth run uses 192,000 presentations with
batch size 32. The dataset size and presentation count are recorded separately
in `config.json`, `history.csv`, and `result.json`.

## Training objective and residual distribution

The default objective is `solver_teacher`. For a sampled residual `r`, the
raw term measures the true physical residual after the learned correction,

\[
  \rho_{\rm raw}(r)
  = \frac{\|r-A_h M_\theta r\|_2}{\|r\|_2},
\]

where `A_h` is the current discrete transport operator and `M_theta` is the
learned preconditioner. The rollout term applies the same learned map for a
short sequence of residual updates. The teacher term compares with a depth-4
transport-DSA correction, and its action term measures the physically relevant
defect

\[
  \frac{\|A_h(M_\theta r-T_4r)\|_2}{\|r\|_2}.
\]

The teacher weight is ramped over the first 200 epochs. The residual bank is
not just the source: each training batch samples a stage from a depth-4
transport-DSA trajectory, and half of the batch is replaced by low-frequency
random probes. This gives the network exposure to the residual subspaces that
matter after several solver iterations while preserving controlled probe
coverage.

The evaluation uses independently generated coefficient/source samples and
reports both held-out one-step residual reduction and held-out restarted
FGMRES residual after the configured iteration budget. A one-step gain without
an FGMRES gain is not accepted as evidence of a useful preconditioner.

## Remote commands

Print the prioritized configurations:

```bash
python -m rte.run_large_angular_spectral_experiments --plan priority
```

Run all prioritized experiments on a CUDA device:

```bash
python -m rte.run_large_angular_spectral_experiments \
  --plan priority --execute --device cuda:0 \
  --output-root rte/large_angular_spectral_runs
```

Run one experiment with a reusable compact dataset cache:

```bash
python -m rte.run_large_angular_spectral_experiments \
  --name coefficient_ensemble_4096_2000 \
  --dataset-size 4096 --eval-size 32 --size 64 --n-angles 12 \
  --regimes all --batch-size 64 --epochs 2000 --steps-per-epoch 2 \
  --model-kind angular_spectral_defect2 \
  --save-dataset rte/large_angular_spectral_runs/dataset_4096_size64.pt \
  --device cuda:0 --execute
```

The cache stores coefficient fields and compact source parameters, not all
complex source arrays. A run writes `config.json`, `history.csv`,
`checkpoint.pt`, and `result.json`. The best checkpoint is selected by held-out
FGMRES residual. A run can resume from a checkpoint with `--resume`; the
checkpoint includes model, optimizer, scheduler, epoch, and best-score state.

## Interpretation gates

* If the fixed mixed sample-scale run improves while the ensemble run does not,
  optimization noise or coefficient nonstationarity is limiting the current
  conditioning mechanism.
* If both angular and old-UNet controls improve similarly, the prior negative
  result was primarily a data/training-budget failure.
* If the angular model improves on the ensemble while the old UNet does not,
  the retained angular modes and mode coupling are likely useful.
* If one-step loss falls substantially but held-out FGMRES does not improve,
  the objective or residual distribution is misaligned with non-normal Krylov
  behavior; increasing epochs is not the next remedy.
* If neither architecture improves after the first three runs, stop increasing
  epochs. The next experiments should target teacher quality, stability bounds,
  phase-function conditioning, or a learned restriction/prolongation rather
  than more samples alone.
