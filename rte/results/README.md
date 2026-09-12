# Audited laptop experiments

## Handoff Snapshot

See the [workstation handoff](../../docs/research/RTE_WORKSTATION_HANDOFF_2026_09_12.md)
for the later collision/transport experiments, including the one-seed shared
coefficient ensemble. All available weights/settings in this results snapshot
are explicitly tracked for transfer; future `.pt`/JSON outputs remain ignored
by default. Actual ensemble materials/source parameters and a results hash
manifest are in `rte/handoff/2026-09-12`. Verify with
`python scripts/prepare_rte_handoff.py --verify` from the repository root.

## Earlier Per-Medium Experiments

These runs use the corrected working-tree code described in
`docs/research/PROJECT_AUDIT_2026_09_12.md`, based on revision `389b95d`.
They are CPU numerical observations from 2026-09-12, not GPU performance or
coefficient generalization experiments. Environment: Torch 2.8.0, one thread,
float32/complex64, size 16, 12 angles, fixed mixed medium.

- `residual_subspace_mixed`: three seeds, three learned arms, 300 updates each,
  ten deployed methods including paired classical residual-minimization controls.
- `residual_replay_mixed`: three seeds, same legacy network/raw loss, ordinary
  residual sampling versus 25% difficult-residual replay. Extra sampling cost
  is logged. Neither arm beats TSA4 in test solver quality.

Reproduce into fresh directories from the repository root:

```bash
python -m rte.run_residual_subspace_experiment --steps 300 --eval-every 50 \
  --seeds 1 2 3 --output /tmp/rte_subspace_replication
python scripts/summarize_residual_subspace.py /tmp/rte_subspace_replication
python -m rte.probe_residual_amplification /tmp/rte_subspace_replication

python -m rte.run_residual_subspace_experiment --models legacy_raw legacy_replay \
  --steps 300 --eval-every 50 --seeds 1 2 3 --output /tmp/rte_replay_replication
python scripts/summarize_residual_subspace.py /tmp/rte_replay_replication
python -m rte.probe_residual_amplification /tmp/rte_replay_replication --model-kind legacy_replay
```

The original commands used the tracked directory names as `--output`. The
runner now refuses nonempty output directories without `--overwrite`.
CSV histories contain full-bank train/validation physical losses and validation
FGMRES scores at each evaluation checkpoint. `results.csv` reports selected
checkpoints on train/validation/test banks and independent test-source solves;
it does not substitute the final training state for the selected state.
`angular_projected` denotes the native map trained with the projected objective;
`MR_angular_projected` is its intended projected deployment. `MR_` always denotes
additional residual-minimization work, including for classical controls.

Figures and summaries are derived, not separate experiments. Error bars are
seed ranges rather than confidence intervals; three test sources per seed are
not three independent media. Hardware timings are short measurements and
exclude training/setup. All solver A calls are counted, including safeguards.
Different forward-call counts are not equal time.

At the time of these runs, generated settings and checkpoints were ignored by
Git. Their locally available copies are now included in the handoff snapshot.
Historical remote model weights that were never present in this checkout are
still unavailable. Copying CSVs alone is not a trained-model transfer.
