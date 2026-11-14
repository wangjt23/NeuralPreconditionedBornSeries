# Pilot stop state, September 26, 2026

All 1,000 optimizer updates completed. `best.pt` selects update 1,000;
`last.pt` also preserves the optimizer, scheduler and RNG state at update
1,000. Validation on eight media improved from 9.375 to 9.0 TSA8 outer
iterations, with no failed solves. This is validation, not held-out evidence.

The preliminary classical timing sweep was deliberately interrupted with
SIGINT after eight of sixteen validation media, before the test-set stage.
`validation_cases.csv` is partial and must not be treated as a completed
benchmark or used to select a final comparison. No test data were generated
or evaluated by this run. `settings.json` consequently has no completion flag.

Continuation: `source_predictor_terminal6_20260926`, using these selected
weights and the identical saved teacher bank. This is a weights-only
continuation with a deliberately reset optimizer, scheduler and sampling RNG,
not a bitwise restart. The new objective differentiates through six TSA8
FGMRES steps, targets the original-RHS tolerance of 1e-5, and uses a smaller
learning rate. Classical validation tuning expands to fourteen candidates.
The continuation uses a fresh, precommitted test-seed offset of 4,000.

The partial CSV, completed checkpoints and teacher bank are preserved.
