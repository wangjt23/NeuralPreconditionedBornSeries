# Lightweight handoff validation

Earlier isolated 564 MB snapshot: all 2922 copied-source hashes matched, three existing handoff tests passed. The revised lightweight snapshot deliberately removes numerical tensors and checkpoints; these earlier serialization tests are NOT presented as tests of this reduced checkout.

Current checks are source/text selection, original-source hash identity, metadata, and final read-only verifier. Expected malformed historical artifacts are preserved where selected: empty telemetry JSON and window_failed_syntax.py. AppleDouble resource forks are excluded. Originals are never repaired or erased.

Common credential-pattern scan found no private-key headers, GitHub tokens or AWS key IDs in the original text snapshot; final metadata is separately inspected. This is not a general security guarantee.

No training, solver campaign or clean-environment install was run for this handoff. ENVIRONMENT.json and PIP_FREEZE.txt record the actual environment, not a portable lockfile. No Actions workflow found; no CI pass claimed. FWI running worktree and jobs untouched.
