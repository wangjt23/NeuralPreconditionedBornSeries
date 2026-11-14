# Independent A/B implementation result audit

2026-10-04. Standard-library scalar reconstruction only: no SSH, jobs, original arrays, native backend, FFT, transport action or full solve was rerun. Final A and B receipts are complete, exit zero and remain within their recorded hard deadlines. This review verifies the scalar/code evidence; it does not independently repeat FWI preservation or read the private NPZ arrays.

**Decision: the stagewise checker and its upper-only specialization pass the bounded engineering implementation gate.** All eight A certificates and four B certificates satisfy the unchanged exact 1e-5 residual, 1% vector-radius reserve, 1e-3 relative solution-error and 1e-4 paired-field criteria. Twelve direct pair controls give six positives and six rigorous lower-bound negatives. No full solver comparison or learned advantage is admitted.

The audit reconstructs every predicate from exact Fractions, directed binary64 endpoints and exact stored coercivity margins. It also reproduces all 432 ledger rows, including 360 FFT-layer radius updates, the outward sqrt(2) multiplier, inverse 1/N scaling, stored-D64 norm propagation, collision error and final addition/subtraction. The final residual bounds equal max(0,C_L-E_U) and C_U+E_U rounded in the required directions. All four shared N256 cases have identical scalar enclosures, gates and entire non-timing stage ledgers in A and B. Native guard/library seals agree; all 17 local frozen execution-source hashes match the prospective policy. FWI preflight reports 47 tests and 34 subtests passed.

| N256 engineering field | E_U/Q_L | R_U/Q_L | Relative solution upper | A call (s) | B call (s) |
|---|---:|---:|---:|---:|---:|
| ID0 baseline | 2.46038953e-11 | 2.47022454e-11 | 1.83678703e-10 | 33.235730 | 29.127211 |
| ID0 positive | 2.46044218e-11 | 9.53698921e-07 | 7.09147097e-06 | 32.803661 | 29.378309 |
| ID3 baseline | 9.90457322e-12 | 9.95762662e-12 | 8.27498286e-11 | 35.733160 | 28.785715 |
| ID3 positive | 9.90564739e-12 | 9.53684222e-07 | 7.92535803e-06 | 34.395328 | 28.596509 |

The largest N256 radius/source upper is 2.46044218e-11, below the fixed 1e-7 radius/source allowance. Saved teacher baselines on N64/N128 also pass. The newly persisted N256 baseline b is one existing M(original-source) application and its checker RHS is represented q=rounded original GPU H(b). Positive a is the represented small gain perturbation of b; neither field is a new solution of the original physical source. Teacher baseline=self and gain pairs are engineering checks, not two independent solver outputs. Six fixture snapshots were recorded as persisted before the separate fresh checker, then read back and hash-checked; candidate/source/operator identities link A and B to those snapshots. The earlier transient N256 NO_CERTIFICATE result remains preserved and is not retroactively reclassified.

Four matched N256 certificate calls total 136.167879 s for A and 115.887744 s for B, a descriptive 14.893480% reduction. A owned-process wall is 166.736132 s and B wall 119.350026 s. A contains eight certificates, twelve pair controls and prior N64/N128 twiddle warming; B contains four N256 certificates with its own cold process. They ran under concurrent bounded batch load, without a dedicated randomized repeated paired timing study. The observed differences therefore do not establish a causal cold-workflow speedup. In particular, the external A/B clocks must not be compared as equal workloads.

Snapshot generation/setup/transfers/persistence, native compilation/preflight, fresh imports/readback/hashes, pair controls and serialization remain billable. Inner action/stage/certificate timers are nested and must not be added twice. The shared snapshot freeze records two M and two H actions; A/B themselves are CPU checker jobs with zero full solves and training. This review makes no overall batch-action-count claim about the separate C/D directions.

The new staged driver includes deadline checks after certificates, before/after paired work and before its complete flag, fixing the paused-draft review concern. Paused files and old failed gates are untouched. Job B settings omit three redundant scope booleans (source-export, paired-solver comparison and global-GPU contract); its per-certificate contract flags, zero-solve/training counts, frozen source and declared persisted engineering inputs establish the narrower implemented work, rather than invented missing metadata.

Enclosure validity remains conditional on the reviewed compiler/IEEE/nextafter/exact-bit contract and original nominal stored operator. Finite preflight tests are supporting evidence, not a universal CPU/GPU arithmetic proof or continuum-accuracy theorem. The upper-only specialization saves unused lower-bound work; no margin cache or accuracy relaxation was introduced. Fully charged checker cost still takes tens of seconds for these N256 inputs and is not a deployable learned-preconditioner benefit.

Reproduce scalar audit with `python3 deliverables/batch_20261004/audit_stagewise_results.py --output NEW_AUDIT.json`. The script requires final complete receipts, refuses to overwrite the output and imports no numerical/project backend.

Current audit JSON SHA256: `7519cd745cf5f4789ec141c096f5d464e7f9bc934e9faf20f8cecab43b342e39`. Authored reducer SHA256: `f5a70b3432b35a9baecfeafecc4b72d639a02adf4ab2f40f3d6937b4950a74be`.
