# Proposal to test E on the existing inclusion stress case

Recommendation: worthwhile as one bounded CPU stress qualification, using the already confirmed sequential angular implementation E. Start with one fresh compact4-to-E pair. Only if it passes all identity, original-operator and cost gates should one reverse E-to-compact4 pair run. No experiment, new implementation or numerical work is started by this proposal. GPU no-go G remains closed.

## Read-only evidence audit

The source and receipts inspected are the sealed B `stress_inclusion_cpu.py`, `stress_cpu_01/settings.json`, `stress_cpu_01/batches.json` and `analysis.json`, plus the established E source and confirmation. B changes only inclusion amplitude3 to6. Its constructor preserves N128, periods1, beta10/3, center(0.34,0.60), width0.08 and transition width2/128. It explicitly verifies that its amplitude3 arrays are byte-equal to the original constructor.

B material identity in float32:

- Amplitude3 total:9d9256eedd9242ad1d13f3853a9149b5d5e29a4fbfe6c63caf5a0e012f714b74.
- Amplitude3 scattering:5539f482895689656b2ee0dc05722c6428d3f08b8a3eb3c75f7050f2e8eaf356.
- Amplitude6 total:ff8f1415243874d4d483a18439a23b8c2b193e5c59665ff88e635e645eeeb85e.
- Amplitude6 scattering:12f896ee1a7bbad64d762abd599c62424ad54e97ac5b84edcb1fa71e4e369b0a.

The B constructor source SHA256 is62fed575d02f34676b916dfe1678c561b0970ad97d364e7424bb3bd75284eb74. The old dense inverse source SHA256 isb560e467e0b81a8293adfa2fc71bd5570ec875a797fea67696083f1430ba3a77. Use B's actual constructor and byte hashes, not a new approximate formula or a GPU-material reconstruction.

On that one previously declared stress instance, B recorded compact4 outer calls[4,4,4,4], and dense P8 outer calls[2,3,2,3] with inner iterations[6,6,6,6]. Inner residuals were[8.919350e-5,4.043777e-5,9.829863e-5,4.006225e-5]. The third source is close to the unchanged1e-4 limit. B's paid costs23.323353 and20.550533 seconds are historical context only, not current timing controls.

E's actual sequential angular source SHA256 isd152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481. E has the three paired CPU confirmations on amplitude3. F has only a single unpaired full CPU bridge; its smaller observed time is not grounds to select it as a new CPU winner. The new candidate must therefore be E unchanged, not F and not the stopped GPU implementation.

## Debate round one on whether this adds useful evidence

The positive case is that E's complete cost has been confirmed only at amplitude3. Stronger inclusion coupling increases the required projected solve from19 total inner applications in the amplitude3 results to24 in old B, and changes the outer call pattern. E's original-G repairs and more expensive application path therefore receive a stronger workload, even if background construction stays favorable. Measuring that tradeoff addresses a real limitation of the current CPU claim.

The strongest objection is that old B already established the usefulness of the coupled P8 solve with dense factors. This is not a discovery of a second independent material family, nor is it a fresh held-out test. Its scientific contribution is narrower: determine whether the economical E representation retains the existing stress result after all construction, validation and repair costs. That narrow question is worth a small gate, not a broad sweep.

Three alternatives differ materially. A fixed amplitude6 comparison tests coupling robustness without choosing a new favorable example. A new random-material campaign would address broader scope but is more expensive and would require a new protocol and independent sample selection. More GPU execution tuning targets a different bottleneck and is not justified by this CPU question. Select the fixed B case and keep the other directions closed for this round.

The layered background is selected from tangent minima. At the far side of the periodic Gaussian, the inclusion may already round away in float32, so the amplitude3 and6 background arrays may be identical even though the full operator changes. This is a source-level expectation, not a newly measured fact. Record the selected axis and background hashes in the eventual run. Never assume equality, reuse a previous factor for free, or skip fresh construction/validation on that basis. A possible unchanged background is itself useful attribution: it separates stronger local coupling from a changed background inverse.

## Debate round two on the minimal decisive comparison

One forward pair is the cheapest complete cost screen but is vulnerable to order effects. Unconditionally running multiple paired repetitions spends resources even when the representation fails. Choose a predeclared conditional AB/BA gate: compact4 then E; stop on any failed gate or no complete-cost advantage. Only if that pair passes, run E then compact4 once with entirely fresh construction. Two agreeing orders support a bounded stress result; disagreement is unresolved and ends this round. Do not repeat until favorable.

The decision control is current paid compact4 on the exact amplitude6 arrays. The attribution bridge is unchanged E source, the B material/source identities, all17 independent original-A0 residual checks, and the old B call pattern. A fresh full dense timing batch is unnecessary for the first question. If the expected call pattern changes, report it as a failed attribution bridge and stop rather than silently changing settings or launching a dense rescue comparison.

All source IDs and their original byte hashes remain[0,4,7,14]. Keep N128,L12,g0.65, zero initial states, TSA64 prefix1, P8 modes-8..8, inner tolerance1e-4/max8, original outer tolerance1e-5/max100, evaluated relative-error bound1e-3 and field-difference gate1e-4. E's per-column original-G target remains1e-10/max3 repairs, including its strict-zero handling. No higher-accuracy teachers, reserved samples or extra source reuse are introduced.

## Proposed execution and accounting if authorized

1. In a new directory, verify B/E source hashes, all four material-array hashes, original source hashes and production/dependency hashes before any timing interpretation. Record normal-axis selection and actual background hashes. Do not modify B or E.
2. Compute four ordinary native accepted reference outputs for field checks, separately accounting for their research cost. They must not become candidate initial states or directions.
3. Run fresh compact4 and then fresh E on the same four sources in order. E pays construction of all17 modes, the same independent original-A0 mode checks and all repair work. Maintain source/mode trace receipts, not only wall times.
4. Verify compact4 calls[4,4,4,4], and E outer calls[2,3,2,3] with inner iterations[6,6,6,6], alongside actual inner residuals. The close third-source residual is retained without changing the threshold. An unexpected pattern triggers review and stops expansion.
5. If the forward pair fully qualifies and E's complete observed four-source wall is strictly lower, run exactly one reverse fresh E/compact4 pair. Report first-RHS and complete-four-RHS observed walls separately, with first RHS measured directly rather than divided or amortized from the total.
6. Preserve physical/TSA setup, factor assembly/chain/corner costs, all17 validation, online solves/repairs/checks, call counts, logical factor storage and process peak RSS. The complete observed wall covers any component-timer gaps. Record the current shared-host state; do not compare these CPU seconds with GPU C or G.

Resource proposal: CPU0, one thread, no CUDA. Coordinate whole-batch timing with Schur and other memory-heavy work. Use180-second alarms for each forward/reverse stage and a600-second aggregate numerical ceiling. Stop if admission changes or a stage reaches its limit. No software installation, paid resource, deletion or permission change is involved.

## Closing conditions and allowed conclusion

Stop on an identity mismatch, an original-G/A0 repair failure, a changed call-pattern bridge, failed original-A acceptance, inner failure within8 iterations, exhausted wall budget, or no fully paid E advantage in the forward pair. If the reverse order reverses the cost conclusion, report uncertainty and stop. Keep a slower first-RHS result visible even if the four-source total wins; do not add sources to hide it.

A positive outcome may be described only as preservation of E's CPU benefit on the already specified amplitude3-to6 inclusion-strength perturbation. It cannot be called broad material generalization or an independent held-out result. A negative outcome narrows the cost claim to the previously confirmed amplitude3 case; it does not erase E's valid original-operator representation. This proposal provides no permission to resume GPU work.
