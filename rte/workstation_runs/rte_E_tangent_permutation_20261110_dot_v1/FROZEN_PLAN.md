# After the N256 gate: direction debate and proposed next test

Status: read-only analysis and planning. No new material array, factor, operator action, solver or GPU job has been run. J2 is sealed; the following is a proposal awaiting review, not permission to start another experiment.

## What is now established

Unchanged E has three paid CPU pairs on the fixed N128 amplitude3 scene, two orders on its predeclared amplitude6 perturbation, and one forward N256 pure-scale gate. J2's actual TSA16 binding is verified, all original acceptance gates passed, and E cost77.2541 s versus compact4's98.6712 s, TSA16's105.2351 s and TSA64's107.7072 s. Its first source31.4116 s was slower than TSA16's25.9100 s. The prior mechanical failure cost375.6764 s remains separate; J2 cost411.4393 s including controller startup/shutdown. There is no incomplete result promoted to a full batch.

This is a CPU-constrained fixed four-source result. GPU compact4's separately measured N128 complete cost0.4613 s is far below the CPU-scale seconds. That is deployment context, not a matched N256 GPU comparison. G and I remain closed. N256 is development scale, not a held-out material claim. The original sixteen-material Q1 protocol, seed9626 and its unlocking requirements remain separate and untouched.

Evidence: J2 archive SHA2567f6e6839129931bb95d70475dc03aa6bd8769fac4b921251a9e1b87d4b4a4315; raw receipt SHA256504000b6746aad51cf1ba8f63420c1872ef55252143c4850b70c4bddb85daaad. All options below use this evidence, not a selected alternative run.

## Debate round one: three non-equivalent questions

### A. Reverse-order N256 confirmation

Question: does the same-scene paid cost advantage survive order/cache/host variability? A fresh E-to-compact reverse pair, plus ordinary native checks, is a clean test of that uncertainty. It changes no material or source, and could falsify the timing conclusion without new algorithm engineering.

Objection: the initial gain is21.7%, its calls match the N128 pattern, and repeated N128 evidence already exists. Another roughly200-300 s N256 batch would mainly refine confidence in this one scene. It would not identify why fixed P8 succeeds, or whether the advantage survives a different transverse material spectrum. A contradictory reverse result must be called uncertain, not followed by repetitions until favorable. Keep this available for confirmation/publication, but do not default to it merely because the preceding run passed.

### B. Change the transverse coefficient spectrum while preserving its distribution and background

Question: is E's observed benefit dependent on the unusually smooth transverse inclusion fitting P8? Use one fixed, prespecified, integer pullback of the already tested N128 coefficient arrays: for normal index i and tangent index j,

  total_new[i,j] = total_old[i,(3*j) mod128],
  scatter_new[i,j] = scatter_old[i,(3*j) mod128].

This is a bijective column permutation because3 is coprime to128. It preserves every normal row's coefficient multiset, extrema, mean and norm in exact arithmetic. The source's actual background selection is not a mean reduction: it uses total.min along each candidate tangent axis, gathers scattering at the selected indices, and chooses an axis by a squared-residual sum. Minima, tie handling in the scattering gather, and reduction-order effects in the score must be checked on the actual stored arrays before claiming identical background. It changes the ordering of the transverse texture, producing a threefold periodic pattern rather than simply increasing inclusion amplitude. Normal layering, stored angular geometry, absorption relation, coefficient contrast and the four source arrays remain unchanged. This is a discrete coefficient rearrangement, not a continuous-PDE equivalence or a solution-coordinate transformation: the derivative operator is NOT multiplied by3 or permuted into a different equation. The new coefficients go into the same original physical A and all checks use that A.

Its Fourier identity is exact for the finite discrete arrays: with inverse3=43 modulo128 and the actual FFT normalization, F_new[k] = F_old[(43*k) mod128]. Thus total coefficient spectral energy is preserved but its placement relative to P8 changes. This is modular aliasing of an explicitly defined discrete material, not an assertion that a continuously resampled Gaussian is band-limited or a17-point cyclic approximation. It is not necessary to invent a new random seed, choose a better material, tune amplitude or broaden the source family.

Objection: any such test is still one designed material perturbation, not generalization to arbitrary media. Less coefficient energy covered by P8 does not imply a harder solve: original streaming grows at higher frequencies and may suppress the shifted components more effectively. Modular aliasing also moves some frequencies downward. Either a gain or a loss is possible; no deterioration is presumed. The source alignment changes as well as mode coupling, so outer convergence cannot be attributed solely to one Fourier-energy number. Qualified exact coefficient/background identities and unchanged factor inputs provide a stronger causal constraint than an amplitude sweep, but cannot remove all source/operator interactions. A loss or inner-budget failure would be informative and must be retained without P8 enlargement.

### C. Choose an execution path using the already paid source-horizon information

Question: when does E's setup pay, and should low-latency first-source requests use native TSA16 instead? This can first be answered by read-only arithmetic; no extra solver is needed. PAID_PREFIX_LEDGER.json uses directly measured first/four-source clocks and bounds intermediate prefix wall by setup plus measured solve/check phases, with all nonnegative remaining batch overhead placed either before or after the prefix. It does not call these derived bounds direct prefix measurements.

For the original source order[0,4,7,14], one source favors TSA16,25.9100 s versus E31.4116 s. For the first two already evaluated sources, E's derived interval47.3078-47.4482 s lies below TSA16's51.9544-52.5466 s, TSA64's54.1369-54.3135 s and compact4's59.8575-60.5126 s. At three sources E's interval62.8801-63.0206 s is also below all controls; four-source totals are directly measured. These conclusions use subsets of the existing four-source receipt, not an increased source count or extrapolated average source.

Objection: this is post hoc economics for one known instance/order, not a validated adaptive selector. Trying several methods online and selecting a winner would require charging their probes and discarded factors. The source horizon may not be known, and source-to-source difficulty can change elsewhere. A hybrid first-source-native/later-E implementation is not needed to establish the current cost tradeoff and is not yet a measured capability. Stop the economic task at this honest ledger rather than creating a controller without a demonstrated need.

## Debate round two: strongest nulls and choice

A is the strongest immediate timing confirmation, but its information is local to an already observed21.7% gap. C is useful and cheapest; its main unresolved accounting question is already answered by the saved phase bounds without new numerical work. Neither explains whether E's17-mode coverage is fragile to a different transverse spectrum.

Select B as the next proposed numerical question, at N128 rather than combining a material change with N256 scaling. N128 has the strongest repeated E attribution baseline and much lower setup/qualification cost. The experiment is one new discrete material, not a further claim about scale or the old sixteen-material test. Use the fixed multiplier3 once. No search over permutations, widths, frequencies, source orders or amplitudes is permitted. If it does not stress the anticipated spectral quantity, report that null and stop rather than trying another multiplier.

The strong representation null is the original material under identity permutation, which must remain byte-identical, plus exact recovery of the new material by inverse permutation43. Source hashes must remain unchanged throughout; the actual test never permutes the sources. This costs no new solver. Independent coefficient FFTs must satisfy the modular-index identity in complex128 to relative1e-12. This does not test or claim conjugation of the entire operator, which would require transforming streaming and sources too and would be a different experiment.

Background policy is fixed in advance: run the original background_from_material on the new material every time, never force or reuse the sealed A0 background. Record old/new axis, total/scattering profile hashes, maximum absolute differences, unequal-element counts and the candidate-axis scores. The original code is min plus gather rather than mean, but this does not justify skipping a stored-array check. If either background profile differs even by1 ULP, or axis0 is not selected, stop the same-background mechanism qualification and report the actual difference. Do not hide it, round it away, relabel the result as exact, or substitute the old profile. Summation-order score differences alone are recorded and need not fail if the selected axis and stored profiles are identical. A changed background could motivate a separately reviewed question later; it is not repaired in this gate.

The mechanism diagnostic is the fraction of tangent coefficient-deviation energy outside modes-8..8, computed for actual stored total/scattering deviations from their actual background. It is a material-spectrum diagnostic, not a convergence theorem or a surrogate acceptance criterion. Fix the candidate before computing it. If it does not increase, preserve the result and stop this proposed stress before an expensive solver batch; do not screen another material. If it does increase, run the full cost gate regardless of whether E later wins.

The attribution baseline is immutable E on the original scene, including its three-pair call pattern, not an old wall time used as a decision control. The decision controls are three freshly paid same-material CPU methods: TSA64-zero, compact4 and TSA16-zero. The ordinary TSA64 outputs supply only field checks. No accurate teacher, dense oracle or retained reference field is used as initialization or an enrichment vector. A positive outcome means only that E beats these locked applicable controls on this single prespecified spectrum perturbation. It does not establish the globally fastest classical method.

## Proposed executable gate, only after review

1. New independent directory. Verify E angular_modes.py SHA256d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481, the source/production hashes from J2, original N128 total/scatter fingerprints, and source IDs[0,4,7,14] with their existing N128 hashes. Seed9626, original sealed directories, I and G remain untouched.

2. CPU0 admission, one thread, CUDA disabled. Generate only the unchanged N128 material and the single specified column permutation. Record hashes and validate the identity/inverse/background/spectrum conditions above. Source arrays are the same existing bank tensors promoted exactly as before; they are not permuted with the material. No operator/solution coordinate remapping is allowed.

3. Qualify all17 original G/A0 modes, strict-zero and mixed ordinary/tiny columns through unchanged E. The actual background is expected identical, but the validation/factor construction is paid anew and never reused free in a timed method. Use the existing deterministic qualification seed and signed order. Original G/A0 target1e-10, at most3 repairs; exact-zero rule and scale-safe tiny-column checks unchanged.

4. One fixed forward fresh sequence: TSA64-zero, compact4, actual TSA16-zero, E. Use the repaired requested/effective/object depth assertions64/64/16/64 from J2 and its independent AST/stub binding test. No reverse automatically. Each method pays fresh physical/operator/cache/factor setup. E pays all17 qualification again in its own setup, including original-A0 actions and zero/tiny cases. Record first-source and complete four-source observed wall, not means or amortized first-source estimates.

5. Original12 angles,g=.65, stored directions/weights/phase, original normal Nyquist, P8 signed modes-8..8, inner1e-4/max8, outer1e-5/max100, zero initial state and one enrichment after prefix1 remain unchanged. Final original-A residual1e-5, evaluated relative bound1e-3 and field difference1e-4 against ordinary accepted references. Record original and inner calls and every G correction. Increased iterations or failure at these fixed budgets are results, not implementation defects to tune away.

6. After each accepted control, retain its complete observed cost. Stop E if its paid setup lower bound reaches the cheapest complete control. Otherwise finish the fixed four sources. Preliminary positive requires all identities, qualifications, acceptance and resource limits AND E's complete observed time below every accepted control. Any real numerical, identity, resource or budget failure, or final cost loss, ends the gate. Preserve all partial costs. No extra RHS, method tuning, repeated trials or easier material replacement.

Proposed budget:240 s aggregate numerical controller wall,90 s per stage,4 GiB owned-worker RSS, CPU0 only and no CUDA. N128 E/compact logical storage is about277/302 MB, and prior full-process evidence was below this ceiling. This is a budget estimate, not a guaranteed fit or runtime. Check admission before every executable stage; record concurrent jobs and shared-host limitations, without reserving or blocking all other research. The watchdog may stop only this worker. If a purely mechanical interface error appears, preserve its source/cost and seek review before any fresh numerical attempt; no automatic retry.

## Reporting and alternatives retained

Report original-A acceptance and actual first/four-source costs even if E fails. If background is exactly unchanged but E's inner/outer calls rise, that isolates a limitation beyond background factor quality, though not a proof of one exclusive cause. If calls and costs remain favorable, it establishes only one spectrum-transfer success. If the spectral premise itself fails, that is the cheapest decisive negative result and closes this particular proposal without another material.

A remains a sensible later confidence check if a deliverable needs repeated N256 timing; C's ledger is already useful for CPU planning. Neither requires reopening G/I, comparing N128 GPU time with N256 CPU time as a speedup, or changing the original four-source contract. Await review before implementing or running B.
