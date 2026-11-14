# Fixed two-source original-equation error-response diagnostic

**Completed:** original K material, original sources 0 and 4, four actual intermediate residuals. All four response solves passed the original equation. No training, unseen-material result or speedup is claimed.

## What was actually solved

The unchanged one-shot P8 + TSA64 path was run at outer tolerance 1e-5. The observer recorded residuals immediately after its first coarse injection and after classical call2, and verified them against q-Ax. For each residual r, ordinary TSA64-preconditioned FGMRES solved A e=r to tolerance 1e-10. A coercivity-based bound independently certifies each response field. No assumption equating residual and error frequency was made.

P8 means tangent spatial Fourier |k_t|<=8, with all 12 angular components. The stored streaming directions, including their small floating-point deviations from ideal angles, were preserved.

## Measured energy fractions

| Source/state | Residual Q8 | Error Q8 | Total error in 9-16 |
|---|---:|---:|---:|
| 0 / after_coarse | 99.8728% | 94.2487% | 93.4335% |
| 0 / after_classical_2 | 98.3725% | 97.0251% | 96.5777% |
| 4 / after_coarse | 99.6489% | 67.2912% | 65.8506% |
| 4 / after_classical_2 | 97.6834% | 98.3291% | 97.2891% |

Maximum original response residual: 8.10118358e-11. Maximum evaluated relative response-error bound: 2.69496692e-08. The response solves required 8,9,8,8 TSA64 calls respectively.

The source4 after-coarse state is the decisive warning: residual energy is 99.65% outside P8 but error energy is only 67.29% outside. At the later state actually targeted by a second correction, however, source0/4 error is 97.03%/98.33% outside P8, and 96.58%/97.29% of total error is in the fixed 9-16 shell. This supports a narrow exterior-support experiment, not an arbitrary cutoff sweep.

## Angular and block-coupling evidence

At the later state, Q8 error in directions with |omega_tangent|<=1e-6 is only 2.42%/2.60%, and in directions with |omega_normal|<=1e-6 only 0.427%/0.135%. Thus the axis-aligned grazing groups do not explain these tails. They are predominantly oblique, and the exact angular weights differ by source. This does not by itself establish full two-dimensional characteristic alignment, which was not measured.

| Source/state | ||P A Q e||/||P r|| | ||Q A P e||/||Q r|| |
|---|---:|---:|
| 0 / after_coarse | 0.687750 | 0.001931 |
| 0 / after_classical_2 | 0.448189 | 0.006484 |
| 4 / after_coarse | 0.699717 | 0.004793 |
| 4 / after_classical_2 | 0.128613 | 0.003583 |

The low-frequency action receives appreciable contributions from high-frequency error; low/high labels are not interchangeable through A. The converse cross contribution to Q residual is below0.65% in these states. All block actions were evaluated using the original high-precision operator, and their sum reproduces r within the measured response residual.

## All paid offline cost

External /usr/bin/time process wall: **59.90 s**, user CPU 56.69 s, system CPU 1.64 s, peak RSS 1023.41 MiB. This includes interpreter startup/shutdown and final writes. All reference responses are offline research cost, never free online corrections.

| Exclusive internal category | Seconds |
|---|---:|
| input_verification_seconds | 0.165339 |
| operator_and_TSA_setup_seconds | 0.160692 |
| factor_setup_seconds | 1.306360 |
| factor_qualification_seconds | 1.493161 |
| incumbent_solves_seconds | 9.613133 |
| incumbent_checks_seconds | 0.158190 |
| response_solve_seconds | 41.501387 |
| response_check_seconds | 0.107677 |
| spectral_coupling_seconds | 0.497945 |
| tensor_save_and_hash_seconds | 0.649766 |
| import_startup_logging_and_other_seconds | 3.495216 |
| Remaining process envelope beyond internal clock | 0.751133 |

The standalone observer bridge also passed (zero final-field difference, matching call counts, exact recorded residual identities); its measured numerical portion was0.0342s and is additional validation work, not included in the main process wall. Its interpreter/startup was not separately instrumented. Code preparation and transfer time are not a solver-performance claim.

## Consequence and limits

The next authorized test is a fixed second-direction support comparison: one-shot P8 incumbent, repeated P8 at the same second-injection point, and the fixed9-16 shell. It must solve the actual projected equation rather than inject these stored error responses. The new shell factors and every qualification/solve/check must be charged. Sources0/4 are discovery sources for this choice;7/14 were not used in choosing it but remain historical K development sources, not new heldout data.

No P12/P16 sweep or training was launched by this diagnostic. Original production and sealed E hashes are unchanged. Reference residual/response tensors are retained on FWI only, with hashes and total size 25179774 bytes. No Library upload was attempted.

Raw result SHA256: `c19730a31009f638cc8bcfb6180ce2de1ca3827d7b43940f6aacd584f032bd1e`.
