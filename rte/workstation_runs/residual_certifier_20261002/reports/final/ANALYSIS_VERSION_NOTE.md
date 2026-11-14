# Scalar analysis version note

The final main report, executed seven-file certifier code, frozen policy, raw receipts and acceptance predicates are unchanged. The initial analysis snapshot was staged privately on FWI before this local reporting correction; it is preserved there rather than overwritten.

| Authored artifact | Initially staged SHA256 | Final bundled SHA256 |
|---|---|---|
| analyze_certifier.py | 226ae23b70c14835eebc3f8bf75055d0a892a67a5d0dec7bdd0d7e8b138d3f00 | e3edf9631014baf51c1477b1470ce22798a76de25f7402695a4b0f25f9553ceb |
| SCALAR_ANALYSIS.json | 8ac28f2ef086a86077aa2e05a8451e93806790e1d329cdd7f7ec97daa276c9bc | 1aead6cef0a185de2353cbc0b82d24d886a1b116667c19d8574a173596d1e42d |

The event display `exact_relative_solution_upper` became `approximate_relative_solution_ratio`: nearest-rounded display conversion of the exact Fraction ratio is not itself an outward bound. Its numerical values are unchanged; raw certificate bounds remain directed and all decisions still use exact Fraction cross-products. Local filesystem provenance labels became stable policy-snapshot/hash labels so the analysis reproduces byte for byte after extraction. The final bundled analyzer records this display scope explicitly. No thresholds, field inputs, kernels, costs, experiments or acceptance decisions changed.

Final scalar decisions remain four saved-teacher acceptances, two manufactured N256 radius-reserve failures and four zero-control rejections; all 40 individual scalar predicates agree with receipts. The bundle contains the current final analyzer/analysis and this prior-snapshot metadata, not a duplicate older snapshot. Parent-side version-staging receipts record preservation of the initial artifacts and separate final staging. This note does not independently repeat those FWI operations.
