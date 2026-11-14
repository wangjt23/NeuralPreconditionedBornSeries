# Acceptance-aware continuation verification — 2026-10-02

Nine CPU tests passed locally with Python 3.9 and Torch 2.8.0; execution took
0.068 seconds for the tests. The current read-only RTE source snapshot supplied
the original masked loss and deployed augmented solver. The compact snapshot's
missing Helmholtz GMRES import dependency was supplied from the existing local
project; no dependency was installed or source changed. FWI execution/precision
parity remains the consuming executor's responsibility.

The meaningful counterexample uses `A=diag(1,2,3,4)`, old image basis `Q=e1`,
residual `r=(0,1,1,0)`, and retained frontier `(2,-1,-1,0)/sqrt(6)`. A rejected
zero proposal yields original masked squared ratios `[1,1,1]`. The corrected
helper yields `[1,1/26,approximately 0]`. A separate test reconstructs this
prefix with the actual deployed solver, rejects the zero injection, and checks
the normalized deployed trace against the helper.

Other passing checks cover finite/near-dependent image rejection; independent
real least-squares residuals for each enlarged space; mixed-batch zero, teacher,
dependent and nonfinite proposals; positive/negative real proposal scaling;
joint phase rotation of residual, proposal, image basis and original frontier;
monotone residuals; and a teacher image inside the old Arnoldi span but outside
the old image span. That teacher is accepted as a happy breakdown, with no
subsequent logical classical calls.

From the full FWI repository root, the portable verification command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /data1/software/miniconda3/bin/python PATH_TO/test_continuation_acceptance.py
```

The helper is imported from its directory as:

```python
from continuation_acceptance import corrected_continuation
result = corrected_continuation(
    action, precondition, proposal, residual, image_basis, original_frontier,
    steps=2,
)
ratios = result["ratios"]  # [batch, injection-attempt + two classical slots]
```

It is a no-gradient offline diagnostic. It uses two projection passes and
relative numerical rank thresholds, so the theorem concerns exact arithmetic;
finite-precision deployed trace parity is an empirical comparison. Logical
classical-call counts exclude inactive batch rows, which can still participate
as zeros in vectorized operations. These counts are not measured runtime costs.
The measured previous cohort had zero rejected proposals; this correctness
repair therefore does not explain its observed runtime gap.
