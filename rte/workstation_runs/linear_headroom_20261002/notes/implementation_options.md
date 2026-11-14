# Implementation headroom after the failed precision repair

The best-supported economical implementation test is replay of the **exact
original prepared FP32 TSA action**, leaving every physical defect, stored
coefficient, reference inverse, rounding order and outer FP64 acceptance check
in place. A small refresh of omitted shallow classical controls should accompany
it before any learned replacement. Neither route has been measured in this
delegated audit; no remote command, training, original-source edit or reserved
seed9626 access occurred here.

The two94-FFT prototypes remain negative results. Conservative algebraic TSA
and the selected retained-FP64-sum variant each failed five of48 unchanged
physical-image parity cases. The latter's worst error remains5.18003e-4 against
1e-4. They must not be used as accuracy-qualified timings or replaced silently
by a CUDA graph. Replay below captures the original126-FFT implementation.

## What the existing costs actually show

The uninstrumented sixteen-media development comparison reports frozen
portfolio warm means139.315/135.830/143.695 ms at N64/N128/N256. Winners are
reference-tail at N64/N128 and TSA16-zero at N256; warm repeats were reduced to
one median per medium. Plain TSA16 means144.564/137.723/144.751 ms in that run,
with4.5625/4.5/4.5 mean classical calls. Learned mixed-tail insertion did not
save mean classical calls at N128/N256. These are existing run results, not
predictions for replay or a new cohort.

Supplementary synchronized profiles use two eligible cases per prefix:
TSA8 IDs0/1 and TSA16 IDs0/3. The following values are means per profiled solve:

| N | TSA8-reference B/call ms | TSA16-plain B/call ms | TSA16 B total ms | TSA16 outer action + checks ms | TSA16 projection/LS ms | TSA16 profiled wall ms |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 14.130 | 27.365 | 177.871 | 13.053 | 7.610 | 200.885 |
| 128 | 13.862 | 27.630 | 165.777 | 11.909 | 6.790 | 186.421 |
| 256 | 14.092 | 28.141 | 168.847 | 20.101 | 9.642 | 200.846 |

TSA16's mean eligible classical calls are6.5/6/6 here, rather than the full
cohort's approximately4.5. Homogeneous reference callback sub-calls take about
0.27–0.36 ms. Profile synchronization, first-call outliers and different case
selection prevent using these stage fractions as an attainable speed ceiling.
They do support prioritizing B and testing whether its nearly grid-independent
per-call cost contains substantial dispatch overhead. No kernel-level trace
yet divides B into FFT, angular matrix work, casting, layouts or host launches.

Primary receipts:
`receipts/rte/workstation_runs/direction_diagnostic_20261002/results/runtime_summary.csv`
and `eligible_profiles/profiles_n{64,128,256}.json`. Profile category definitions
are in the isolated `deliverables/followup/instrumented_fgmres.py`; callback
reference/model sub-times overlap the callback category and must not be summed
twice.

## Exact source-derived work and cache facts

`remote_reference/rte/transport_baselines.py:52–70` gives a TSA-depthd action
`d G + d R + (2d-1) A`, hence8d-2 spatial transforms. TSA16 has16 transport
divisions,16 reference matrix applications,31 physical angular-scattering
actions and126 spatial transforms. Its sixteen inner and fifteen outer physical
defects are part of the original numerical control.

The reference-only depthd action instead uses `d R + (d-1) A`, hence4d-2
transforms. Exact operation counts provide useful coverage checks:

| Action | Spatial transform calls | Physical A calls | Reference R calls |
|---|---:|---:|---:|
| R1 | 2 | 0 | 1 |
| R2 | 6 | 1 | 2 |
| R4 | 14 | 3 | 4 |
| TSA2 | 14 | 3 | 2 |
| TSA4 | 30 | 7 | 4 |
| TSA8 | 62 | 15 | 8 |
| TSA16 | 126 | 31 | 16 |

Each R includes a spatial FFT pair and a stored full-angle matrix action;
these counts do not equate the reference and TSA actions' wall costs.

The homogeneous inverse is already eagerly cached by `prepare()` and lazily
retained by `PeriodicRTE2D.reference_transport_inverse`
(`transport_baselines.py:33–50`, `models.py:244–257`). Warm production B receives
complex64 through `benchmark_tail_solver.py:91`; consequently
`reference_transport_inverse.to(rhs.dtype)` normally performs no conversion.
Claims that warm FP32 TSA repeatedly refactorizes or promotes its inverse are
incorrect. A native complex128 input **does** repeatedly cast the FP32 inverse
from72 to144 MiB at N256/L12; that is relevant only to a separately selected
wider-reference action or diagnostic, not the current production B.

There is other genuinely repeated immutable work:

- `_stream_symbol` (`models.py:189–195`) reassembles the stored rounded stream
  on every G and A call,47 times inside TSA16. Caching the actual assembled
  tensor can remove construction work while retaining every126 transforms.
- G (`models.py:215–221`) also assembles `stream+sigma0`16 times. A cache must
  retain **division by that exact denominator**; replacing division by a stored
  reciprocal multiplication changes rounding.
- Scattering/A (`models.py:182–208`) repeatedly converts stored coefficients
  and phase to input dtype. Cache their exact casts if that separate ablation
  is selected; preserve weight-times-field, phase contraction, sigma_s multiply
  order. Precombining phase/weights changes rounded arithmetic.
- Reference RHS permutation/reshape and matrix multiplication recur16 times.
  The inverse layout is already cached; a reshape can still require materialized
  RHS/layout storage. Actual profiling is needed before changing contractions.

An immutable-tensor cache should be a separate code revision/ablation from
replay. It keys setup generation, input dtype/device/shape/layout, all material
and geometry tensor pointers/versions, phase/quadrature and hyperparameters,
retains input references and invalidates on changes. Capturing the untouched
body first gives a cleaner dispatch-only hypothesis.

## First implementation test: exact prepared TSA replay

The source body is plausibly capturable once `base.prepare()` has created its
inverse. It has a fixed depth, static tensor shapes and no tensor `.item()`,
`bool()` or `float()` calls within its prepared arithmetic path. Python metadata
checks and cache decisions can run before capture/replay. Lazy inverse building,
new setup, logging, profiling synchronization and validation remain outside.

PyTorch2.5.1 documents static shapes/control flow, fixed memory addresses and
no CPU/GPU synchronization during capture. Inputs/outputs need retained storage;
warmup occurs on a side stream. Replay removes repeated CPU dispatch, while
CPU checks inside capture do not execute on later replays. These constraints
support capturing B alone, not the data-dependent outer solver. They do not
guarantee this FFT/GEMM workload's capture succeeds or that replay is faster.
[PyTorch2.5.1 CUDA graph documentation](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst#cuda-graphs).

Minimal isolated wrapper contract:

```python
# Prepare once for the current immutable material, settings and complex64 shape.
base.prepare()
static_input = exemplar64.clone()
# Warm the original base on a correctly synchronized side stream; charge it.
# Capture ONLY the original base(static_input), using the existing kernels.
with torch.cuda.graph(graph):
    static_output64 = base(static_input)

def apply64(rhs64):
    check_external_signature_and_shape_dtype_device(rhs64)
    static_input.copy_(rhs64)
    graph.replay()
    return static_output64.to(torch.complex128)  # fresh owned deployment result
```

This is an implementation sketch, not executable prepared code. The static
input is initialized with an existing source/residual so warmup never consumes
uninitialized values. The cast to complex128 occurs after replay and allocates
an independently owned result, just like the current deployment wrapper.
Returning the reusable graph output would corrupt previously retained Krylov
directions after another replay. Capturing a complex128 output and cloning it
afterward is also ownership-safe but adds a copy beyond the original output
cast; charge that if chosen. A native-complex128 direct call should bypass this
complex64 graph and call the original base at complex128, preserving the old
direct-call semantics and48-case diagnostic fallback.

External guards are mandatory on **every** replay: Python `prepare()` inside
the captured body cannot protect later calls. Include material/geometry/reference
buffer identities and versions, source setup generation, shape/dtype/device,
depth, shift and relaxation; retain tensors to avoid allocator address reuse.
Changed signatures must reject or deliberately rebuild a new graph outside the
application timing, never replay stale coefficients. Keep no concurrent capture
in this process, use correct stream ordering, preserve CUDA arithmetic/backend
settings, and count each wrapper's capture/warmup/private-pool cost. These are
specific replay-contract requirements, not additional solver acceptance rules.

Start with the already saved N256 ID0 material and source plus its actual saved
Arnoldi frontier/residual if available. Compare original eager B and copied-
input graph replay using both CPU wall time and CUDA events, charging input
copy/output ownership and reporting graph preparation/peak memory. Event elapsed
time across a queued sequence can still include launch-starvation gaps; it is
not a pure sum of arithmetic kernels. Do not synchronize every TSA stage in the
performance test.

If capture works, preserve all48 original case/precision/probe parity checks
with1e-4/1e-9 thresholds, native128 fallback and original physical image. Add
zero response and an ownership check that a first returned result remains
unchanged after replay on a distinct second input. Verify the captured original
body has126 transforms per depth16 call; replay does not call Python FFT entry
points again, so mock counts apply to the captured body, not each replay.

Only a passing application gate warrants separately timed original-physical
full solves on all sixteen development media, retaining current acceptance.
Contemporaneous competitors are the unchanged frozen per-grid portfolio and
the same portfolio accelerated by the same available replay technique. Graph
speed would be a classical implementation improvement, not a learned victory.
Measure setup-plus-one-solve as well as warm solves. With capture extra cost C
and per-source warm saving deltaT, a repeated-source claim requires actual reuse
Q satisfying Q*deltaT>C; do not assume that one source amortizes capture.

## Classical coverage refresh and fixed-material polynomial headroom

The current tail portfolio contains TSA4/8/16/32, reference16/32, HSS16/32,
TSA8/16-zero, reference-tail and moment-reference
(`benchmark_tail_solver.py:28–30`). R1/R2/R4 and TSA2 are omitted. A bounded
refresh of R1/R2/R4 zero/balanced and TSA2/4 alongside frozen competitors is
cheap, uses existing operators, and may reveal that a large fixed stationary
depth is unnecessary once outer Krylov optimization pays for more cheap calls.
Actual complete solves determine that tradeoff: shallow B can increase A/LS
work enough to lose despite fewer transforms per B call.

These are coverage gaps in the current comparison, not untried mechanisms.
`audit_transport_baselines.py:102–106` already includes reference/TSA1/2/4;
September13 tuned reference/TSA2/4/8. The stiff September26 source study includes
TSA2/4/8/16/32 and reference4/8/16/32 with zero/balanced starts
(`train_source_predictor.py:287`, source-initialization review:133–141).
Its default seed2626 implies validation4626, matching the current family's seed
if defaults were used; recover actual historical settings/CSV/input identities
before claiming it reused the same16 media. R1/R2 and current N256/solver coverage
remain clearer gaps than repeating historical R4 tuning.

Use existing IDs0/3 on the nested N64/N128/N256 material/source arrays for a
modest screen; they are the previously eligible TSA16 cases, not selected by
timing. This gives six resolution-case rows but only two independent media.
Record failures and stopping on all attempted cases. These two cases cannot
establish a winner; a surviving predefined shortlist must face all16 current
development media, including easy/already-converged cases, before a new gate or
portfolio selection. No fresh teachers or source generator calls are necessary.

An inner-GMRES4/TSA1 or four-step Anderson-like action is a different stationary-
depth alternative, conditional on the shallow refresh and charged cost. Four
original TSA1 applications plus four original-A Arnoldi actions use32 spatial
transforms,34 with an independent terminal A check, versus TSA4's30 and TSA16's126.
If inner A is FP64, report its8 or10 transforms separately from24 FP32 TSA1
transforms. Two-pass four-step MGS has20 dot/axpy projections and roughly nine
full FP64 fields of V/Z storage. Dense LS, host synchronization and final checks
are paid within the preconditioner; transform savings alone are not enough.

The inner LS coefficients depend on the input RHS, so this is nonlinear
preconditioning and should use the already flexible outer solver. It is not a
complex-linear fixed-material learned map. There is no deployed Anderson or
inner-GMRES/TSA1 accelerator found in the inspected snapshot, but an old
per-sample complex-scalar residual polynomial already exists
(`train_learned_hard_cases.py:19–69`), and short FGMRES already supplies the
training objective (`train_krylov_transport.py:25–58`). If paid inner optimization
exhibits a decisive cost/accuracy benefit, it can motivate material-fixed scalar
coefficients inferred once from material; such coefficients preserve exact-
arithmetic complex linearity and avoid residual-conditioned routing. It needs
the optimal classical inner-GMRES control and a matched fixed analytic polynomial
before any training. Generic cached linearity or new scalar coefficients alone
would repeat old work.

## Secondary headroom and invalid shortcuts

Outer FP64 physical A repeatedly reconstructs immutable symbols/casts too,
but outer actions plus checks are a smaller recorded category than B. It must
retain the original stored-factor FP64 physical definition; the promoted
assembled FP32 stream used in the precision ledger is a different object.
Its safe immutable-cache ablation can preserve operation order, whereas
substituting low A or deleting true checks changes the acceptance protocol.

The traced and augmented real-stacked FGMRES implementations extract norms to
Python, run sequential MGS reductions/subtractions, rebuild direction stacks
and solve a growing small LS each iteration. Fixed-size preallocation and
incremental Givens rotations may reduce overhead, but constitute a separate
numerical solver revision; simultaneous projections alter orthogonalization
rounding. The current algorithms check physical candidate residuals, stopping,
restart state and the final original RHS. Removing most of those checks cannot
be called a like-for-like speed comparison. Capturing this dynamic outer solver
as if it were the static B is inappropriate.

Setup is already reused for warm repeated solves. Sharing a reference inverse
across same-material classical depths saves portfolio-screen construction cost,
not the deployed single baseline's warm time. For genuinely different RHS at
one material, using RHS as matrix columns can avoid repeated expanded inverse
storage; source-independent inverse sharing is legitimate and must be supplied
to all controls. It cannot batch sequential Arnoldi directions that depend on
previous actions without changing the Krylov algorithm.

Simply increasing the tensor batch in the current solver is not independent
batching: its norm/inner products flatten the entire tensor, giving one direct-
sum Krylov problem and global stopping, which can mask an individual failure.
Independent application batching requires per-case norm, H/LS coefficients,
active masks and final original-RHS acceptance, with declared latency/throughput
and actual material reuse. That is a larger unselected implementation task.

## Data identity and next coordination boundary

Reused development arrays live remotely under
`rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{N}.pt`.
Verified parent hashes from existing receipts are:

| N | SHA256 |
|---:|---|
| 64 | bbebfd1c7c4808733628ee8fdecac9e9ce09521d70f7664a5d5391c8af10813a |
| 128 | 9ade993bce700505d134fa197ceeee72327790bc6660e670a95bea8c43052de1 |
| 256 | fb346212e23752d289a06e7f9086fee94a0ca2fa2b16dc6b840ddd14adc64fa8 |

Material construction remains L12/g=.65/epsilon=.1, with original source,
stored geometry and unchanged FP64 physical image. Saved prefix eligibility
and frontier provenance are in the direction-diagnostic receipts; recover exact
serialized frontier fields if needed, rather than manufacture a new random
corpus. No test seed9626, other held-out test receipt or teacher bank is used by
this audit. The parent owns remote staging and the next bounded experiment.

Recommended order: omitted shallow classical refresh plus one original-action
replay feasibility measurement; then full-cohort paired cost only after parity.
An inner optimized-polynomial control comes next if it addresses remaining
expensive stationary depth. A cheaper fixed-material learned action needs an
actual representation/cost oracle before training. Original code and all
negative results remain preserved throughout.
