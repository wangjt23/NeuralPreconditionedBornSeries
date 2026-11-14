# Exact-original CUDA graph v2: full development verification

2026-10-02. This is a stronger **traditional implementation comparator**, not
a learned-preconditioner result. The failed94-transform algebraic TSA and its
finite-precision repair remain preserved negative results. V2 captures the
unchanged original TSA arithmetic and retains the original126 spatial
transforms at depth16.

## Access, scope and preservation

Actual authorized host/user: `FWI-Standard-PC-i440FX-PIIX-1996` / `wjt`, reached
through the existing `FWI` SSH alias with `ClearAllForwardings=yes`,
`BatchMode=yes`, `StrictHostKeyChecking=yes`, `ConnectTimeout=10` and one
connection attempt. No trust, forwarding, credential or SSH configuration
changes were made. The initial sandboxed SSH attempt returned `Operation not
permitted`; the authorized escalated connection succeeded. An earlier local
shell read returned `exec-server transport disconnected` and then recovered;
neither error was treated as an FWI Internet outage.

Repository: `/data/wjt/NeuralPreconditionedBornSeries`, branch `main`, HEAD
`028caea3f35d100225f656e2b1e910c1366aaafe`. New remote files/results exist only
under `rte/workstation_runs/linear_headroom_20261002/v2`. All155 original-source
hashes, frozen v1 files, three existing development parents and frozen v2 code
hashes were verified before/after. Original and uncommitted project work was
preserved. No training, reserved seed9626, held-out data, package installation,
job cancellation, Git push/merge or public sharing occurred.

## Implementation and policy

`graph_classical_v2.py` retains strong base/operator/tensor/static-buffer
references and checks object identities, versions, dtype/device, shapes,
strides, storage offsets, grid/setup generations and existing scalar settings.
The guard rejects a missing reference inverse without invoking its rebuilding
property. Static input/output version changes from normal application are
permitted, while their objects/layouts remain fixed. Actual transpose-view
replacement and subsequent restoration are tested. Graph outputs use a fresh
complex128 allocation, preserving retained FGMRES Z vectors across later
replays. Native complex128 inputs use the original eager action. The wrapper
is serial; concurrent applications sharing the static buffers are outside its
contract. The verified recipe uses TSA, not arbitrary untested classical kinds.

CPU tests:13 local tests passed in0.146s; all13 passed again on the original FWI
Python3.12.4/Torch2.5.1 environment in0.554s (3.552s subprocess wall). The remote
test child had CUDA hidden and ran only CPU fixtures.

Frozen policy:
`f08fb914327d23f8975cfad7ec50b93ddf4455959ca7cb45bea6147f30359c72`.
It requires48 application-parity cases (IDs0/3 × N64/128/256 × depths1/16 ×
complex64/128 × saved physical source/grazing Nyquist probe) and12 ownership/
stale-state guard bundles before any full solve. Limits remain1e-4 for64 and
1e-9 for128, with additional bitwise equality. Complex128 rows exercise the
original native fallback. TSA8 is then tested through the full N64/N128 solves,
rather than through a separate standalone depth8 entry grid.

Full verification uses **all16 existing development media** (seed4626) on each
of three nested grids, checking all seven regenerated material/source fields.
The original frozen recipes remain `reference_tail` (TSA8 with original
reference initialization/balance and one tail reference) at N64/N128 and
`tsa16_zero` at N256. Only the required base is constructed. Per arm/case:
one recorded full-solve warmup and two measured repeats, with shuffled first
order and reversed second order. The48 material/grid inputs represent16
independent media, not48 independent media or192 independent timing samples.

The launcher enforces a300s direct-child timeout and checks no other wjt
research processes before launch. It never terminates another process. The
runner also checks the deadline between phases/cases/solves and records
partial results. A narrow future interruption caveat remains in this frozen
revision: its final checkpoint writes `settings.complete=True` before other
tables, so a kill within that write window could leave that one file marked
complete. Always require the launcher's exit/timeout receipt and all expected
tables. This actual run exited0 without a timeout and has every expected row.

## Observed results

Process wall218.324s; runner wall216.263s. Before launch, no other wjt research
processes were found; afterward the GPU returned to the same desktop process
using8MiB. All48 operator checks passed bitwise, all12 guard/ownership bundles
passed, and all288 full solves (96 recorded warmups +192 measured repeats)
passed independent accuracy checks. Every paired full solution was
`torch.equal`; original iterations/classical-call counts matched.

Worst independently recomputed original-FP64 residual:9.9850116e-6.
Minimum finite positive stored-phase coercivity margin:0.0127060314.
Maximum evaluated relative error bound:2.3688606e-4, below1e-3.
Maximum paired field difference:0. The coercivity/error calculation uses the
actual weighted phase and original FP64 A/RHS, without directed rounding; it
is a discrete numerical estimate, not a continuous-error certificate.

Warm times are means over16 media after reducing two repeats to each medium's
median:

| Grid | Frozen original | Frozen graph v2 | Warm saving |
| --- | ---: | ---: | ---: |
| N64 |134.205ms |33.328ms |75.166% |
| N128 |136.580ms |40.847ms |70.093% |
| N256 |148.651ms |96.806ms |34.877% |

Every individual warm comparison improves. Individual savings range72.86–
76.27%,67.51–71.87%,34.21–35.27% respectively. The declared exploratory
development gate (>=5% at N128/N256; no >10% individual regression) passes.
No held-out, OOD, multi-user GPU-load or population claim follows.

Whole-process peak allocated memory:1,151,242,752bytes; reserved memory:
2,409,627,648bytes. These include entry captures/guards and the whole screen,
not an isolated one-method memory requirement.

## Setup and conditional reuse costs

Mean common physical/required-base setup is7.832/8.454/11.208ms. Mean extra
graph setup (three side-stream application warmups + capture) is120.481/
122.534/186.229ms for N64/N128/N256. Both arms charge common setup; only the
graph arm charges graph setup. Captured input copy and fresh output cast are
included in solve timing. One timer limitation remains: `setup_seconds` stops
immediately before the final host reference/signature binding, so that
unmeasured host step is omitted from the Q estimates. Base construction already
prepares and charges the reference inverse; the graph's repeated `prepare()`
uses that warm cache. No initial inverse setup is missing.

The following are **conditional setup + Q×warm-median estimates**, averaged
over16 media; repeated measurements use the same saved RHS. They are not
measured distinct-RHS solves or process-cold first-solve latency:

| Q | N64 estimated saving | N128 estimated saving | N256 estimated saving |
| --- | ---: | ---: | ---: |
|1 |−13.80% |−18.48% |−84.06% |
|4 |51.96% |46.94% |3.49% |
|16 |69.30% |64.24% |26.92% |
|64 |73.70% |68.62% |32.88% |

Mean-cost first integer break-even Q is2/2/4. This is not an individual-media
guarantee: at Q1 only5/16,4/16,0/16 graph estimates win, and at N256 Q4 only
8/16 win. At Q16 and64 every individual conditional estimate wins. Keep the
eager classical recipe available for single-RHS setup-inclusive comparisons;
any learned candidate must match the actual deployment/reuse cost scope.

## Authoritative private artifacts and call contract

Local archive: `receipts/headroom_v2/fwi_v2_receipts.tar.gz`, SHA256
`3db00d3a448368b65057f4d04e66c14967b182d436b1b26bbc6242168e2a4135`.
Safe regular-file extraction: `receipts/headroom_v2/verified_remote/v2/`.
Complete file hashes: `receipts/headroom_v2/verified_artifact_manifest.json`.
Independent Q arithmetic: `receipts/headroom_v2/independent_cost_summary.json`.
See `staging_receipt.json`, `execution_receipt.json`, `collection_receipt.json`
and the `full16/` parity/guard/solver/setup/median/settings JSON files. The
consuming parent owns Library materialization using its current local paths.

The already executed remote command was:

```sh
/data1/software/miniconda3/bin/python \
  /data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/linear_headroom_20261002/v2/execute_graph_full16_v2.py \
  --policy /data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/linear_headroom_20261002/v2/graph_full16_v2_policy.json \
  --output /data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/linear_headroom_20261002/v2/full16 \
  --max-wall-seconds 300
```

Existing output/log/receipt paths are rejected, so this is a provenance record,
not an instruction to overwrite or repeat this run. Frozen source hashes are
in the policy. No execution blocker remains. The learned-project objective
remains open; this successful classical implementation must strengthen its
next comparator, while the failed94-transform precision results remain intact.
