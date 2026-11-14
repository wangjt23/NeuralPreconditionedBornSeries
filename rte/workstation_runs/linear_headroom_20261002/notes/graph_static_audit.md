# Original classical CUDA-graph screen: static audit

2026-10-02; source review only. No CUDA computation, GPU job, held-out access, source modification or numerical-result validation was performed by this reviewer. Inspected `graph_classical.py`, `graph_oracle.py`, `graph_policy.json` and the existing original solver paths.

## Verified scope and sequencing

The graph captures the original prepared `ClassicalTransportStack` implementation. It preserves its stored FP32 operator/reference cache and all original TSA stages; it does not use the failed94-transform algebraic formula. The capture counter expects `fftn=ifftn=4*depth−1`, or126 transforms at depth16. Side-stream warmup waits for the current stream, then the current stream waits before capture.

Complex64 replay copies into a static input, replays the captured graph, and converts the static complex64 output into **fresh complex128 storage**. The returned field therefore survives later replays, including retention as a Krylov Z vector. The caller's additional `.to(complex128)` is then a no-op on that already owned returned allocation. The script explicitly holds one output across another replay and checks it is unchanged. The complex128 input path invokes the original base rather than the complex64 graph; native complex128 parity is a fallback check, not a wide-precision graph result. The wrapper is used serially here; concurrent replay calls sharing its buffers are outside the verified scope.

The same48 frozen cases cover three grids, IDs0/3, depths1/16, complex64/complex128 input and physical-source/Nyquist probes. Parent hashes and all seven regenerated material/source fields are checked. Every row requires finite outputs and original physical-image/field parity at1e-4 or1e-9. The script asserts all48 pass **before its first full solve**. Standalone application timing occurs during this entry phase, but no full solver does. The later TSA8 graph is not itself among the depth1/16 application-parity rows; its fixed implementation and full-solve accuracy are separate evidence.

## Cache guard gap

Settings, grid/setup generations, and tensor pointer/version/dtype/device/shape metadata are compared before replay. Ordinary setup and in-place edits are covered by those checks. The tested stale-state example changes only depth.

The wrapper retains no independent tuple of captured tensor objects and omits stride/storage-offset/operator-object identity. A concrete bypass is replacing square-grid `low.sigma_t` with `low.sigma_t.transpose(-1,-2)` without setup: its pointer, shared version, dtype/device and shape remain equal while native material layout/action changes. Replay retains the captured layout, so the current metadata check can accept a stale graph. Direct replacements can also release captured external allocations without strong references. Retain the original tensor objects, require `is` identity plus versions and base/operator identity, and reject before rebuilding a missing cache. This is a wrapper-hardening issue; the inspected frozen screen performs no such material/cache/view replacement, so it does not invalidate that run by itself.

## Accuracy and budget enforcement gaps

Full-solve residuals are independently recomputed against unchanged original FP64 A/RHS. Timed rows include final-field comparison and the stored-phase coercivity diagnostic. However, the code's `accuracy_passed` expression should explicitly require a finite positive margin and finite returned fields. A negative margin could otherwise produce a negative apparent error bound that passes `<=1e-3`. Inspect those actual values independently in the receipts rather than assume the theorem applies.

Warmup outputs have an independently evaluated residual, but no full accuracy-gate row is retained. Timed accuracy failures are recorded but do not stop subsequent cases. The declared300-second soft budget is not checked in the script. Disclose these enforcement limits; do not describe all warmups as accuracy-validated or the budget as implemented merely from the policy. A failing accuracy arm must remain a negative/inconclusive result, irrespective of its time. These observations were sent to the root while execution was active.

## Counting and comparator fairness

The full-solve bound is96 total invocations: six material/grid inputs × four methods × (one warmup + three timed repeats). It yields **72 measured rows and24 warmups**, not96 measured solves. These represent two independent media on nested grids. Parity applications and standalone apply timings are additional diagnostic work; timing repeats are not extra scientific samples.

The frozen-graph arm uses graph TSA8 inside `reference_tail` at N64/N128 and graph TSA16-zero at N256. Reference initialization/tail injection remain original native operations and are timed. Graph preparation is extra to original material/base setup; charge the used per-grid graph's warmup/capture and input/output copy/cast costs. Both TSA8 and TSA16 graphs are prepared by the screen, but an eventual single-method first-RHS estimate should identify which graph is actually required rather than silently count unused graph setup as zero or necessary.

Reference1/2-zero are additional classical frontier controls. Any successful graph/shallow improvement is an **expanded traditional comparator**, not an ML contribution or a learned-project victory. Freeze subsequent portfolio selection on development information; show the prior frozen recipe and expanded recipe separately, and compare any learned candidate with the successful expanded implementation fairly. This two-media screen alone supplies no population/OOD claim. No timing or numerical outcome was inferred from this source audit.
