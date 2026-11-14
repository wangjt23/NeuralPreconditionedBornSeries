# E at N256: preliminary pure-scale CPU pass

## Result and scope

The unchanged E sequential angular-block implementation passed the predeclared N256 gate on the same constructed physical scene and four exactly nested saved sources. Its fully paid four-source cost was **77.254074623 s**, versus **98.671172381 s** for compact4, the cheapest accepted CPU control: **21.7055% less observed time**. Genuine TSA16-zero cost105.235147658 s and TSA64-zero107.707240995 s. This is one forward run, not a repeated timing claim, broad material generalization, convergence study, neural gain, or victory on the original sixteen-material Q1 protocol.

The first-source result differs: E cost **31.411620836 s**, slower than TSA16-zero25.910007523 s and TSA64-zero27.356099055 s, although faster than compact4's43.362585682 s. The first-source cost is measured directly, not the four-source total divided by four. No source was added to improve amortization. E is currently relevant to this CPU-constrained four-source workload; available GPU compact4 remains a much faster execution option in its separately measured N128 workload. No N256 GPU time or cross-device speedup was measured here. Closed I/G implementations remain closed.

## Identity and immutable implementation

The governing plan is FROZEN_PLAN.md, SHA2563be997b61d1f65d10ca7c254ff7f2b3d752cfc7c5694513a5e7244f22b75ea3f. E is imported unchanged from rte_angular_full_20261110_dot_v1: angular_modes.py SHA256d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481, run_full_cpu.py SHA2562450fa79b22f0640fd36cd5cd7571b37a6c219ae9130eab341b901f67de2d5de. All dependency hashes matched before and after execution. This driver changes size-dependent qualification dimensions, not E's mathematics.

Material is the existing fixed screen_structured.medium layered-plus-Gaussian construction, periods1, beta10/3, transition physical width2/128, inclusion center(.34,.60), width.08, amplitude3. The width is not rescaled to2/256. The N128 total/scattering hashes match sealed E, and N256 even-grid entries equal N128 exactly. N256 total SHA25622703485ebf1a4e3bd4eba553a4a3f714b7d31253109c22d7af2108f4cc4d7b4; scattering SHA2569750a48fb0a4734805c6824640566b36e5536d6fe2322470a211e47a40c2db31. Background axis0 has total SHA256a4289ae8ce23fe0e91dbe1a89e43c2aac2af430e0b0d6ccc4bd54341a232347c and scattering SHA2564b99abec1782b55fbe1c4ff3bfc14cf71037f722f4fb490636477b6047dfee70. Fresh background hashes matched in every method; no old factor was reused.

Sources remain IDs[0,4,7,14]. Existing N256 source records are used only for their saved source arrays and provenance, never their neural solution fields. Each source is an exact complex64-to-complex128 promotion and its even-grid entries equal the original N128 development source exactly. SAVED_INPUT_AUDIT.json and analysis.json contain all four containing-file/source hashes. The older source-record parents establish provenance only; their different random material arrays are not substituted for the constructed scene. N256 is a previously used development resolution, not an untouched material test. Seed9626 was not opened. No training, accurate teacher, new source generation, or CUDA operation occurred.

## Mechanical failure and independent repair

Attempt1 exposed a driver error: its TSA16-labelled batch called construct without forwarding depth, so the actual object used64. That label is invalid evidence for TSA16. Only E's first source had finished when the owned worker was terminated. Attempt1 has no complete E result and remains separately sealed as rte_J1_N256_mechanical_failure.zip, SHA256f434bb3c80c52fba832dca98ba244ab28bcd465c28527182f18765cfb277755d,40037 bytes.

The sole numerical driver repair is construct(rhs[0],depth=depth), accompanied by a hard runtime binding assertion. The independent AST/stub test imports no solver/Torch: it reproduces old16-to64 binding, verifies all four new mappings, and rejects deliberately dishonest metadata16 when the real stub object remains64. DEPTH_BINDING_TEST.json records this qualification. Actual runtime ClassicalTransportStack objects report depths64/64/16/64, kind tsa, reference shape256x256 and cache shape1x256x256x12x12. Effective depth is read from the actual object, not copied from a requested method label. No threshold, material, source, ordering, or solver setting changed.

Attempt2 source run_n256_gate.py SHA256e12e8aacb260e98bbc34148e6817cde2d2677ce756d74b365c0bd99edeb79806. Raw attempt1 data, E source, and all earlier sealed packages were not overwritten.

## Contract and independent acceptance

The original periodic Fourier operator uses the actual stored geometry,12 angles,g=.65, original normal Nyquist, and both signed tangent modes-8..8. Outer tolerance1e-5/max100, zero initial guess; compact4/E retain one enrichment after prefix1. E's original full-grid P8 A P8 inner solve retains tolerance1e-4/max8. Its approximate angular band representation is repaired against the original G with tolerance1e-10 and at most3 corrections. It is not bitwise equal to an idealized angular operator. Strict zero columns remain exactly zero; nonzero/tiny columns use scaled relative norms without an absolute tolerance floor.

There were34 independently checked mode records and12 zero/mixed/tiny boundary records across the disposable preliminary qualification and fresh paid E setup,70 full original-A0 vector checks in total. Maximum original-A0 residual was8.013796204339805e-12. All zero/tiny tests passed. At runtime19 inverse applications covered323 mode inversions. Every mode required exactly one defect correction, maximum final original-G residual3.933267923629979e-12. No correction budget was relaxed.

All16 method/source results passed original full-A acceptance. Maximum original residual8.955381855600596e-6 versus1e-5; maximum evaluated relative error bound4.0652210311522486e-5 versus1e-3; maximum field difference3.3715221796641204e-7 versus1e-4 against the ordinary TSA64 accepted outputs. E alone: original residual at most7.420198105168608e-6, evaluated bound3.218178843218502e-5, field difference1.6266668376084935e-7. Maximum original projected inner residual8.893509010616991e-5 versus1e-4. These are not comparisons against an uncharged high-accuracy teacher.

## Paid costs and calls

| Method, fixed forward order | Setup s | Measured first source s | Complete four sources s | Outer classical calls |
| --- | ---: | ---: | ---: | --- |
| TSA64-zero |0.401890818|27.356099055|107.707240995|5,5,5,5|
| compact4 |21.800463662|43.362585682|98.671172381|4,3,4,3|
| TSA16-zero, actual depth16 |0.401549071|25.910007523|105.235147658|17,17,17,17|
| E, unchanged sequential blocks |14.692833674|31.411620836|77.254074623|2,2,2,2|

E's fresh factor construction cost6.019369629 s, and its full17-mode original-A0 plus boundary qualification cost8.077273854 s. Both are included in setup14.692833674 s. Compact4's factor construction cost21.397583971 s and is included in its setup. E's online solve plus checks totaled62.420803345 s, including15.297735864 s of inner work; compact4 online totaled76.215605241 s. E inner iterations were5,5,5,4. Outer/native A calls were48/48/144/36 for TSA64/compact/TSA16/E; E additionally used46 inner A calls. Every method paid4 separate final original-A calls. Qualification A0 and original-G repair work are additional paid activities, not hidden in these solver counters. Physical/material/TSA construction and all projections/conversions/repair/validation are timed. There is no host-device transfer.

Per-method observed totals include bookkeeping between sources, so setup plus reported online phases does not exactly equal the complete total. First-source and complete totals are direct clocks from each fresh method start. Shared source-file identity preparation cost1.188503764 s; disposable preliminary factor/qualification and stage overhead are separately present in the research wall. Timed methods receive the same preloaded source arrays and each pay fresh physical/operator/cache/factor setup; these totals are not interpreter/process-cold latency. TSA64's107.707240995 s is both a paid control and the cost of ordinary reference outputs used only for checks, not a teacher charged again inside every method. Its reference cost and the initial qualification are included in aggregate research wall, never omitted from the research ledger.

Logical E factor/mode data:1,105,707,336 bytes; compact4:1,359,249,408 bytes. These are the source's logical byte counts, not independent physical allocation measurements. Worker peak RSS3,287,212,032 bytes, below6 GiB. Since all methods ran in one process and allocator/high-water state persists, later per-method RSS cannot establish intrinsic method-specific peak memory.

## Resources and complete research ledger

Worker1287842 and controller1287840 ran on CPU0, one numerical thread, CUDA disabled. Each stage had180 s limit, total600 s and6 GiB worker RSS ceiling. Recorded stages were17.4681,107.8312,99.1159,105.7400,77.5457 s. Worker wall407.938739458 s; controller wall411.439319939 s; returncode0, reaped true, no stop reason. The difference includes controller/process startup and shutdown. Admission snapshots at every stage show no other active numerical job. This is a shared server with normal services and kernel activity, not a reserved whole-host performance benchmark. GB's earlier CPU28/29 pilot overlapped attempt1, not these recorded attempt2 snapshots; no known overlap is imputed to attempt2 merely because coordination was necessary.

Attempt1's controller wall375.676437430 s remains an additional failed research cost. Combined attempt1+attempt2 controller wall is **787.115757369 s**. The fresh authorized attempt2 individually stayed within600 s; the old cost is not erased or treated as free. Reporting, transfer and read-only audits are outside these numerical controller clocks. At2026-11-10T13:46:23+08:00, explicit ps inspection found neither worker nor controller present. No reverse run or new numerical process has been started.

## Decision

Accept only a preliminary same-scene CPU scale result: E's original precision and call pattern survived N128-to-N256, with paid four-source savings against all three applicable CPU controls. The first-source penalty and increased absolute factor/storage/online costs are real. One order cannot settle timing variability; one constructed material cannot establish general applicability. The evidence is sufficient to stop this gate and debate whether confirmation, a prespecified mechanism-changing material test, or source-horizon economics provides the most useful next information. No new numerical work is authorized by this report itself.

Reproduction/audit: gate_01/results.json is the full raw receipt; WATCHDOG.json is the controller receipt; analyze_n256.py is a no-Torch independent saved-result audit and analysis.json its output. SHA256SUMS and the lightweight sealed archive bind the source, plan, identity audit, depth test, logs, raw evidence and this report. Large source/solution arrays are not copied into the archive; their exact original locations and hashes are retained.
