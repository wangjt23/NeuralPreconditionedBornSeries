# RTE FWI: verified exact capture confirmation

Verified 2026-10-02. The authorized bridge recovered; explicit-capture screen, full16 development confirmation and final preservation audit completed on FWI. The earlier environment_offline error in EXECUTION_BLOCKER.json is historical. No direct-capture job launched during that failed attempt. Current access has no unresolved blocker.

## Identity and scope

SSH alias FWI reached wjt on FWI-Standard-PC-i440FX-PIIX-1996, login /home/wjt. Repository/realpath: /data/wjt/NeuralPreconditionedBornSeries, subtree rte/. Branch main, HEAD 028caea3f35d100225f656e2b1e910c1366aaafe, origin git@github.com:wangjt23/NeuralPreconditionedBornSeries.git. RTE is the steady radiative transfer equation: existing periodic 2D unit-square FFT model, twelve angles, g=.65, epsilon=.1.

All substantive new runs used FWI's RTX A6000 (49,140MiB), existing /data1/software/miniconda3/bin/python and Torch2.5.1/CUDA12.4. Mac CPU fixtures/receipt analysis were auxiliary. Every SSH call disabled configured forwarding, retained existing host-key checking/credentials, and used bounded connection attempts. No trust/access/credentials/SSH/proxy/VPN/network changes. Separately paused OL_preconditioner untouched.

## Scientific audit and unchanged authority

Accepted H=high_precision_copy(low) promotes the stored FP32 material/geometry buffers and recomputes streaming products in FP64. This is the original finite discrete operator, not ideal/native geometry or continuum RTE. Original real-stacked augmented FGMRES remains tol1e-5, max100/restart25 with source-normalized physical residual. N128 uses original TSA8/reference-tail initialization/injection; N256 uses TSA16/zero initialization.

For skew-adjoint periodic streaming, H=S+tI-sK and positive m=min(t-s lambda_max(Herm(K))) imply exact-arithmetic ||e||<=||q-Hx||/m. The evaluated relative bound divides by ||x||-||q-Hx||/m when positive. Eigenvalues/margins/residuals are ordinary FP64 computations, not directed-rounded certificates. Gates retain finite solutions, original residual<=1e-5, evaluated relative bound<=1e-3 and paired field difference<=1e-4. Capture additionally requires bitwise fields and entire solver-status equality.

Training3626, repeatedly reused development4626, historical heldout8626 and reserved9626 are distinct. Existing development parents were hash-verified/regenerated for identity. No new training/teacher data;9626 stayed sealed. Sixteen development media at each resolution are exploratory confirmation, not independent generalization evidence. Nonperiodic boundaries, changed angular/absorption regimes, continuum errors and uniform convergence remain untested.

Retained negatives:126->94-transform TSA rewrite failed FP32 parity; FP64 accumulation did not fix it. Exact full replay improved warm costs but paid substantial Q1 setup. Fixed-angular q3/whitening failed charged warm gates; raw-source brackets around.60-.66 do not settle projected-frontier optima. The established full transport response U=H0^-1Q, Q=(0,+1,-1), plus TSA4 homogeneous correction lost all four setup-inclusive contexts; removing response construction alone did not rescue it. This does not prove all heterogeneous coefficient maps impossible. Earlier learned tails saved no mean strong-classical calls. Primary-source comparisons/ranked alternatives are in NEXT_RESEARCH_DECISION.md; no novelty follows merely from angular reduction, material caching or residual linearity.

## Actual change and tests

New helpers/code live only under rte/workstation_runs/direct_capture_20261002. DirectFullGraph/DirectSegmentedTSA retain explicit warmed nondefault streams, caller waits/synchronization, global capture-error checks, private pools, strong tensor/cache/identity/version/layout guards, serial ownership and native complex128 original-eager fallback. Raw capture avoids context entry's GC/cache-flush calls without changing global GC/allocator settings. Uncertain capture failure aborts; no eager fallback. The full helper preserves the original body error if capture_end also fails.

Selected direct segmented capture keeps the original eager first TSA stage and captures one unchanged subsequent stage, repeated d-1 times with billed copies. It preserves arithmetic/rounding and4d-1 fftn plus4d-1 ifftn per action, rather than rewriting physical algebra. Both helpers passed16 FWI CPU tests each (32 total), including ownership/cache/native-wide and capture lifecycle contracts. Local flat-staged32 CPU fixtures passed with existing Python3.9.6/Torch2.8.0/CUDA unavailable; they did not replace FWI GPU testing.

Five contemporaneous arms: eager, context full, context segmented, explicit full, explicit segmented. Constructor order counterbalanced; solves shuffled. Shared physical/base/coercivity setup, whole capture construction/warmups/host binding, copies, solver and independent residual/error/finite acceptance are billed. Field/bitwise QA, profiling, source IO and interpreter startup are excluded. **Q1 is recorded per-material setup plus median warm accepted solve, not fair process-cold or total service latency.** First-use charges remain visible, including all four confirmation shard charges.

| Experiment | Full solves (timed/warm) | Entry checks | Result |
|---|---:|---:|---|
|Prior homogeneous response|96 (72/24)|20|Accurate; all four Q1 comparisons lost|
|Prior context segmented|48 (36/12)|52|14.179% N128,3.989% N256;5%BOTH gate failed|
|Explicit capture screen IDs0/3|80 (60/20)|108|13.331% N128,21.738% N256; gate passed|
|Frozen full16 confirmation|640 (480/160)|864|20.784% N128,19.645% N256; gate passed|

The screen selected ONE backend per grid, `direct` segmented at both. frozen_selection.json records screen policy/archive/execution/audit hashes before confirmation. Four serial four-ID shards each used both grids, all216 entry checks before160 solves,3 timed repeats/1 warmup,100s soft/120s hard bound. No reselection/per-case selector. All four finished within bounds: total process251.4765s, combined runners235.9584s; maximum across shards allocated1,763,894,784B/reserved3,686,793,216B.

|Grid|Best prior mean Q1|Frozen direct mean Q1|Saving|Worst individual saving|
|---|---:|---:|---:|---:|
|N128|223.3476ms|176.9263ms|20.7844%|7.9839%|
|N256|162.1031ms|130.2582ms|19.6448%|8.9071%|

Direct is cheapest among ALL FIVE successful arms in all32 contexts. All864 checks/640 unique full solves pass bitwise field and entire solver-status/residual equality, original work/FFT counts and accuracy. Minimum evaluated margin.01270603136; maximum physical residual9.836659818e-6; maximum evaluated relative bound.000234034808. Actual execution roots: confirmation/shard0..3. Local direct_confirmation_merged is explicitly derived analysis, preserving all shard charges.

Profiled copy controls measured gc.collect60.860–78.931ms (89.154–96.316% of control totals), empty_cache cumulative.768–8.544ms. Overlapping capture-begin cumulative time is not additive. Raw copy reused prepared buffers/stream; totals alone are not matched causal attribution. GC dominance is established in these profiled controls, with production attribution qualified. Actual whole-constructor/solver measurements and parity establish the tested classical improvement.

## Learned headroom and next direction

This is classical credit, not learned superiority. With H=(1-g)Cb-Cc and savings db for baseline/dc for candidate, Hnew-Hold=dc-(1-g)db. An unchanged candidate loses allowance; equal savings preserve absolute advantage and improve gate margin only by g*db. Candidate inheritance requires implementation/measurement.

Against the new ALL-FIVE baseline, a hypothetical same-setup candidate with ZERO extra costs needs mean warm reductions8.8463/6.5129ms for5%Q1 advantage at N128/N256, or17.6926/13.0258ms for10%, before acquisition/inference/offline costs. These are necessary whole-solve allowances, not feasible correction budgets. The5%classical gate does not weaken the prior10%learned gate. No actual affordable learned progress witness emerged; earlier geometric/cost negatives remain.

Ranked next: (1) matched shallow-TSA/full-angle reference-defect control with already-paid inverse, same precision/material/start, compared to q3 using actual useful H-images and charged costs; (2) constrained cheaper residual-linear replacement only after real action/storage reduction; (3) one fixed heterogeneous coefficient correction in U only if closure/progress and budget support it. Preserve R1/R2/TSA4/q3 negatives. Do not train, expand q or acquire fine teachers merely because capture became cheaper. Remaining gaps: affordable learned action, independent heldout/OOD, rigorous rounding, continuum/nonperiodic scope, all offline costs over a declared served workload, matched process-cold and distinct-RHS reuse.

## Preservation and deliverables

Final postexperiment audit verified318 scoped hashes with no mismatch: all155 original sources; original data/models/development parents/reports;28 prior response/segmented payloads; current execution code/policies/results against downloaded bytes. Main/HEAD/origin unchanged; all9 original tracked dirty files unchanged relative to preflight. No other research process; GPU37MiB total/8MiB desktop only. Exact evidence: receipts/direct_final_preservation.json; remote final_preservation.json.

No new training,9626 access, original-data overwrite, unrelated-job cancellation, Git commit/push/merge, public sharing, paid compute, package/system/access/network change. No continuation job remains.

Local deliverables/response, deliverables/segmented and deliverables/direct_capture retain code, protocols, audits, ranked decision and negative results. Raw receipts are in response_cost, segmented_screen, direct_screen and direct_confirmation. Earlier verified private bundle RTE_FWI_Exact_Replay_Angular_Obstruction_2026_10_02.zip, libfile_6cba03176e988191aca1f593b55205cb, SHA4ae6ae69935cea8c815255e4122bfb8a1ff7efe1245d8109d32bf5ab2c0d91d9 was neither re-materialized nor reuploaded. New report/bundle Library identity and exact download verification are recorded separately after saving, avoiding self-referential hashes.

Automatic approval review rejected an optional eight-file original-source export to the Mac before process creation. It was removed, not retried. The bundle includes new experiment code, results and original hashes/provenance; original source bytes remain a dependency in the preserved FWI repository. Authorized documentation saves continue without that export. Exact review/resolution is in SOURCE_EXPORT_REVIEW.json.

For future work refresh FWI identity/HEAD/jobs and preserve these isolated helpers as strongest successful classical controls. Stage a newly frozen matched discriminator if pursued; this handoff does not authorize a training campaign or holdout access. Code SHA identities: direct_segmented98eb806b8482990c1727cb80de3a2182c520be01090a512b742feb0417e0d1d1; direct_fullcf29f0b3d4b2dfe773660ac06a33385216719f6fcf363c1a33d44973ab372cd0; driver00d9dc949871ccb45cf6a479d6d97260a8d6451926c0d8ffdc16acba92edd1c6. Screen policySHA b5f430a0287b331c814c3d8987ee8d0a9d6b3cf8b5f8fdd22795139a875537e2.
