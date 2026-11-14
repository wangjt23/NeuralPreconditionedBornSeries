# RTE + mgRTE integrated lightweight handoff
## 0. Identity and authority
Supplement to the historical HANDOFF.md and mgrte/HANDOFF.md. User explicitly authorized a new branch in wangjt23/NeuralPreconditionedBornSeries on 2026-11-14. This is an archival comparison, not a production merge. Main and running worktrees are untouched. Pro combined-route synthesis is pending, not consensus.
## 1. Problem
Learn better preconditioners by adapting analytic/iterative structures. Periodic Fourier RTE O/K and historical ATFPS reduced diffusion-mode mgRTE solve different discrete systems: do not treat them as head-to-head comparisons.
## 2. Claim ledger
Periodic: four O/K two-seed endpoints completed 2048 updates. Recorded TRAIN8/extra RHS16/DEV32 all reach external original residual 1e-5. K improves self-FGMRES over O but loses to CT/SCT strong baselines. Complete independent numerical rerun is not certified.
ATFPS: 8 saved diffusion media x 2 RHS x BJ/Mg1000/Mg2000 = 48/48 batch-one solves reach external residual 1e-6. Saved-media independence from training is unresolved; no unseen-media generalization claim.
Cached BJ: 64 probes exactly match original BJ; 48 paired histories/iteration counts/final residuals match. Local implementation equivalence is not independent PDE certification or strongest multilevel baseline evidence.
## 3. Reasoning and limits
Physical RHS refers to selected-mode reduced ATFPS, not the full continuous PDE. Original coefficients are FP32 promoted to FP64 for A; learned preconditioning is FP32. External residual shares the same A implementation and saved solution vectors are not included for independent residual recomputation. Historical checkpoints were selected by training loss. Keep negative periodic and promising saved-media evidence separate.
## 4. Algorithm and two code trees
mgrte/mac and mgrte/fwi preserve distinct dirty trees at base 0e05cc8930f892c9f3fb580adef4ac4b93c26ce7. Preserve patches separately, never flatten or apply both blindly. TREE_DIFFERENCES.json records common-file differences and unique files. Packaging introduces no solver modifications.
## 5. Cost model
Count online construction, model loading, warmup, inference and full original-equation iterations. Original BJ repeatedly assembles local solves. Cached LU is the fairer comparator. Three repeated totals reuse one measured setup/warmup charge, not three newly measured setups. Model load about 0.207/0.195 s and operator generation excluded. Concurrent GB jobs existed: not idle-GPU timing. Fixed-forward single/four-process throughput applies only to that workload, not general optimizer training.
## 6. Numerical evidence
Periodic mean M calls [TRAIN,RHS,DEV]: O1 [33.625,33.8125,35.625], O2 [32.875,32.9375,34.875], K1 [26.625,28.875,31.5], K2 [26.5,28.5625,31.65625]. DEV K approximately 247-248 ms; CT4 36.835 ms/5.5625 calls, SCT4 37.677 ms/5.59375, TSA 76.439 ms/15.96875. Different concurrent conditions preclude precise speed ratio. Old fixed Arnoldi defects favor O (.228/.235) over K (.319/.320), unlike full own-solve performance. Retain short windows and early convergence.
ATFPS physical mean iterations BJ58.25/Mg1000 9.75/Mg2000 12.625. Cached BJ setup+warm+solve physical repeats 6.031/6.113/6.078 s vs Mg1000 2.305/2.324/2.311; random 7.077/7.077/7.103 vs 2.426/2.421/2.435. Raw JSON and scripts are authoritative; limits above apply.
## 7. Artifact inventory
Original slim code/evidence preserved. mgrte/ contains both source snapshots, patches and inventories with external SHA/location. integrated_evidence/ captures replay/cache audit and current periodic endpoint/Arnoldi/fixed-panel text evidence. Its INVENTORY.json describes copies and omissions. Data, weights and files >1 MiB are external, never deleted at source. This packet is source-first, not numerically self-contained.
## 8. Reproduction
From repository root run python3 handoff/rte_20261114/verify_handoff.py and python3 handoff/rte_20261114/mgrte/verify.py. Restore external objects from source locations and verify hashes before numerical replay. Existing scripts retain FWI absolute paths and require safe path adaptation on another host. Do not target active outputs. No new training was run as packaging validation.
## 9. Prior art and Pro provenance
Analytic ansatz, multigrid/Born-inspired neural iteration are user motivations, not established novelty. See source references. Pro conversation https://chatgpt.com/c/6af7bfca-fe30-83ea-8cf0-4d53daa4a65c received text evidence, not a complete source audit. Combined-route V5 synthesis pending at capture. Full Pro transcripts are not archived; do not infer a final consensus.
## 10. Open problems
Unseen-media independence, strong multilevel baseline, independently implemented A, full online cost, complete fixed/common-prefix diagnostics and total independent numerical endpoint audit remain unresolved. External data required for reproduction.
## 11. Next plan
Await combined Pro critique, compare problem settings and training sufficiency, then test promising historical structures on genuinely held-out media with equal residual tolerance and full online costs. Do not generalize periodic negatives to ATFPS, or saved-media successes to unseen media. Preserve failures, fixed budgets and DEV isolation.
## 12. Integrity
See INTEGRATED_VALIDATION.md for hashes, source parse/compile, size and secret checks. Remote SHA is verified separately after push. Upload is not scientific PASS. Missing external dependencies and original failures remain explicit.
