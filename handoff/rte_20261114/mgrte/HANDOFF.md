> DESTINATION UPDATE: user explicitly authorized this supplement in wangjt23/NeuralPreconditionedBornSeries new branch. Earlier rte_pre destination below is historical.

# mgRTE / rte_pre lightweight reconciliation handoff
## 0. Identity and authority
2026-11-14. Source-first two-tree snapshot, not self-contained numerical archive. Both trees have HEAD `0e05cc8930f892c9f3fb580adef4ac4b93c26ce7` on `RTE_born`. Mac origin is SQC31/mgRTE and secondary `rte` remote is xlliu2017/rte_pre; FWI origin is xlliu2017/rte_pre. No push or commit performed. This supplements, and does not supersede, either live tree or the separate NeuralPreconditionedBornSeries package.
## 1. Problem and target object
DOM/ATFPS steady-state RTE with inflow boundary treatment. This is not the current periodic Fourier RTE campaign. Literal discretization, selected modes, source normalization and solver tolerances must be taken from each version's configuration and output JSON; no mixed global performance claim is made.
## 2. Claim ledger
- IMPLEMENTED: two distinct working trees and their common HEAD verified; exact source and patch preserved separately.
- MEASURED: file hash equality/difference, inventories and omitted-byte totals are packaging facts.
- OPEN: deciding which divergent training/preconditioning implementation is authoritative; neither silently replaces the other.
- RECOVERY-ONLY: numerical efficacy claims in historic notes not independently rerun here.
## 3. Load-bearing proof chain
No new theorem is claimed. Data generator -> ATFPS tensors -> learned correction -> true-residual GMRES is an implementation dependency chain, not a proved convergence argument. `docs/training_scheme_audit.md` identifies fresh residual and relative-loss geometry. Mac-only `docs/born_failure_architecture_notes.md` records metric/physical residual mismatch and a U-Net candidate; these are historical analysis, not fresh validation.
## 4. Algorithm contract
Use exactly one tree. Mac has changes to MgNet/U-Net architecture, configuration, inference and tests; FWI has loader, train, preconditioning and replay collection changes. Preserve separate source namespaces. Do not apply both patches to one checkout. Each patch is relative to the common HEAD and excludes untracked files, which are included separately in source snapshots.
## 5. Complexity ledger
Packaging is lightweight text/notebook only. Numeric data, model weights, image outputs and compiled document artifacts are external. No training runtime, speedup or memory conclusion is established by this packaging action. Current periodic Fourier timings must not be attributed to ATFPS mgRTE.
## 6. Numerical evidence
FWI source contains raw/Born/replay diffusion and interface inference JSON and loss logs. Mac contains tiny audit inference JSON and architectural notes. No experiments rerun; no numerical PASS asserted. The 18 historical model weights and separate diffusion/interface/replay tensors are listed but not copied.
## 7. Artifact inventory
`mac/inventory.json` and `fwi/inventory.json` contain complete scoped non-cache file inventories, SHA256, bytes, original retrieval path, materialization flag and omission reason. `.git`, Python caches, pytest cache and virtual environments intentionally excluded. `TREE_DIFFERENCES.json` lists every shared hash mismatch and unique path. Data retrieval remains restricted to authorized access to the source machine; paths are not public links.
## 8. Reproduction commands
Run `python3 verify.py` from this directory for copied-source hashes, JSON/notebook parse and Python syntax. It does not run model training or numerical tests. For a clean independent replay, check out the pinned HEAD in a new checkout, choose one tree, apply its `working-tree.patch`, then restore its source snapshot (including untracked files). Retrieve omitted required tensors/weights using their exact source paths and verify SHA256 before executing the chosen tree's README commands. Never apply patches in the live trees. Environment/model smoke tests and complete numerical reproduction remain unexecuted.
## 9. Prior art and claim boundary
McMg/Born and ATFPS ingredients are existing research; no novelty claim. Both mgRTE trees are separate from NeuralPreconditionedBornSeries's partial `mgRTE_ablation`. The latter does not cover all these files or data.
## 10. Open problems
- Confirm publication destination/scope before any upload; previous NPBS upload approval is pending and cannot be reused as mgRTE approval.
- Resolve divergent implementation semantics before merging; preserve both variants until user/research review decides.
- External data and weights are required for full numerical reproducibility. Generated data may not reproduce exact bytes without recorded seeds/runtime.
- Full relevant Pro transcripts are not bundled here. Existing dedicated Pro discussion concerns the periodic campaign; do not imply it audited these ATFPS sources.
## 11. Next plan
Suggested repository separation: periodic campaign and its existing handoff remain in wangjt23/NeuralPreconditionedBornSeries; independent mgRTE sources and this two-tree supplement belong to xlliu2017/rte_pre, without overwriting main or discarding either dirty tree. Cross-link the two rather than flattening. Include a compact compare ledger and explicit external-data manifest. Do not upload until scope is confirmed.
## 12. Integrity checklist
Head/status/remotes captured; separate patches and source snapshots; copied hashes and parse/syntax checks available; missing binaries explicitly inventoried. No source worktree modified, no Git commit/push, no remote SHA or CI PASS claimed. Numeric tests not run. All result claims limited to existing outputs and packaging validation.
