# Independent direct-capture confirmation audit — 2026-10-02

The frozen **direct segmented** backend passed the unchanged classical gate: **20.784351% N128 /19.644823% N256** recorded Q1 saving against each material's cheapest successful prior eager/context-full/context-segmented implementation. Worst individual savings were **+7.983911% /+8.907118%**; all 32 contexts improved. Selection remained fixed from the screen. Direct is also cheapest among all five classical arms in every context. This confirms an implementation advantage on 16 reused development media, including the screen IDs; it establishes no learned or held-out gain.

Independently verified **640 unique full solves (480 timed/160 warmup),864 passing checks**, finite fields, zero failures, explicit bitwise field equality and equality of entire solver status/work/residual. Each shard completed 216 checks before its 160 solves. Minimum margin **0.01270603136**, maximum physical residual **9.836659818e-6**, computed relative error bound **0.0002340348081**. Certificates are ordinary floating-point discrete evaluations. Original TSA8 reference-tail/TSA16-zero recipes, physical acceptance and application FFT counts remain unchanged.

| Grid | Best prior mean Q1 ms | Direct setup ms | Direct warm accepted ms | Direct Q1 ms |
|---|---:|---:|---:|---:|
|128|223.348|127.031|49.896|176.926|
|256|162.103|27.542|102.716|130.258|

Warm direct solves remain slower than full graph; cheaper capture setup supplies the complete-cost advantage. Setup and independent acceptance are charged. Startup/IO, comparison QA and profile diagnostics are excluded: this is **setup+warm**, not matched process-cold latency. All four first-use common charges (401.556–455.211ms at first N128 contexts) remain. Repeats were reduced to medians, not independent samples; nested grids share the same media.

Raw/merged records match exactly after the explicit `source_shard` annotation; all 36 shard code hashes match the screen. Frozen screen policy/execution/audit/archive references and each shard selection/policy binding match. All negatives remain. Root separately reports all 318 final preservation hashes matching, including 155 original sources and nine tracked dirty files unchanged.

Across 32 profiled copy controls, exclusive GC took 59.417–86.716ms (88.313–97.324% of setup); cache clearing cumulative time was 0.777–7.306ms. Raw copy capture took 0.177–0.231ms. GC dominates these controls; cumulative capture entries overlap and must not be added. Differing preparation/profiling scope prevents exact attribution in unprofiled constructors or a universal cost floor.

All four processes exited 0 below 120 s hard/100 s runner bounds. Combined process/runner times were 251.476501/235.958447s. Maximum allocated/reserved memory was 1,763,894,784/3,686,793,216 bytes across five alternatives; GPU receipts returned to desktop-only use.

The classical gate remains **5%**, and the prior learned goal remains **10%**. Against the new all-five baseline C*, a candidate sharing measured setup S needs entire warm cost `T <=(1-eta)C* -S -D`, charging added acquisition/setup D:

| Grid | Target | Zero-extra-cost warm allowance range ms | Mean allowance | Mean required warm reduction ms |
|---|---:|---:|---:|---:|
|128|5%|12.990 to 84.944|41.049|8.846|
|128|10%|-10.086 to 69.055|32.203|17.693|
|256|5%|43.629 to 215.217|96.203|6.513|
|256|10%|39.747 to 202.405|89.690|13.026|

These are entire-solve allowances, not proposal budgets or feasible learned witnesses. The10% goal requires 17.693/13.026ms mean warm reductions before added costs. Two N128 per-case 10% allowances are negative under this recorded setup model; this does not prove aggregate or universal infeasibility. Earlier negative experiments remain intact.

Confirmation archive SHA256: `36084c5f1f1022a8bbfc7879536217688dad8cb73aa5cf4dcae7ef17e8bfc924`.
Frozen-selection SHA256: `e1de6b0fb7f929f9bbda092a09b12da50a4a33acc752e4387acd72a0429f30ad`.

Full costs/profiles/receipt hashes are in `independent_confirmation_audit.json`. Reproduce the raw audit without Torch/SSH/GPU:

```sh
python3 deliverables/direct_capture/audit_receipts.py receipts/direct_confirmation_merged --frozen-selection deliverables/direct_capture/selected_backend_mapping.json --output /tmp/rte_direct_confirmation_audit.json
```

Require both numerical/protocol and cost gates true. No new solver run, training, source data or teacher acquisition was performed by this audit.
