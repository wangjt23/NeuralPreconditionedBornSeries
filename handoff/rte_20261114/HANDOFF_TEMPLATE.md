# Research handoff template

Use this template for every substantive continuation. The purpose is to make a future reader able
to distinguish **proved mathematics, implemented algorithms, finite evidence, negative results,
unmaterialized ideas, and open problems** without reconstructing a chat history.

## 0. Identity and authority

- Project / track:
- Handoff date:
- Parent commit or parent archive SHA-256:
- This handoff commit:
- Authoritative source root:
- Supersedes:
- Does **not** supersede:
- Status: `proved | conditional | implemented | measured | negative-result | open | recovery-only`

State whether this is a complete runnable archive, a source-first mirror, or a documentation-only
recovery. Never call a reduced mirror “complete” unless its dependency closure is actually present.

## 1. Problem and target object

Give the literal mathematical/computational object being solved or approximated. Record boundary
conditions, normalization, coefficient class, precision model, source class, and error norm.

Write the target acceptance criterion explicitly. For example,

[
\frac{\|u-u_*\|_A}{\|u_*\|_A}\le \varepsilon,
]

or the exact operator inequality being claimed.

## 2. Claim ledger

| ID | Claim | Status | Assumptions | Proof/source | Code | Evidence | Non-implications |
|---|---|---|---|---|---|---|---|
| C1 | ... | PROVED | ... | ... | n/a | exact check | does not imply ... |
| C2 | ... | IMPLEMENTED | ... | ... | ... | tests | no uniform theorem |
| C3 | ... | OPEN | ... | ... | ... | ... | ... |

Allowed status vocabulary should be small and explicit:

- **PROVED** — complete mathematical proof under stated assumptions.
- **CONDITIONAL** — theorem depends on an unresolved hypothesis.
- **IMPLEMENTED** — source exists and was executed; no theorem implied.
- **CERTIFIED-FINITE** — exact/rigorous certificate for specified finite records.
- **MEASURED** — empirical observation only.
- **NEGATIVE** — proved or certified obstruction/counterexample.
- **OPEN** — unresolved load-bearing requirement.
- **RECOVERY-ONLY** — remembered/reported but source/evidence bytes unavailable.

## 3. Load-bearing proof chain

Write the proof dependency graph, not only a theorem list.

```mermaid
flowchart LR
  A[Assumption / identity] --> B[Local theorem]
  B --> C[Global theorem]
  C --> D[Algorithmic consequence]
  D --> E[Original-problem acceptance]
```

For every arrow, state the lemma/theorem that justifies it. Mark any arrow that is conditional,
numerical, heuristic, or still open.

## 4. Algorithm contract

Specify:

- input representation;
- exact preprocessing;
- randomized choices and seeds/policy;
- ordering/tie-breaking;
- stored representation;
- setup algorithm;
- application/solve algorithm;
- stopping rule;
- original-equation verifier;
- fallback/refusal behavior;
- finite-precision assumptions.

If a prototype first constructs an object that is later discarded, that work and memory still count.

## 5. Complexity ledger

Separate and quantify:

| Component | Work | Storage | Apply / RHS | Bit/precision cost | Status |
|---|---:|---:|---:|---:|---|
| assembly | ... | ... | — | ... | proved/measured |
| factor/reference | ... | ... | ... | ... | ... |
| certification | ... | ... | ... | ... | ... |
| fallback/refusal | ... | ... | ... | ... | ... |
| total | ... | ... | ... | ... | conditional/proved |

Do not hide dense response formation, residual evaluation, source checking, compilation, failed
attempts, refinement/rebuilds, or oracle construction when they are part of the stated contract.

## 6. Numerical evidence

For each campaign record:

- immutable input hash / generator version;
- parameter grid and seeds;
- number of complete runs;
- incomplete/refused/failed runs;
- accuracy criterion;
- whether ground truth is online or diagnostic only;
- timings included/excluded;
- thread/host/runtime information;
- retained versus peak memory;
- whether repetitions are independent;
- negative controls and ablations.

Never pool incompatible hosts or silently replace failed runs.

## 7. Artifact inventory

| Path | Role | Source of truth | SHA-256 | Required to reproduce? | Materialized? |
|---|---|---|---|---|---|
| manuscript/...tex | theorem narrative | this commit | ... | yes | yes |
| src/... | implementation | this commit | ... | yes | yes |
| tests/... | regression | this commit | ... | yes | yes |
| results/...json | evidence | immutable run | ... | yes | yes |
| inputs/parent.zip | dependency | external archive | ... | yes | no/yes |

Every omitted file must have one of: exact hash + retrieval location, explicit “derived”, explicit
“bulky raw evidence”, or explicit “unrecovered”. “Not copied” is not an integrity classification.

## 8. Reproduction commands

Provide commands for:

1. integrity/hash verification;
2. environment setup;
3. unit/exact tests;
4. a small deterministic smoke case;
5. the main experiment or an explicitly smaller replay;
6. manuscript build;
7. comparison against preserved outputs.

State which steps are expected to reproduce bytes, exact integers/strings, numerical values within
tolerance, or only qualitative behavior.

## 9. Prior art and claim boundary

List the closest methods and the exact aspects already established in prior work. Separate:

- established ingredients;
- new theorem or construction;
- implementation engineering;
- empirical advantage;
- claims **not** made.

Do not infer novelty from a failed search.

## 10. Open problems

For every open item include:

- why it is load-bearing;
- whether theoretical, implementation, numerical, or provenance;
- the exact missing statement/artifact;
- what would count as closure;
- known obstructions/counterexamples.

## 11. Next plan

Order tasks by scientific leverage, not convenience. Each task should have an acceptance gate such as
“prove uniform full-space bound”, “construct source-first runnable closure”, or “match original-energy
accuracy against native baseline with all setup charged.”

## 12. Integrity checklist

Before pushing:

- [ ] parent/base identity frozen;
- [ ] recursive source manifest verifies;
- [ ] all load-bearing TeX/source/tests/results present;
- [ ] all internal links resolve;
- [ ] JSON/YAML parses;
- [ ] hashes and byte sizes match authoritative artifacts;
- [ ] proof/status ledger agrees with manuscript;
- [ ] source-specific evidence is not called all-source;
- [ ] conditional theorem is not called unconditional;
- [ ] failed/refused/incomplete work retained;
- [ ] external dependencies pinned;
- [ ] known missing artifacts listed;
- [ ] reproduction commands tested or explicitly marked unexecuted;
- [ ] remote commit SHA recorded;
- [ ] remote CI status recorded without inventing a pass.

Use `HANDOFF_MANIFEST_TEMPLATE.json` as the machine-readable companion.
