"""Render completed FWI scalar evidence; no PDE computation or training here."""
import base64, hashlib, html, json, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'ANALYSIS.json').read_text())
R=D['runtime']['benchmark_joint']
T=D['training']
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for arm,label in [('training/continuation','Continuation, 3000 updates'),('training/low_lr_control','Lower LR, 1000 updates')]:
    rows=T[arm]['scores']; axes[0,0].plot([x['step'] for x in rows],[x['work_proxy'] for x in rows],marker='o',ms=3,label=label)
axes[0,0].set(title='Weak-prefix solver work reaches a small plateau',xlabel='Additional optimizer updates',ylabel='Mean calls + 0.5 injections (64/128)')
axes[0,0].legend(fontsize=8)
for arm,label in [('alignment_retry/tsa8','TSA8 training (13 eligible media)'),('alignment_retry/tsa16','TSA16 training (6 eligible media)')]:
    rows=[x for x in D['panels'][arm] if x['split']=='development' and x['grid']==128 and x['method']=='learned']
    initial=rows[0]['squared_ratio_after_2']
    axes[0,1].plot([x['step'] for x in rows],[x['squared_ratio_after_2']/initial for x in rows],marker='o',label=label)
axes[0,1].set(title='Alignment improves its fixed residual panel',xlabel='Additional optimizer updates',ylabel='Two-step squared residual / own step-zero value')
axes[0,1].legend(fontsize=8)
for method,label,offset in [('continued_selected','Extended weak-prefix model',-.10),('aligned16_tsa16','Aligned TSA16 model',.10)]:
    rows=[next(x for x in R['comparisons'] if x['size']==n and x['neural']==method and x['classical']==('reference_tail' if n<256 else 'tsa16_zero')) for n in (64,128,256)]
    vals=[x['saving_percent'] for x in rows]
    axes[1,0].errorbar([i+offset for i in range(3)],vals,yerr=[[v-x['saving_percent_ci95'][0] for v,x in zip(vals,rows)],[x['saving_percent_ci95'][1]-v for v,x in zip(vals,rows)]],fmt='o',capsize=3,label=label)
axes[1,0].axhline(0,color='gray',lw=1);axes[1,0].axhline(5,color='gray',ls='--',lw=1)
axes[1,0].set(xticks=range(3),xticklabels=['64','128','256'],xlabel='Spatial grid',ylabel='Saving against frozen classical control (%)',title='No classical speed gate passed')
axes[1,0].legend(fontsize=8)
for method,label in [('frozen_tsa16','Frozen neural correction'),('aligned8_tsa16','Matched TSA8-trained correction'),('aligned16_tsa16','Matched TSA16-trained correction')]:
    rows=[next(x for x in R['summary'] if x['size']==n and x['method']==method) for n in (64,128,256)]
    axes[1,1].plot(range(3),[x['classical_calls'] for x in rows],marker='o',label=label)
axes[1,1].set(xticks=range(3),xticklabels=['64','128','256'],xlabel='Spatial grid',ylabel='Mean classical calls',title='Same TSA16 deployment: alignment saves work')
axes[1,1].legend(fontsize=8)
fig.suptitle('RTE training priority · completed FWI experiments · 4 October 2026',fontsize=14)
fig.savefig(HERE/'training_priority_evidence.png',dpi=180);fig.savefig(HERE/'training_priority_evidence.svg');plt.close(fig)

def summary(n,method): return next(x for x in R['summary'] if x['size']==n and x['method']==method)
def comparison(n,method): return next(x for x in R['comparisons'] if x['size']==n and x['neural']==method and x['classical']==('reference_tail' if n<256 else 'tsa16_zero'))
runtime_rows=[]
for n in (64,128,256):
    base='reference_tail' if n<256 else 'tsa16_zero'
    for method,label in [(base,'Frozen classical control'),('frozen_tsa16','Frozen neural / TSA16'),('continued_selected','Extended weak-prefix'),('aligned8_tsa16','Matched TSA8-trained / TSA16'),('aligned16_tsa16','Matched TSA16-trained / TSA16')]:
        x=summary(n,method)
        runtime_rows.append(f"| {n} | {label} | {x['seconds']*1000:.3f} | {x['setup_seconds']*1000:.3f} | {x['setup_plus_seconds']*1000:.3f} | {x['classical_calls']:.4f} |")
interval_rows=[]
for n in (64,128,256):
    for method,label in [('continued_selected','Extended weak-prefix'),('aligned16_tsa16','Aligned TSA16')]:
        x=comparison(n,method);lo,hi=x['saving_percent_ci95']
        interval_rows.append(f"| {n} | {label} | {x['saving_percent']:+.3f}% | [{lo:+.3f}, {hi:+.3f}]% |")
train_rows=[]
for arm,label in [('training/continuation','Continuation'),('training/low_lr_control','Lower LR control'),('alignment_retry/tsa8','Matched TSA8'),('alignment_retry/tsa16','Matched TSA16')]:
    x=T[arm];train_rows.append(f"| {label} | {x['updates']} | {x['unique_original_training_ids']} | {x['selected_step']} | {x['receipt']['timing']['training']:.3f} |")
text=f'''# RTE training priority: completed results and handoff

4 October 2026. All physics, training, checkpoint inspection and solver work ran on FWI. The Mac bridge rendered returned scalar evidence only.

## Decision

The earlier blanket recommendation to stop tail training was premature. A matched 1000-update experiment on TSA16 deployment residuals improved the learned direction, approximately halved its two-step development residual ratio, and reduced mean classical calls from 4.5000 to 4.3750 at all three grids. More updates on the old weak-prefix objective produced only a small solver-work improvement. Neither experiment established a speed advantage over the strongest previously frozen, measured classical controls.

All twelve benchmark candidate labels fail the predeclared gate: at least 5% mean time savings at BOTH 128 and 256, zero convergence failures, and a positive lower paired interval. The reserved seed 9626 was never generated or accessed. There is no fresh-test, independent-training-seed, OOD, continuum-error or global training-saturation conclusion.

The next priority is a bounded, tolerance-aware objective on the aligned residual distribution: optimize the expensive classical calls needed to reach the original-RHS tolerance. Hold architecture, matched data, update budget and scheduler fixed; compare against the present two-step loss. This is a hypothesis for the next study, not a demonstrated remedy. This batch is complete and no further training was launched.

## Exact workspace and preservation

SSH alias FWI authenticated as wjt on FWI-Standard-PC-i440FX-PIIX-1996. Access used existing credentials, strict host-key checking, bounded timeout and disabled forwarding.

Repository: `/data/wjt/NeuralPreconditionedBornSeries`; branch `main`; HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`.

Experiment: `/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/training_priority_20261004`.

RTE means radiative transfer equation in this repository. This study uses the steady periodic two-dimensional Fourier-discretized problem, 12 angular directions, g=0.65 and epsilon=0.1, with the existing mixed medium/source family. It is distinct from the separately paused OL_preconditioner project.

All 133 pre-existing files in the scoped preservation manifest remained byte-identical. HEAD, branch and the original nine tracked dirty files remained unchanged; no original implementation was edited. New experiment code, logs, banks and checkpoints are isolated under the experiment directory. No existing jobs were cancelled, no Git push/merge occurred, and no packages, access settings or system configuration were changed.

## Training protocol and provenance

All arms start from model weights in `resume_audit_20261002/paired_training/mixed64_128/best.pt`, selected at update 800. Optimizer, scheduler and sampling RNG are reset explicitly; this is weights-only continuation, not an exact optimizer-state resume. TailResponseNetwork has 670811 parameters, width 32, four blocks and six modes. Batch size is four, alternating grids 64/128, with AdamW weight decay 1e-5, gradient clipping at one and cosine decay to 5% of the initial learning rate. The objective is mean log squared residual ratios after 0/1/2 classical steps plus 0.05 normalized teacher-direction loss.

Training uses existing seed-3626 media; validation and timing reuse seed-4626 development media. Original accurate teacher targets are retained and reused. The alignment study creates prefix residual states, not new accurate teacher solves. Nested spatial grids share physical media and do not constitute independent validation cohorts.

| Arm | Additional updates | Unique original train IDs | Selected update | Optimizer seconds |
|---|---:|---:|---:|---:|
{chr(10).join(train_rows)}

Continuation and lower-LR control use 112 eligible physical IDs per grid, shared seed 41004 and identical first 1000 sample IDs/phases. Initial rates are 1.5e-4 and 5e-5, both with scheduler horizon 3000. Rate comparisons are valid at matched update 1000; comparing 3000 with 1000 confounds rate and budget.

The matched alignment arms use the same 56 original physical IDs eligible in both prefix banks at both grids, identical sample IDs/phases, seed 41005, initial rate 1.5e-4 and horizon 1000. Both are warm-started from the same original checkpoint. The contrast is TSA8-balanced versus TSA16-zero prefix states: prefix depth AND initialization change together. It does not isolate depth alone. Cross-study comparisons also change the eligible cohort (112 versus 56).

Full solver validation uses all 16 development media at both training grids every 250 updates. Selection is lexicographic: failures, mean classical calls plus 0.5 mean injections, then mean original-RHS residual. Fixed phase-zero panels use the first 16 eligible training IDs and all eligible development IDs: 13 for TSA8 and six for TSA16 per grid. Reference and exact-teacher diagnostics are retained. Diagnostic residual ratios are normalized to each prefix residual, not the original source; they are squared and cannot be compared directly with the 1e-5 solver tolerance.

All 6000 planned optimizer updates completed with finite logs. All observed gradients were clipped. Selected checkpoints are the final endpoints in every arm, with identical selected/final model weights confirmed by tensor-content hashes; duplicate endpoint benchmark labels measure timing variability, not model differences. Checkpoint file hashes and model hashes are retained separately.

## What the learning curves support

Weak-prefix continuation reduced the validation work proxy from 7.8750 to 7.8125 by update 1000 and remained there through 3000. Loss continued to improve, while development two-step improvement was small and six-step diagnostics were flat or slightly worse. Relative weight drift diminished across successive 1000-update windows (0.1286, 0.0727, 0.0204). This supports a task-level plateau for this objective and cohort; it does not prove adequate training for every objective or global convergence.

Matched TSA16 training reduced its own work proxy from 4.6875 to 4.5625 at update 250, then remained there through 1000. Its development two-step squared ratio improved from 0.0010048 to 0.0004743 at grid 64 and from 0.0009655 to 0.0004977 at grid 128. Teacher-direction geometry also improved. Learning continued after the discrete work plateau. The matched TSA8 arm made much smaller changes. These fixed panels have different eligible cohorts and normalization; their absolute values are not directly comparable across prefixes.

![Completed scalar evidence](training_priority_evidence.png)

Figure: upper right is normalized within each arm's own fixed panel; its panels contain different eligible media. Lower-left intervals are conditional, exploratory paired media intervals on reused development data. Lower right holds TSA16 deployment fixed.

## Final synchronized timing

The final joint sweep contains 1920 measured solves plus 640 discarded warmups; the earlier completed sweep contributes another 912 measured solves plus 304 warmups. For every method/case/grid, one warmup precedes three synchronized, shuffled repeats. Case medians are taken before averaging across 16 physical media. Timing includes reference initialization, neural inference, projection, physical solver work and the final original-RHS residual check. Physical/base setup is measured separately; the constructor prepares its classical inverse eagerly. Model loading is excluded. Setup plus warm time is an estimate, not measured cold-start latency. There was no periodic GPU telemetry or exclusive GPU lock; start/end resource snapshots were retained.

| Grid | Method | Warm solve ms | Setup ms | Setup + warm ms | Classical calls |
|---:|---|---:|---:|---:|---:|
{chr(10).join(runtime_rows)}

The classical comparison is reference_tail at grids 64/128 and tsa16_zero at grid 256, frozen from previous measurements. These are the strongest previously frozen measured controls in this project protocol, not an exhaustive comparison against all radiative-transfer preconditioners.

Positive saving means faster than that grid's frozen classical control.

| Grid | Candidate | Mean saving | Conditional 95% interval |
|---:|---|---:|---:|
{chr(10).join(interval_rows)}

At grid 256 the aligned model is 1.08% slower than classical (interval crosses zero); the extended weak-prefix model is 6.03% slower. Alignment improves the same TSA16 neural pipeline relative to frozen neural weights by 3.69%, 2.62% and 3.36% at 64/128/256, respectively. Against the matched TSA8-trained model deployed behind the same TSA16 prefix, savings are 3.78%, 3.62% and 3.01%. These exploratory intervals are unadjusted; the 128 interval crosses zero. Identical final/selected weights differ by about 1–2 ms between duplicate labels, illustrating timing noise. The deterministic reduction of mean classical calls by 0.125 is the stronger evidence that alignment changes solver behavior.

The 10000-resample paired bootstrap resamples physical media after timing repeats are reduced by median. It quantifies conditional cohort variability and does not account for checkpoint-selection reuse, training-seed variation or multiple comparisons. Independent NumPy recomputation verified all 123 published comparison intervals to maximum error 1.78e-15 percentage points. No candidate passed the gate, including the separately timed final endpoint labels.

## Accuracy, mathematical assumptions and failures

All 2832 measured solves converged at original-source relative residual below 1e-5, with zero failures and maximum residual 9.9850116e-6. The saved 48 extended-model fields have source tensors bit-identical to reconstructed original right-hand sides. Independently reevaluating their residuals reproduces the same maximum. At 64/128, 32 fields were compared with existing accurate teachers for the same original sources; maximum relative difference was 6.4454e-7. No new grid-256 teacher was acquired. Aligned fields were not saved for this extra teacher comparison, although their original-RHS solver acceptance was checked during every measured solve.

For the actual discrete phase action P, let mu be the largest eigenvalue of its Hermitian part and m=min(sigma_t-sigma_s*mu). Uniform angular weights, nonnegative scattering and positive m make periodic Fourier streaming skew-adjoint and give the exact-arithmetic estimate ||error|| <= ||residual||/m. With delta=||residual||/m, the relative solution-error bound is delta/(||computed solution||-delta) when the denominator is positive. All 48 saved cases pass the evaluated assumptions; maximum evaluated relative bound is 2.3689e-4. Ordinary FP64 evaluation is not a directed-rounding enclosure, interval certificate, continuum discretization-error bound or positivity guarantee.

Four existing gradient/physical-trace/batch-independence tests passed (3.69 seconds), and a two-update smoke run completed before full training. An initial unittest invocation collected zero tests and was replaced with pytest. The first alignment attempt saved the grid-64 bank, then failed in our new CSV writer because an ineligible row omitted a column; the retry supplies the column consistently and reuses the bank after verifying its parent hash. No aligned training had started at the failure. The failed receipt and original attempt remain preserved. A temporary Mac tool transport failure recovered; remote jobs continued. There is no remaining access blocker.

## Costs and reproducibility

FWI hardware: one RTX A6000, 49140 MiB; Python 3.12.4, PyTorch 2.5.1/CUDA 12.4. GPU jobs ran serially. Actual optimizer time totals {sum(x['receipt']['timing']['training'] for x in T.values()):.3f} seconds. Validation, fixed panels, checkpoint I/O and bank preparation add cost: internal phase-one wall time 650.226 seconds, alignment retry 454.879 seconds, original failed preparation 25.269 seconds, final timing 380.690 seconds, and saved-field accuracy audit 7.374 seconds. The earlier timing wrapper took 198.193 seconds. These phases exclude human audit and Mac rendering; they are not a single elapsed batch duration. Peak GPU reserved memory was 8.645 GiB for continuation and 4.916 GiB for alignment, below the 14 GiB guard. There are 126 retained tensor/checkpoint files totaling 3542487710 bytes, including preserved duplicated banks and saved solution fields. They remain on FWI.

Reproduction entry points under the remote experiment: `code/train_adequacy.py`, `code/train_alignment.py`, `code/benchmark_adequacy.py` and `code/audit_saved_accuracy.py`. Run from the repository root with its existing Python environment. Read `PROTOCOL.json`, `PROTOCOL_ALIGNMENT.json`, `FINAL_ENDPOINT_CONTROLS.json` and each phase's `settings.json` for exact launch settings and paths before rerunning. Fresh output directories are required; do not overwrite these results. `training/*` and `alignment_retry/*` retain sampling ledgers, fixed panels, update histories, full solver validation, receipts and checkpoint hashes. `benchmark_joint/*` retains raw repeats, case medians, summaries and comparisons. `accuracy/*` retains original-source checks. `ANALYSIS.json`, `ALL_CANDIDATE_GATES.json` and `bootstrap_verification.json` contain independent checks.

## Remaining evidence needed

Prioritize an aligned, tolerance-aware objective with a matched two-step control before increasing model size or launching longer campaigns. Include independent initializations, clipping/learning-rate diagnostics, equal budget and identical original physical sample IDs; preserve unsuccessful runs. Require the same timing gate with all setup/inference costs and original-RHS acceptance before opening a fresh test cohort. Only then evaluate held-out media, source families, scattering/anisotropy changes, angular resolution and boundaries. One warm start, one sampling seed and repeatedly reused 16 development media cannot support generalization or global saturation claims.
'''
(HERE/'RTE_Training_Priority_Results_2026_10_04.md').write_text(text)

# Small deterministic Markdown renderer, with no external assets or dependencies.
def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    return s
parts=[]; lines=text.splitlines();i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('#'):
        level=len(line)-len(line.lstrip('#'));parts.append(f'<h{level}>{inline(line[level:].strip())}</h{level}>');i+=1
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
            i+=1
        parts.append('<div class="table"><table>'+''.join('<tr>'+''.join(f'<{("th" if j==0 else "td")}>{inline(c)}</{("th" if j==0 else "td")}>' for c in row)+'</tr>' for j,row in enumerate(rows))+'</table></div>')
    elif line.startswith('!['):
        data=base64.b64encode((HERE/'training_priority_evidence.png').read_bytes()).decode();parts.append(f'<img alt="Completed experimental evidence, learning curves and paired timing" src="data:image/png;base64,{data}">');i+=1
    else:
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
            paragraph.append(lines[i]);i+=1
        parts.append('<p>'+inline(' '.join(paragraph))+'</p>')
page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RTE training priority — completed results, 4 October 2026</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#172c3b;max-width:1100px;margin:32px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:36px;color:#075b70}code{font-size:13px;overflow-wrap:anywhere}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #d5dfe3;padding:8px}th{background:#eef5f7}.table{overflow:auto}@media print{body{font-size:11px;margin:0}h2{break-after:avoid}tr,img{break-inside:avoid}}</style><body>'+''.join(parts)+'</body></html>'
(HERE/'RTE_Training_Priority_Results_2026_10_04.html').write_text(page)
print(json.dumps({'report_bytes':len(page.encode()),'markdown_sha256':hashlib.sha256(text.encode()).hexdigest(),'html_sha256':hashlib.sha256(page.encode()).hexdigest()}))
