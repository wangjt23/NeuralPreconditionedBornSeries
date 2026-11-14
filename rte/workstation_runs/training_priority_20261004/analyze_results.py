"""Independent scalar receipt audit; no training, tensors or PDE solves."""
import collections,csv,hashlib,json,math,pathlib,statistics

ROOT=pathlib.Path(__file__).resolve().parents[2]
DATA=ROOT/'receipts/training_priority_20261004_execution'
OUT=pathlib.Path(__file__).resolve().parent
def rows(path):
 with path.open() as f:return list(csv.DictReader(f))
def read(path):return json.loads(path.read_text())
def mean(group,key):return statistics.fmean(float(r[key]) for r in group)
results={'scope':'completed fixed-checkpoint, reused-development experiment; no fresh holdout','training':{},'panels':{},'runtime':{},'checks':[]}
arms=[('training','continuation',3000),('training','low_lr_control',1000),('alignment_retry','tsa8',1000),('alignment_retry','tsa16',1000)]
sampling={}
for family,arm,updates in arms:
 p=DATA/family/arm;h=rows(p/'history.csv');v=rows(p/'validation.csv');sample=rows(p/'sampling_consumed.csv');receipt=read(p/'receipt.json')
 assert receipt['complete'] and receipt['updates']==updates
 assert [int(r['step']) for r in h]==list(range(1,updates+1)) and len(sample)==4*updates
 assert all(math.isfinite(float(r[k])) for r in h for k in ['loss','teacher_loss','grad','lr','terminal_squared_ratio'])
 scores=[]
 for step in sorted({int(r['step']) for r in v}):
  group=[r for r in v if int(r['step'])==step]
  assert len(group)==32 and {r['grid'] for r in group}=={'64','128'}
  for grid in ['64','128']:assert sorted(int(r['index']) for r in group if r['grid']==grid)==list(range(16))
  scores.append({'step':step,'failures':sum(int(r['failed']) for r in group),'work_proxy':statistics.fmean(float(r['classical_calls'])+.5*float(r['injection_calls']) for r in group),'true_residual':mean(group,'true_residual')})
 selected=min(scores,key=lambda r:(r['failures'],r['work_proxy'],r['true_residual']))
 assert selected['step']==receipt['selected_step']
 assert all(int(r['failed'])==int(float(r['true_residual'])>1e-5) for r in v)
 counts=collections.Counter(r['original_parent_index'] for r in sample)
 windows=[]
 for first,last in [(1,250),(max(1,updates-249),updates)]:
  group=[r for r in h if first<=int(r['step'])<=last]
  windows.append({'first':first,'last':last,**{k:mean(group,k) for k in ['loss','teacher_loss','injection_squared_ratio','terminal_squared_ratio','grad','lr']}})
 key=f'{family}/{arm}'
 results['training'][key]={'updates':updates,'presentations':len(sample),'unique_original_training_ids':len(counts),'sample_min':min(counts.values()),'sample_max':max(counts.values()),
  'gradient_clip_fraction':sum(float(r['grad'])>1 for r in h)/len(h),'selected_step':selected['step'],'scores':scores,'windows':windows,'receipt':receipt}
 sampling[key]=sample
 panel=rows(p/'fixed_panel.csv');summary=[]
 groups=collections.defaultdict(list)
 for r in panel:groups[(int(r['step']),r['split'],r['grid'],r['method'])].append(r)
 identities={}
 for (step,split,grid,method),group in sorted(groups.items()):
  if method=='learned':
   ids=tuple(sorted(int(r['original_parent_index']) for r in group));idx=(split,grid)
   if idx in identities:assert identities[idx]==ids
   identities[idx]=ids
  summary.append({'step':step,'split':split,'grid':int(grid),'method':method,'count':len(group),**{k:mean(group,k) for k in ['objective','teacher_direction_loss','optimally_scaled_field_error','projected_image_cosine','squared_ratio_after_0','squared_ratio_after_2','squared_ratio_after_4','squared_ratio_after_6']}})
 results['panels'][key]=summary
assert sampling['training/continuation'][:4000]==sampling['training/low_lr_control']
assert sampling['alignment_retry/tsa8']==sampling['alignment_retry/tsa16']
results['checks'].append('All four optimizer budgets, exact shared sampling, fixed panel identities, finite logs and checkpoint selections independently verified.')
for family in ['benchmark','benchmark_joint']:
 settings=read(DATA/family/'settings.json');assert settings['complete']
 all_summary=[];measured=0;failures=0;maxres=0.;recomputed={}
 for grid in [64,128,256]:
  raw=rows(DATA/family/f'runtime_n{grid}.csv');published=rows(DATA/family/f'case_medians_n{grid}.csv')
  groups=collections.defaultdict(list)
  for r in raw:
   assert int(r['failed'])==int(float(r['true_residual'])>1e-5)
   assert math.isfinite(float(r['seconds'])) and float(r['seconds'])>0
   groups[(r['method'],int(r['index']))].append(r)
   maxres=max(maxres,float(r['true_residual']));failures+=int(r['failed'])
  cases={}
  for (method,index),group in groups.items():
   assert len(group)==3 and sorted(int(r['repeat']) for r in group)==[0,1,2]
   record={'method':method,'index':index,'failed':max(int(r['failed']) for r in group),'true_residual':max(float(r['true_residual']) for r in group)}
   record.update({k:statistics.median(float(r[k]) for r in group) for k in ['seconds','setup_seconds','setup_plus_seconds','iterations','classical_calls','injection_calls']})
   cases[(method,index)]=record
  assert len(cases)==len(published)
  for r in published:
   actual=cases[(r['method'],int(r['index']))]
   for k in ['seconds','classical_calls','iterations','failed']:assert math.isclose(float(r[k]),actual[k],rel_tol=1e-12,abs_tol=1e-12)
  for method in sorted({k[0] for k in cases}):
   group=[r for (m,i),r in cases.items() if m==method];assert len(group)==16
   all_summary.append({'size':grid,'method':method,'independent_media':16,**{k:mean(group,k) for k in ['seconds','setup_seconds','setup_plus_seconds','iterations','classical_calls','injection_calls']},'failures':sum(r['failed'] for r in group)})
  recomputed[grid]=cases;measured+=len(raw)
 assert measured==settings['measured_solves']
 comparisons=read(DATA/family/'comparisons.json')
 for c in comparisons:
  cc=recomputed[c['size']];left=[cc[(c['classical'],i)]['seconds'] for i in range(16)];right=[cc[(c['neural'],i)]['seconds'] for i in range(16)]
  saving=100*(1-statistics.fmean(right)/statistics.fmean(left))
  assert math.isclose(saving,c['saving_percent'],abs_tol=1e-10)
 results['runtime'][family]={'measured_solves':measured,'warmup_solves':measured//3,'failures':failures,'max_true_residual':maxres,'summary':all_summary,'comparisons':comparisons,'decision':read(DATA/family/'decision.json')}
results['checks'].append('Both timing sweeps: all repeat coverage, failures, case medians, mean savings and solve counts independently verified.')
collection=read(ROOT/'receipts/training_priority_20261004_collection.json')
assert collection['preservation']['all_unchanged']
results['checkpoint_metadata']=collection['checkpoint_metadata'];results['checkpoint_drifts']=collection['checkpoint_drifts']
results['preservation']=collection['preservation'];results['environment']=collection['environment'];results['accuracy']=read(DATA/'accuracy/summary.json')
assert results['accuracy']['complete'] and results['accuracy']['cases']==48 and results['accuracy']['failures']==0
assert results['checkpoint_metadata']['training/continuation/best.pt']['model_sha256']==results['checkpoint_metadata']['training/continuation/last.pt']['model_sha256']
assert results['checkpoint_metadata']['alignment_retry/tsa16/best.pt']['model_sha256']==results['checkpoint_metadata']['alignment_retry/tsa16/last.pt']['model_sha256']
results['checks'].append('Selected/final endpoint weight equality verified; duplicate timing differences are measurement variability, not model differences.')
results['scalar_input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()}
(OUT/'ANALYSIS.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps({'checks':'passed','updates':sum(x['updates'] for x in results['training'].values()),'measured_solves':sum(x['measured_solves'] for x in results['runtime'].values()),'accuracy_cases':48,'source_files_preserved':results['preservation']['existing_source_count']}))
