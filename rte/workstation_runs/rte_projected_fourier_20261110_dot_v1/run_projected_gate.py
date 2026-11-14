"""Locked CPU projected-Fourier action bridge and conditional three-way cost gate."""
import os, sys, time, json, hashlib, signal, resource, gc, subprocess
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
ROOT = Path('/data/wjt/NeuralPreconditionedBornSeries')
E = ROOT / 'rte/workstation_runs/rte_angular_full_20261110_dot_v1'
B = ROOT / 'rte/workstation_runs/rte_inclusion_stress_20261110_dot_v1'
sys.path[:0] = [str(E), str(ROOT), str(ROOT/'rte/workstation_runs/rte_recycling_20261110_dot_v1'), str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'), str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1')]
from angular_modes import FullAngularModes
from run_full_cpu import construct, project8, validate_all_modes, ModeQualificationFailure
from layered_inverse import LayeredInverse, background_from_material
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres
from projected_operator import ProjectedOperator, MODES, relative, tensor_hash

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
def admission():
    return subprocess.check_output(['ps', '-eo', 'pid,psr,pcpu,args', '--sort=-pcpu'], text=True).splitlines()[:20]

class GateStop(RuntimeError):
    def __init__(self, code, message): self.code = code; super().__init__(message)

def locked_probes():
    gen = torch.Generator().manual_seed(2026111004)
    old = json.loads((E/'full_01/mode_qualifications.json').read_text())
    expected = {r['k']: r['RHS_sha256'] for row in old for r in row['records']}
    vectors = []
    for k in MODES:
        q = torch.complex(torch.randn(1536, 1, dtype=torch.float64, generator=gen), torch.randn(1536, 1, dtype=torch.float64, generator=gen))
        if tensor_hash(q) != expected[k]: raise GateStop('STOP_PROBE_IDENTITY', str(k))
        vectors.append(q.reshape(1, 12, 128))
    dense = torch.stack(vectors, dim=-1)
    left = torch.zeros_like(dense); right = torch.zeros_like(dense)
    signs = (1 - 2 * (torch.arange(128) % 2)).to(torch.complex128)
    left[0, 0, :, 0] = signs; right[0, 3, :, -1] = signs
    zero = torch.zeros_like(dense)
    return [('locked_all_modes', dense), ('minus8_normal_nyquist_angle0', left), ('plus8_normal_nyquist_angle3', right), ('zero', zero), ('tiny', 1e-200*dense), ('mixed_columns', torch.cat([zero, dense, 1e-200*dense], dim=0))]

def qualify(adapter, factor, low, high, record, save):
    tick = time.perf_counter(); record['action_inverse_probes'] = []
    for label, q in locked_probes():
        rec = dict(name=label, input_sha256=tensor_hash(q)); record['action_inverse_probes'].append(rec)
        v = adapter.lift(q)
        actual_full = high.A_op(v)
        expected = adapter.restrict(actual_full)
        compressed = adapter.action(q)
        lifted = adapter.lift(compressed)
        expected_full = project8(actual_full)
        ad = relative(compressed-expected, expected); fd = relative(lifted-expected_full, expected_full)
        rec.update(action_difference=ad.tolist(), lifted_action_difference=fd.tolist(), action_output_sha256=tensor_hash(compressed), original_projected_output_sha256=tensor_hash(expected), input_amax=float(q.abs().max()), output_amax=float(compressed.abs().max()))
        if not bool(torch.isfinite(compressed).all()) or not bool((ad<=1e-10).all()) or not bool((fd<=1e-10).all()):
            save(); raise GateStop('STOP_PROJECTED_ACTION_BRIDGE', str(rec))
        z = adapter.inverse(q)
        # The sealed full-grid E implementation is batch1. Compare each column
        # separately; compressed mode inversions still exercise mixed columns.
        old = torch.cat([project8(factor.apply(project8(v[j:j+1]))) for j in range(len(v))])
        old_z = adapter.restrict(old)
        field_diff = relative(z-old_z, old_z)
        old_image = project8(high.A_op(old))
        new_image = project8(high.A_op(adapter.lift(z)))
        image_diff = relative(new_image-old_image, old_image)
        rec.update(inverse_field_difference=field_diff.tolist(), inverse_original_A_image_difference=image_diff.tolist(), inverse_output_sha256=tensor_hash(z), inverse_original_output_sha256=tensor_hash(old_z))
        zero_cols = ~(q.reshape(len(q), -1) != 0).any(dim=1)
        if bool(zero_cols.any()): rec['strict_zero_solution'] = bool((z[zero_cols] == 0).all())
        rec['tiny_preserved'] = bool((z.reshape(len(z),-1).abs().amax(dim=1)[~zero_cols] > 0).all())
        if not bool(torch.isfinite(z).all()) or not bool((field_diff<=1e-10).all()) or not bool((image_diff<=1e-10).all()) or not rec.get('strict_zero_solution', True) or not rec['tiny_preserved']:
            save(); raise GateStop('STOP_PROJECTED_INVERSE_BRIDGE', str(rec))
    record['probe_seconds'] = time.perf_counter()-tick
    record['probe_inverse_profiles'] = adapter.inverse_profiles.copy()
    record['probe_full_E_inverse_profiles'] = factor.profiles.copy()
    save()

@torch.no_grad()
def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert set(os.sched_getaffinity(0)) == {0} and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    es = json.loads((E/'full_01/settings.json').read_text()); bs = json.loads((B/'stress_cpu_01/settings.json').read_text())
    assert digest(HERE/'FROZEN_PLAN.md') == '19b95306d355f2fae8f5e46d038d207f49fa49d53f989ac6a8991890cf1ea902'
    for p,h in es['source_hashes'].items(): assert digest(E/p)==h
    for p,h in {**es['production_hashes'],**es['dependency_hashes']}.items(): assert digest(ROOT/p)==h
    assert digest(ROOT/'rte/npbs_unet_ablation.py') == '4da8fb3a2998a02b45d508f7e8387b12c99849cfc06ff8363327b667aa17d5f2'
    out=HERE/'gate_01';out.mkdir(exist_ok=False); started=time.perf_counter(); current=None
    s=dict(complete=False,pid=os.getpid(),affinity=[0],device='cpu',threads=1,stage_limit_seconds=180,total_limit_seconds=600,source_ids=[0,4,7,14],grid=128,L=12,g=.65,amplitude=3,prefix=1,modes=MODES,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,relative_bound_limit=1e-3,field_difference_limit=1e-4,block_target=1e-10,block_max_corrections=3,action_bridge_target=1e-10,heldout9626=False,new_accurate_teachers=False,neural_training=False,orders=[['compact4','full_E','reduced_E'],['reduced_E','full_E','compact4']],expected_calls={'compact4':[4,3,4,3],'full_E':[2,2,2,2],'reduced_E':[2,2,2,2]},expected_inner=[5,5,5,4],own_sources={p.name:digest(p) for p in HERE.glob('*.py')},frozen_plan_sha256=digest(HERE/'FROZEN_PLAN.md'),E_source_hashes=es['source_hashes'],production_hashes=es['production_hashes'],dependency_hashes=es['dependency_hashes'],shared_host='CPU0 only. Schur may run a <=120-second CPU16-31 variance diagnostic; no whole-host exclusivity assumed.',stages=[],rows=[],batches=[],references=[],qualification={})
    def save():
        target=out/'results.json'; temp=out/'results.json.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,target)
    def expired(signum,frame): raise GateStop('STOP_WALL_OR_INTERRUPT','Owned stage signal '+str(signum))
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    def stage(name):
        nonlocal current
        if current is not None: current.update(status='completed',seconds=time.perf_counter()-current.pop('_clock'))
        remaining=600-(time.perf_counter()-started)
        if remaining<=0:raise GateStop('STOP_TOTAL_WALL','600 seconds exhausted')
        current=dict(name=name,status='running',_clock=time.perf_counter(),admission=admission());s['stages'].append(current);signal.setitimer(signal.ITIMER_REAL,min(180,remaining));save()
    def modecheck(factor,low,high,record):
        ti=time.perf_counter()
        try:record['mode_qualifications']=validate_all_modes(factor,'angular8_inner',low,high)
        except ModeQualificationFailure as exc:
            record['mode_qualifications']=exc.records;raise GateStop('STOP_ORIGINAL_A0',str(exc))
        record['mode_qualification_seconds']=time.perf_counter()-ti
    print(json.dumps(dict(event='started',pid=os.getpid(),own_sources=s['own_sources'])),flush=True);save()
    try:
        stage('identity_and_qualification')
        bank=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
        rhs=[bank['bank']['q'][i:i+1].to(torch.complex128) for i in s['source_ids']]
        s['rhs_sha256']={str(i):tensor_hash(bank['bank']['q'][i:i+1]) for i in s['source_ids']};assert s['rhs_sha256']==es['rhs_sha256'];del bank;gc.collect()
        low,high,base,margin,physical,tsa=construct(rhs[0]);del base
        identity=dict(total_sha256=tensor_hash(low.sigma_t),scatter_sha256=tensor_hash(low.sigma_s))
        assert identity['total_sha256']==bs['material_identity']['baseline_total_sha256'] and identity['scatter_sha256']==bs['material_identity']['baseline_scatter_sha256']
        axis,tb,sb,score=background_from_material(low.sigma_t,low.sigma_s);assert axis==0
        bg=dict(axis=axis,total_sha256=tensor_hash(tb),scatter_sha256=tensor_hash(sb),score=score)
        s['material_identity']=identity;s['background_identity']=bg
        t=time.perf_counter();factor=FullAngularModes(low,high);s['qualification']['factor_seconds']=time.perf_counter()-t
        adapter=ProjectedOperator(high,factor);s['qualification'].update(physical_seconds=physical,unused_TSA_seconds=tsa,coefficient_seconds=adapter.setup_seconds,coefficient_hashes=adapter.fingerprints)
        modecheck(factor,low,high,s['qualification']);qualify(adapter,factor,low,high,s['qualification'],save)
        s['qualification']['passed']=True;save();print(json.dumps(dict(event='qualification_passed',seconds=time.perf_counter()-started)),flush=True)
        del adapter,factor,low,high,tb,sb;gc.collect()
        stage('references_and_forward_triplet')
        t=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);refs={}
        for sid,b in zip(s['source_ids'],rhs):
            ti=time.perf_counter();u,st=solve_from_initial(high.A_op,b,pre,None,tolerance=1e-5,maxiter=100);s['references'].append(dict(source_id=sid,seconds=time.perf_counter()-ti,status=st))
            if st['failed']:raise GateStop('STOP_REFERENCE','Ordinary reference failed')
            refs[sid]=u
        s['reference_seconds']=time.perf_counter()-t;del low,high,base,pre;gc.collect();save()

        def run(method,rep,position,setup_cutoff=None):
            entry_admission=admission();tick=time.perf_counter();rec=dict(method=method,repeat=rep,position=position,status='running',admission=entry_admission);s['batches'].append(rec)
            low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
            assert tensor_hash(low.sigma_t)==identity['total_sha256'] and tensor_hash(low.sigma_s)==identity['scatter_sha256']
            ax,tb,sb,score=background_from_material(low.sigma_t,low.sigma_s)
            rec['background']=dict(axis=ax,total_sha256=tensor_hash(tb),scatter_sha256=tensor_hash(sb),score=score);assert rec['background']==bg
            rec.update(physical_seconds=physical,TSA_seconds=tsa);ti=time.perf_counter()
            factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64) if method=='compact4' else FullAngularModes(low,high)
            rec['factor_seconds']=time.perf_counter()-ti;rec['factor_bytes']=factor.factor_bytes if method=='compact4' else factor.bytes
            adapter=None
            def cost_check():
                rec['setup_lower_bound']=time.perf_counter()-tick
                if setup_cutoff is not None and rec['setup_lower_bound']>=setup_cutoff:raise GateStop('NO_GO_SETUP_LOWER_BOUND',str(rec['setup_lower_bound']))
            if method!='compact4':
                rec['mode_setup_phases']={str(k):b.setup for k,b in factor.blocks.items()};cost_check();modecheck(factor,low,high,rec)
            if method=='reduced_E':
                cost_check();adapter=ProjectedOperator(high,factor);rec.update(coefficient_seconds=adapter.setup_seconds,coefficient_bytes=adapter.bytes,coefficient_hashes=adapter.fingerprints)
                qualify(adapter,factor,low,high,rec,save);cost_check();adapter.inverse_profiles=[];adapter.action_calls=0
            if method!='compact4':factor.profiles=[]
            rec['setup_observed_seconds']=time.perf_counter()-tick;local=[];counts=dict(outer_A=0,inner_A=0,full_injection_A=0,final_A=0)
            def outerA(v):counts['outer_A']+=1;return high.A_op(v)
            def innerA(v):counts['inner_A']+=1;return project8(high.A_op(v))
            for j,(sid,b) in enumerate(zip(s['source_ids'],rhs)):
                inner=[];bn=float(b.norm());ti=time.perf_counter()
                if method=='compact4':inject=lambda x,r:factor(r).to(torch.complex128)
                elif method=='full_E':
                    def inject(x,r):
                        t=time.perf_counter();z,st=solve_from_initial(innerA,project8(r),lambda v:project8(factor.apply(project8(v))),None,tolerance=1e-4,maxiter=8);inner.append(dict(**st,seconds=time.perf_counter()-t));return z
                else:
                    def inject(x,r):
                        t=time.perf_counter();rr=adapter.restrict(r);z,st=solve_from_initial(adapter.action,rr,adapter.inverse,None,tolerance=1e-4,maxiter=8);field=adapter.lift(z)
                        counts['full_injection_A']+=1;pr=project8(r);check=project8(r-high.A_op(field));actual=float(relative(check,pr)[0]);st['independent_full_projected_residual']=actual
                        inner.append(dict(**st,seconds=time.perf_counter()-t))
                        if not actual<=1e-4:raise GateStop('STOP_FULL_PROJECTED_INNER_RESIDUAL',str(actual))
                        return field
                u,st=augmented_fgmres(outerA,b,pre,None,inject=inject,inject_after=1,tolerance=1e-5,maxiter=100);solve=time.perf_counter()-ti;ti=time.perf_counter()
                counts['final_A']+=1;res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-refs[sid]).norm()/refs[sid].norm())
                accepted=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and 0<=bound<=1e-3 and diff<=1e-4;checktime=time.perf_counter()-ti
                if j==0:rec['first_RHS_observed_seconds']=time.perf_counter()-tick
                st.pop('trace',None);row=dict(method=method,repeat=rep,source_id=sid,solve_seconds=solve,check_seconds=checktime,status=st,inner=inner,accepted=accepted,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff);s['rows'].append(row);local.append(row)
                if not accepted or any(i['failed'] or i['true_residual']>1e-4 for i in inner):raise GateStop('STOP_ORIGINAL_ACCEPTANCE',str(row))
                if st['classical_calls']!=s['expected_calls'][method][j] or (method!='compact4' and [i['iterations'] for i in inner]!=[s['expected_inner'][j]]):raise GateStop('STOP_CALL_PATTERN_BRIDGE',str(row))
            rec.update(status='completed',complete_four_RHS_observed_seconds=time.perf_counter()-tick,online_phase_seconds=sum(r['solve_seconds']+r['check_seconds'] for r in local),inner_phase_seconds=sum(i['seconds'] for r in local for i in r['inner']),calls=[r['status']['classical_calls'] for r in local],inner_calls=[i['iterations'] for r in local for i in r['inner']],A_call_counts=counts,process_peak_RSS_bytes=rss())
            if method!='compact4':rec['runtime_inverse_profiles']=adapter.inverse_profiles if adapter else factor.profiles
            if adapter:rec['A_call_counts']['compressed_A']=adapter.action_calls
            save();print(json.dumps({k:rec[k] for k in ['method','repeat','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','inner_phase_seconds','calls','inner_calls']}),flush=True)
            del inject,outerA,innerA,adapter,factor,low,high,base,pre,u;gc.collect();return rec

        c0=run('compact4',0,0);e0=run('full_E',0,1);cutoff=min(c0['complete_four_RHS_observed_seconds'],e0['complete_four_RHS_observed_seconds']);r0=run('reduced_E',0,2,cutoff)
        if r0['complete_four_RHS_observed_seconds']>=cutoff:s.update(complete=True,result='NO_GO_FORWARD_COMPLETE_COST');return
        s['forward_passed']=True;stage('reverse_triplet');r1=run('reduced_E',1,0,cutoff);e1=run('full_E',1,1);c1=run('compact4',1,2)
        s.update(complete=True,result='PROJECTED_FOURIER_TWO_ORDERS_PASS' if r1['complete_four_RHS_observed_seconds']<min(e1['complete_four_RHS_observed_seconds'],c1['complete_four_RHS_observed_seconds']) else 'UNRESOLVED_REVERSE_COST')
    except GateStop as exc:s.update(complete=True,result=exc.code,stop_reason=str(exc))
    except BaseException as exc:s.update(error=repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if current and '_clock' in current:current.update(status='completed' if s.get('result')=='PROJECTED_FOURIER_TWO_ORDERS_PASS' else 'stopped',seconds=time.perf_counter()-current.pop('_clock'))
        for rec in s['batches']:
            if rec['status']=='running':rec.update(status='stopped',reason=s.get('stop_reason',s.get('error')))
        s.update(wall_seconds=time.perf_counter()-started,process_peak_RSS_bytes=rss(),production_hashes_after={p:digest(ROOT/p) for p in s['production_hashes']},dependency_hashes_after={p:digest(ROOT/p) for p in s['dependency_hashes']});save();print(json.dumps(dict(event='finished',pid=s['pid'],result=s.get('result'),error=s.get('error'),wall_seconds=s['wall_seconds'])),flush=True)

if __name__=='__main__':main()
