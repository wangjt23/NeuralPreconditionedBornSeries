"""Frozen B amplitude6 with unchanged E sequential inverse; conditional AB/BA."""
import os,sys,time,json,hashlib,signal,resource,gc,subprocess
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
E=ROOT/'rte/workstation_runs/rte_angular_full_20261110_dot_v1'
B=ROOT/'rte/workstation_runs/rte_inclusion_stress_20261110_dot_v1'
sys.path[:0]=[str(E),str(B),str(ROOT),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1')]
from angular_modes import FullAngularModes
from run_full_cpu import validate_all_modes,ModeQualificationFailure,project8
from stress_inclusion_cpu import construct,medium,original_medium
from layered_inverse import LayeredInverse,background_from_material
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_hash(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
class GateStop(RuntimeError):
    def __init__(self,code,message):self.code=code;super().__init__(message)

@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    old=json.loads((B/'stress_cpu_01/settings.json').read_text());es=json.loads((E/'full_01/settings.json').read_text())
    assert digest(B/'stress_inclusion_cpu.py')==old['source_hashes']['stress_inclusion_cpu.py']=='62fed575d02f34676b916dfe1678c561b0970ad97d364e7424bb3bd75284eb74'
    assert digest(E/'angular_modes.py')==es['source_hashes']['angular_modes.py']=='d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481'
    assert digest(E/'run_full_cpu.py')==es['source_hashes']['run_full_cpu.py']
    assert {p:digest(ROOT/p) for p in es['production_hashes']}==es['production_hashes']==old['production_hashes']
    assert {p:digest(ROOT/p) for p in es['dependency_hashes']}==es['dependency_hashes']
    out=HERE/'stress_01';out.mkdir(exist_ok=False);started=time.perf_counter();current_stage=None
    s=dict(complete=False,pid=os.getpid(),affinity=[0],device='cpu',threads=1,stage_limit_seconds=180,total_limit_seconds=600,source_ids=[0,4,7,14],grid=128,L=12,g=.65,inclusion_amplitude=6.,prefix=1,tangent_cut=8,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,relative_bound_limit=1e-3,field_difference_limit=1e-4,block_target=1e-10,block_max_corrections=3,heldout9626=False,new_accurate_teachers=False,neural_training=False,orders=[['compact4','sequential_E'],['sequential_E','compact4']],expected_calls={'compact4':[4,4,4,4],'sequential_E':[2,3,2,3]},expected_inner=[6,6,6,6],source_sha256=digest(__file__),E_source_hashes=es['source_hashes'],B_source_hashes=old['source_hashes'],production_hashes=es['production_hashes'],dependency_hashes=es['dependency_hashes'],scope='Existing B amplitude3-to6 inclusion-strength perturbation with frozen E representation. Not material generalization.',admission_processes=subprocess.check_output(['ps','-eo','pid,psr,pcpu,args','--sort=-pcpu'],text=True).splitlines()[:16],stages=[],rows=[],batches=[],references=[])
    def save():
        target=out/'results.json';temp=out/'results.json.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,target)
    def expired(signum,frame):raise GateStop('STOP_WALL_OR_INTERRUPT','Own stage signal '+str(signum))
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    def stage(name):
        nonlocal current_stage
        if current_stage is not None:
            current_stage.update(status='completed',seconds=time.perf_counter()-current_stage.pop('_clock'))
        remaining=600-(time.perf_counter()-started)
        if remaining<=0:raise GateStop('STOP_TOTAL_WALL','600-second numerical budget exhausted')
        current_stage=dict(name=name,status='running',_clock=time.perf_counter());s['stages'].append(current_stage);signal.setitimer(signal.ITIMER_REAL,min(180,remaining));save()
    save();print(json.dumps(dict(event='started',pid=os.getpid(),affinity=[0],source_sha256=s['source_sha256'])),flush=True)
    try:
        stage('identity_and_ordinary_references')
        t0,s0,_=original_medium(128,1,10/3,'cpu');t3,s3,_=medium(128,1,10/3,'cpu',amplitude=3.);t6,s6,info=medium(128,1,10/3,'cpu',amplitude=6.)
        assert torch.equal(t0,t3) and torch.equal(s0,s3)
        identity=dict(baseline_total_sha256=tensor_hash(t3),baseline_scatter_sha256=tensor_hash(s3),stress_total_sha256=tensor_hash(t6),stress_scatter_sha256=tensor_hash(s6))
        assert all(identity[k]==old['material_identity'][k] for k in identity)
        backgrounds={}
        for label,t,scatter in [('amplitude3',t3,s3),('amplitude6',t6,s6)]:
            axis,tb,sb,score=background_from_material(t,scatter)
            backgrounds[label]=dict(axis=axis,total_sha256=tensor_hash(tb),scatter_sha256=tensor_hash(sb),selection_score=score)
        s['material_identity']=dict(**identity,amplitude3_byte_equal_original=True,info=info,backgrounds=backgrounds,background_arrays_equal=(backgrounds['amplitude3']['axis']==backgrounds['amplitude6']['axis'] and backgrounds['amplitude3']['total_sha256']==backgrounds['amplitude6']['total_sha256'] and backgrounds['amplitude3']['scatter_sha256']==backgrounds['amplitude6']['scatter_sha256']))
        del t0,s0,t3,s3,t6,s6,tb,sb
        bank=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
        rhs=[bank['bank']['q'][i:i+1].to(torch.complex128) for i in s['source_ids']]
        s['rhs_sha256']={str(i):tensor_hash(bank['bank']['q'][i:i+1]) for i in s['source_ids']};assert s['rhs_sha256']==old['rhs_sha256']==es['rhs_sha256'];del bank;gc.collect()
        tick=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);references={}
        for sid,b in zip(s['source_ids'],rhs):
            ti=time.perf_counter();u,st=solve_from_initial(high.A_op,b,pre,None,tolerance=1e-5,maxiter=100)
            s['references'].append(dict(source_id=sid,seconds=time.perf_counter()-ti,status=st))
            if st['failed'] or st['true_residual']>1e-5:raise GateStop('STOP_NATIVE_REFERENCE','Ordinary native reference failed')
            references[sid]=u;save()
        s['reference_research_seconds']=time.perf_counter()-tick
        del low,high,base,pre;gc.collect();save()

        def run(method,rep,position):
            tick=time.perf_counter();record=dict(method=method,repeat=rep,position=position,status='running');s['batches'].append(record)
            low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
            assert tensor_hash(low.sigma_t)==identity['stress_total_sha256'] and tensor_hash(low.sigma_s)==identity['stress_scatter_sha256']
            axis,tb,sb,score=background_from_material(low.sigma_t,low.sigma_s)
            record['background']=dict(axis=axis,total_sha256=tensor_hash(tb),scatter_sha256=tensor_hash(sb),selection_score=score)
            assert record['background']==backgrounds['amplitude6']
            record.update(physical_setup_seconds=physical,TSA_setup_seconds=tsa)
            t=time.perf_counter()
            if method=='compact4':factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64);record['logical_factor_bytes']=factor.factor_bytes
            else:
                factor=FullAngularModes(low,high);record['logical_factor_bytes']=factor.bytes
                record['mode_setup_phases']={str(k):b.setup for k,b in factor.blocks.items()};record['repair_profiles']=factor.profiles
            record['factor_setup_seconds']=time.perf_counter()-t;t=time.perf_counter()
            if method=='sequential_E':
                try:record['mode_qualifications']=validate_all_modes(factor,'angular8_inner',low,high)
                except ModeQualificationFailure as exc:
                    record['mode_qualifications']=exc.records;raise GateStop('STOP_ORIGINAL_A0_QUALIFICATION',str(exc))
            record['mode_qualification_seconds']=time.perf_counter()-t;record['setup_observed_seconds']=time.perf_counter()-tick;local=[]
            for j,(sid,b) in enumerate(zip(s['source_ids'],rhs)):
                inner=[];bn=float(b.norm());before=len(factor.profiles) if method=='sequential_E' else 0;t=time.perf_counter()
                if method=='compact4':inject=lambda x,r:factor(r).to(torch.complex128)
                else:
                    def inject(x,r):
                        ti=time.perf_counter();z,st=solve_from_initial(lambda v:project8(high.A_op(v)),project8(r),lambda v:project8(factor.apply(project8(v))),None,tolerance=1e-4,maxiter=8);inner.append(dict(**st,seconds=time.perf_counter()-ti));return z
                u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,tolerance=1e-5,maxiter=100);solve=time.perf_counter()-t;t=time.perf_counter()
                res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-references[sid]).norm()/references[sid].norm());accepted=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and 0<=bound<=1e-3 and diff<=1e-4;check=time.perf_counter()-t
                if j==0:record['first_RHS_observed_seconds']=time.perf_counter()-tick
                st.pop('trace',None);row=dict(method=method,repeat=rep,source_id=sid,solve_seconds=solve,check_seconds=check,status=st,inner=inner,accepted=accepted,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff)
                if method=='sequential_E':row['inverse_applications']=len(factor.profiles)-before
                s['rows'].append(row);local.append(row)
                if not accepted or any(i['failed'] or i['true_residual']>1e-4 for i in inner):raise GateStop('STOP_ORIGINAL_ACCEPTANCE','Original solver gate failed: '+json.dumps(row))
                if st['classical_calls']!=s['expected_calls'][method][j] or (method=='sequential_E' and [i['iterations'] for i in inner]!=[6]):raise GateStop('STOP_B_CALL_PATTERN_BRIDGE','Call pattern differs from frozen B: '+json.dumps(row))
            record.update(status='completed',complete_four_RHS_observed_seconds=time.perf_counter()-tick,online_phase_seconds=sum(r['solve_seconds']+r['check_seconds'] for r in local),calls=[r['status']['classical_calls'] for r in local],inner_calls=[i['iterations'] for r in local for i in r['inner']],process_peak_RSS_bytes=rss())
            save();print(json.dumps({k:record[k] for k in ['method','repeat','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','calls','inner_calls']}),flush=True)
            del inject,factor,pre,base,low,high,u;gc.collect();return record

        stage('forward_compact_then_E');c0=run('compact4',0,0);e0=run('sequential_E',0,1)
        if e0['complete_four_RHS_observed_seconds']>=c0['complete_four_RHS_observed_seconds']:
            s.update(complete=True,result='NO_GO_FORWARD_COMPLETE_COST');return
        s['forward_passed']=True;stage('reverse_E_then_compact');e1=run('sequential_E',1,0);c1=run('compact4',1,1)
        s.update(complete=True,result='CPU_STRESS_ABBA_PASS' if e1['complete_four_RHS_observed_seconds']<c1['complete_four_RHS_observed_seconds'] else 'UNRESOLVED_REVERSE_COST')
    except GateStop as exc:s.update(complete=True,result=exc.code,stop_reason=str(exc))
    except BaseException as exc:s['error']=repr(exc);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if current_stage is not None and '_clock' in current_stage:current_stage.update(status='completed' if s.get('result')=='CPU_STRESS_ABBA_PASS' else 'stopped',seconds=time.perf_counter()-current_stage.pop('_clock'))
        for r in s['batches']:
            if r['status']=='running':r['status']='stopped';r['reason']=s.get('stop_reason',s.get('error'))
        s.update(wall_seconds=time.perf_counter()-started,process_peak_RSS_bytes=rss(),production_hashes_after={p:digest(ROOT/p) for p in s['production_hashes']},dependency_hashes_after={p:digest(ROOT/p) for p in s['dependency_hashes']});save()
        print(json.dumps(dict(event='finished',pid=s['pid'],result=s.get('result'),wall_seconds=s['wall_seconds'])),flush=True)
if __name__=='__main__':main()
