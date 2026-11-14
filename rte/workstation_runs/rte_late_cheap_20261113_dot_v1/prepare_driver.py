from pathlib import Path
here=Path(__file__).resolve().parent
s=(here.parent/'rte_shell_direction_dot_v1/run_shell.py').read_text()
s=s.replace('Fixed second-direction support control: repeated P8 versus |kt|=9..16.','Fixed cheap late direction: unfiltered B8 versus shell-projected S B8 S.')
s=s.replace('from angular_modes import AngularBandMode,FullAngularModes\n','')
a=s.index('class ShellModes(');b=s.index('def project_shell(',a);s=s[:a]+s[b:]
a=s.index('def qualify_shell(');b=s.index('@torch.no_grad()',a);s=s[:a]+s[b:]
s=s.replace("'repeat_P8'","'late_full8'").replace("'shell_second'","'late_shell8'")
s=s.replace("scope='Isolates the support of the second direction after classical call2. Original first P8 correction and TSA64 are unchanged. Shell is selected using paid source0/4 diagnostic; not claimed unseen-source generalization. No oracle response vectors used.'","scope='After classical call2 add exactly one cheap classical direction: B8(r), or S B8(S r). Same existing eight-stage action, with shell FFT overhead paid only by shell branch. Original P8 first correction and outer TSA64 unchanged. No oracle direction, depth sweep, training or unseen-material claim.'")
s=s.replace("HERE/'SHELL_PLAN.md'","HERE/'LATE_CHEAP_PLAN.md'")
old="  report['selection_evidence_sha256']=p.digest(p.ROOT/'rte/workstation_runs/rte_error_response_20261113_dot_v1/results/results.json')"
new=old+"\n  report['prior_audit_sha256']={name:p.digest(p.ROOT/('rte/workstation_runs/'+name+'/results/results.json')) for name in ['rte_coarse_fine_20261113_dot_v1','rte_postcoarse_budget_20261113_dot_v1','rte_shell_direction_20261113_dot_v1']}\n  report['TSA8_bridge']=json.loads((p.ROOT/'rte/workstation_runs/rte_coarse_fine_20261113_dot_v1/SMOOTHER_BRIDGE.json').read_text());check(report['TSA8_bridge']['passed'] and report['TSA8_bridge']['bitwise'],'Existing actual TSA8 bridge failed')"
assert s.count(old)==1;s=s.replace(old,new)
a=s.index('   shell=None\n');b=s.index("   rec['setup_seconds']",a);s=s[:a]+"   rec['extra_factor_seconds']=0.0\n"+s[b:]
s=s.replace('inner=[];events=[];inputs=[];pre_times.clear();ti=time.perf_counter()','inner=[];cheap=[];events=[];inputs=[];pre_times.clear();ti=time.perf_counter()')
a=s.index('    def inject(x,r):');b=s.index('    schedule=',a)
s=s[:a]+'''    def inject(x,r):
     idx=len(inputs)+1;inputs.append((x.clone(),r.clone()));ti=time.perf_counter()
     if idx==1:
      proj=f.project
      def innerA(v):counters['inner_A']+=1;return proj(high.A_op(v))
      z,st=solve_from_initial(innerA,proj(r),lambda v:proj(f.apply(proj(v))),None,tolerance=1e-4,maxiter=8)
      inner.append(dict(**st,seconds=time.perf_counter()-ti,injection_index=idx,support='P8',input_norm=float(r.norm())))
      return z
     check(idx==2 and method in ['late_full8','late_shell8'],'Only one fixed cheap late direction')
     use_shell=method=='late_shell8';tp=time.perf_counter();v=project_shell(r) if use_shell else r;project_seconds=time.perf_counter()-tp if use_shell else 0.
     tc=time.perf_counter();v=v.to(torch.complex64);z=torch.zeros_like(v);stages=0
     for stage in range(8):
      defect=v if stage==0 else v-low.A_op(z)
      z=z+base.single(defect);stages+=1
     z=z.to(torch.complex128);core_seconds=time.perf_counter()-tc
     if use_shell:
      tp=time.perf_counter();z=project_shell(z);project_seconds+=time.perf_counter()-tp
     check(stages==8 and base.depth==64,'Actual cheap/outer stages changed')
     cheap.append(dict(injection_index=idx,support='shell9to16' if use_shell else 'full',actual_stages=stages,seconds=time.perf_counter()-ti,core_seconds=core_seconds,projection_seconds=project_seconds,input_norm=float(r.norm()),direction_norm=float(z.norm())))
     return z
''' + s[b:]
s=s.replace('status=st,inner=inner,observer_events=events','status=st,inner=inner,cheap=cheap,observer_events=events')
s=s.replace("preconditioner_seconds=sum(r['preconditioner_seconds'] for r in rows),A_counts=counters","preconditioner_seconds=sum(r['preconditioner_seconds'] for r in rows),cheap_seconds=sum(v['seconds'] for r in rows for v in r['cheap']),cheap_stages_per_RHS=[sum(v['actual_stages'] for v in r['cheap']) for r in rows],total_transport_stages_per_RHS=[64*r['status']['classical_calls']+sum(v['actual_stages'] for v in r['cheap']) for r in rows],A_counts=counters")
s=s.replace("   if shell is not None:rec['shell_inverse_profiles']=shell.profiles\n",'')
s=s.replace("-rec['inner_seconds']-rec['preconditioner_seconds']","-rec['inner_seconds']-rec['preconditioner_seconds']-rec['cheap_seconds']")
s=s.replace('del low,high,base,f,shell,u,inject,pre,A','del low,high,base,f,u,inject,pre,A')
a=s.index("  baseline=run('one_shot_P8')");b=s.index("  report['complete']=True",a)
s=s[:a]+'''  baseline=run('one_shot_P8');full=run('late_full8');shell=run('late_shell8')
  forward=[full,shell];winners=[c['method'] for c in forward if c['complete_four_RHS_seconds']<baseline['complete_four_RHS_seconds'] and any(a<b for a,b in zip(c['classical_calls'],baseline['classical_calls']))]
  report['forward_cost_and_call_winners']=winners
  if winners:
   # Same fixed methods in reverse order; no tuning or source changes.
   rshell=run('late_shell8');rfull=run('late_full8');rbase=run('one_shot_P8')
   confirmed=[c['method'] for c in [rshell,rfull] if c['method'] in winners and c['complete_four_RHS_seconds']<rbase['complete_four_RHS_seconds'] and any(a<b for a,b in zip(c['classical_calls'],rbase['classical_calls']))]
   report['confirmed_two_order_methods']=confirmed;report['result']='LATE_CHEAP_TWO_ORDER_GAIN' if confirmed else 'LATE_CHEAP_FORWARD_ONLY'
  else:report['result']='LATE_CHEAP_NO_GO'
''' +s[b:]
compile(s,str(here/'run_late_cheap.py'),'exec');(here/'run_late_cheap.py').write_text(s)
print('prepared',here/'run_late_cheap.py')
