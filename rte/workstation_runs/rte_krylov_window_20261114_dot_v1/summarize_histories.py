"""Summarize saved native histories only, never recompute a solve."""
from pathlib import Path
import json,math
P=Path(__file__).parent
rows=[]
for f in sorted(P.glob('v2_*/eval_native_*.json')):
 obj=json.loads(f.read_text())
 if not obj.get('complete'):continue
 for r in obj['rows']:
  h=r.get('history',[]);n=len(h)-1
  if n<0:continue
  points=sorted(set([0,n//3,2*n//3,n]))
  windows=[]
  for label,s,e in [('source',0,min(8,n)),('middle',n//2,min(n,n//2+8)),('tail',max(0,n-8),n)]:
   windows.append(dict(label=label,start=s,end=e,start_residual=h[s],end_residual=h[e],ratio=h[e]/max(h[s],1e-300)))
  rows.append(dict(file=str(f),panel=r['panel'],index=r['index'],rhs=r['rhs'],repeat=r['repeat'],accepted=r['accepted'],calls=r.get('native_calls'),history_length=n,actual_restarts=max(0,(n-1)//20),restart_points=[dict(call=j,residual=h[j]) for j in range(20,n,20)],third_points=[dict(call=j,residual=h[j]) for j in points],windows=windows))
(P/'onpolicy_history_summary.json').write_text(json.dumps(dict(scope='saved native true-residual history only; not Arnoldi-input action errors; restart20 boundaries; first/warm repeats retained',rows=rows),indent=2))
print('summarized',len(rows),'existing native solves; no extra GPU work')
