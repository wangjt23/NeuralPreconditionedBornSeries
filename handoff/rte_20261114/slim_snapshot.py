from pathlib import Path
import shutil,subprocess,json,hashlib,csv,time,ast
S=Path('/data1/wjt/rte_handoff_20261114');D=Path('/data1/wjt/rte_handoff_20261114_slim')
if D.exists():raise SystemExit('already exists; inspect before retry')
subprocess.run(['git','clone','--no-hardlinks',str(S),str(D)],check=True)
subprocess.run(['git','checkout','-b','handoff/rte-20261114-slim-local'],cwd=D,check=True)
subprocess.run(['git','remote','set-url','origin','git@github.com:wangjt23/NeuralPreconditionedBornSeries.git'],cwd=D,check=True)
H=D/'handoff/rte_20261114';H.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
# Remove binary data only in this disposable isolated clone, not the original research tree.
removed=[]
for p in list(D.rglob('*')):
 if p.is_file() and '.git' not in p.parts and p.suffix.lower() in {'.pt','.npz','.npy','.png','.jpg','.zip','.pdf','.ipynb','.sqlite3','.so','.html'}:
  removed.append(str(p.relative_to(D)));p.unlink()
copy_ext={'.py','.md','.tex','.sh','.toml','.yaml','.yml','.ini','.cfg','.txt','.log','.pre_retry'}
selected=[];omitted=[]
for p in sorted(S.rglob('*')):
 if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
 rel=p.relative_to(S)
 if rel.parts[:2]==('handoff','rte_20261114') and p.name in {'MANIFEST.json','SHA256SUMS.tsv','SOURCE_INVENTORY.json','BINARY_DEPENDENCIES.md','INTEGRITY_AUDIT.json'}:continue
 keep=p.suffix in copy_ext or (p.suffix in {'.json','.csv','.tsv','.jsonl','.svg'} and p.stat().st_size<=64*1024)
 if p.name.startswith('._'):keep=False
 if keep:
  q=D/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);selected.append(str(rel))
 else:omitted.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p),classification='large raw table/log or binary/figure; external to lightweight source handoff',snapshot_source=str(p)))
inv=json.loads((S/'handoff/rte_20261114/SOURCE_INVENTORY.json').read_text())
for r in inv['files']:
 r['materialized']=str(r['path']) in selected
 if not r['materialized']:r['classification']='external raw data/checkpoint/large table or omitted ancillary artifact; exact original source preserved'
inv['representation']='lightweight-source-first; no tensors/checkpoints; raw tables above64KiB external'
inv['slim_created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
(H/'SOURCE_INVENTORY.json').write_text(json.dumps(inv,separators=(',',':'))+'\n')
(H/'SLIM_SELECTION.json').write_text(json.dumps(dict(rule='all source/docs/config; JSON/CSV/TSV/JSONL/SVG <=64KiB; no binary tensors/checkpoints/images/archives/PDF/notebooks; required manifests exempt',selected=selected,omitted_from_prepared_snapshot=omitted,removed_base_binary=removed),separators=(',',':'))+'\n')
print(json.dumps(dict(selected=len(selected),omitted=len(omitted),total_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file() and '.git' not in p.parts))))
