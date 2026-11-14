import json,pathlib,hashlib
r=pathlib.Path(__file__).parent
n=0
for tag in ['mac','fwi']:
 d=json.loads((r/tag/'inventory.json').read_text())
 for x in d['items']:
  if not x.get('materialized'):continue
  p=r/tag/'source'/x['path'];b=p.read_bytes()
  assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],str(p)
  if p.suffix in ['.json','.ipynb']:json.loads(b)
  if p.suffix=='.py':compile(b,str(p),'exec')
  n+=1
print('PASS',n,'materialized source hashes, JSON/notebook parse, Python syntax; no numerical tests')
