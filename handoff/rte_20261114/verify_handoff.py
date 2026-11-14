#!/usr/bin/env python3
"""Read-only verification of the exact materialized handoff. No training/solver calls."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=list(csv.DictReader((HERE/'SHA256SUMS.tsv').open(),delimiter='\t'))
 for row in rows:
  p=(ROOT/row['path']).resolve()
  assert ROOT in p.parents and p.is_file(),row['path']
  assert p.stat().st_size==int(row['bytes']) and digest(p)==row['sha256'],row['path']
 inv=json.loads((HERE/'SOURCE_INVENTORY.json').read_text())
 for row in inv['files']:
  if row['materialized']:
   p=ROOT/row['path'];assert p.is_file() and digest(p)==row['sha256'],row['path']
 print(f"PASS: {len(rows)} materialized files, {sum(int(x['bytes']) for x in rows):,} bytes; {sum(x['materialized'] for x in inv['files'])} authoritative-source hashes match. External omissions are NOT verified by this checkout.")
if __name__=='__main__':main()
