from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
assert not (R/'STOP').exists(),'Campaign stopped'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,h in json.loads((R/'code-freeze.json').read_text()).items():assert sha(R/name)==h,name
c=json.loads((R/'contract.json').read_text());assert c['status']=='passed'
for name,h in c['manifest_sha256'].items():assert sha(R/'inputs'/name)==h,name
for m in c['models'].values():assert sha(Path(m['path']))==m['sha256']
print('CODE_DATA_MODEL_HASH_GATES_PASSED',flush=True)
