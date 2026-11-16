"""Read-only candidate audit. Does not create training manifests or launch jobs."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,wave
B=Path('/lustre/fsw/portfolios/llmservice/users/fciannella/ATC_ELDA/asr-pilot/atc-ablation-v1');G=B/'experiments/atco2-gold-community-ablation-v1/inputs';P=B/'releases/atco2-known-overlap-removed-v2-2026-11-13/protected-evaluation.jsonl'
def read(p):return [json.loads(l) for l in p.open()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pcm(r):
 with wave.open(r['audio_filepath'],'rb') as f:
  assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2)
  start=int(float(r.get('offset',0))*16000);n=int(float(r['duration'])*16000);f.setpos(start);b=f.readframes(n);assert len(b)==n*2
 return hashlib.sha256(b).hexdigest()
def stats(rows):return {'clips':len(rows),'minutes':sum(float(r['duration']) for r in rows)/60}
held=read(P);dev=read(G/'gold-development.jsonl');protected=held+dev
names={Path(r['audio_filepath']).stem for r in protected};paths={str(Path(r['audio_filepath']).resolve()) for r in protected};ids={r['id'] for r in protected}|{r.get('source_record_id') for r in protected if r.get('source_record_id')};dates={(r['airport'],r['recorded_date']) for r in protected if r.get('airport') and r.get('recorded_date')};groups={r['gold_group_key'] for r in protected if r.get('gold_group_key')}
with ThreadPoolExecutor(max_workers=8) as pool:
 sourcehashes=dict(zip(sorted({r['audio_filepath'] for r in protected}),pool.map(sha,sorted({r['audio_filepath'] for r in protected}))))
 pcmhashes=list(pool.map(pcm,protected))
for r,h in zip(held,pcmhashes):assert h==r['release_pcm_sha256']
hashes=set(sourcehashes.values());pcms=set(pcmhashes);heldids={r['id'] for r in held};result={};allkept=[];seen=set();allcandidates=[];filehash={}
for name,path in [('excluded_original_test',G/'gold-community-test.jsonl'),('excluded_previous_gold',G/'gold-train.jsonl')]:
 candidates=[r for r in read(path) if r['id'] not in heldids];allcandidates+=candidates;kept=[];counts=Counter();blocked=[];source_disjoint=[]
 for r in candidates:
  p=r['audio_filepath'];reasons=[]
  if r['id'] in ids or r.get('source_record_id') in ids:reasons.append('source_id')
  if Path(p).stem in names or str(Path(p).resolve()) in paths:reasons.append('source_path_or_basename')
  if p not in filehash:filehash[p]=sha(p)
  if filehash[p] in hashes:reasons.append('source_file_sha256')
  crop=pcm(r)
  if crop in pcms:reasons.append('crop_pcm_sha256')
  if not reasons:source_disjoint.append(r)
  if (r.get('airport'),r.get('recorded_date')) in dates:reasons.append('airport_date')
  if r.get('gold_group_key') in groups:reasons.append('gold_recording_group')
  if crop in seen:reasons.append('duplicate_retained_candidate_pcm')
  if reasons:counts.update(reasons);blocked.append(r)
  else:kept.append(r);allkept.append(r);seen.add(crop)
 result[name]={'candidates':stats(candidates),'source_and_exact_audio_disjoint_before_group_filter':stats(source_disjoint),'excluded':stats(blocked),'retained_under_current_group_rule':stats(kept),'exclusion_reasons':dict(counts),'source_manifest_sha256':sha(path),'retained_ids_sha256':hashlib.sha256(json.dumps(sorted(r['id'] for r in kept)).encode()).hexdigest()}
report={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'protected_eval':stats(held),'protected_development':stats(dev),'protected_eval_sha256':sha(P),'protected_development_sha256':sha(G/'gold-development.jsonl'),'pools':result,'total_candidates':stats(allcandidates),'total_retained_under_current_group_rule':stats(allkept),'retained_unique_pcm_count':len(seen),'checks':['IDs','resolved source paths and basenames','complete source-file SHA256','exact 16-kHz mono signed-int16 crop PCM SHA256','airport/date groups','Gold recording groups','retained-candidate PCM deduplication'],'training_changed':False,'limitations':['Same airport/date or Gold group is a conservative exclusion, not proof of identical audio.','Exact-audio negatives do not establish independence from transformed or unknown upstream data.','Reuse does not confer redistribution rights.','Current training remains Silver-only; Gold is held.']}
print(json.dumps(report,indent=2))
