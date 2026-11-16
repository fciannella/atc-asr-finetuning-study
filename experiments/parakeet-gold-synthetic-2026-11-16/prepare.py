"""Freeze reclaimed Gold and verify the complete available P3 adaptation history."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,random,re,sys
import soundfile as sf
from score_wer import normalize
R=Path(__file__).resolve().parent;B=R.parent.parent;P=B/'experiments/parakeet-tdt-v3-jacktol-transfer-v1';G=B/'experiments/atco2-gold-community-ablation-v1';C=B/'experiments/nemotron-transcribe-p4-curriculum-2026-11-07-r2';release=B/'releases/atco2-known-overlap-removed-v2-2026-11-13/protected-evaluation.jsonl'
I=R/'inputs';I.mkdir(exist_ok=True)
def read(p):return [json.loads(l) for l in Path(p).open()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def lines(p,rows):p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
def pcm(r):
 with sf.SoundFile(r['audio_filepath']) as f:
  assert f.samplerate==16000 and f.channels==1
  n=int(float(r['duration'])*16000);start=int(float(r.get('offset',0))*16000)
  assert start>=0 and n>0 and start+n<=len(f)
  f.seek(start);a=f.read(n,dtype='int16');assert len(a)==n
 return hashlib.sha256(a.astype('<i2',copy=False).tobytes()).hexdigest()
held=read(release);dev=read(G/'inputs/gold-development.jsonl');guard=read(C/'guard-requests.jsonl');assert len(held)==1503 and len(dev)==100
# Original P3 was selected on Silver development, which is audited as ancestry too.
parent_contract=json.loads((P/'contract.json').read_text());sources={k:Path(parent_contract['dataset'][k]['path']) for k in ['atc','general','validation']}
for k,p in sources.items():assert sha(p)==parent_contract['dataset'][k]['sha256']
protected=held+dev;protected_ids={r['id'] for r in protected}|{r.get('source_record_id') for r in protected if r.get('source_record_id')};protected_paths={str(Path(r['audio_filepath']).resolve()) for r in protected};protected_names={Path(r['audio_filepath']).stem for r in protected}
with ThreadPoolExecutor(max_workers=8) as pool:
 pths=sorted({r['audio_filepath'] for r in protected});protected_filehash=dict(zip(pths,pool.map(sha,pths)));protected_pcm=set(pool.map(pcm,protected))
protected_sha=set(protected_filehash.values());cache_files=dict(protected_filehash)
# Reuse only hash-frozen audit rows, not unverified cache files.
frozen=json.loads((C/'data-freeze.json').read_text());cache={}
for name in ['silver-train.json','english-train.json']:
 assert sha(C/name)==frozen[name]
 for r in read(C/name):cache[r['id']]=(r['audio_filepath'],float(r.get('offset',0)),float(r['duration']),r['source_audio_sha256'],r['pcm_sha256']);cache_files[r['audio_filepath']]=r['source_audio_sha256']

def identity(r):
 reasons=[];p=r['audio_filepath']
 if r['id'] in protected_ids or r.get('source_record_id') in protected_ids:reasons.append('source_id')
 if p in protected_paths or Path(p).stem in protected_names:reasons.append('source_path_or_basename')
 c=cache.get(r['id'])
 if c and c[:3]==(p,float(r.get('offset',0)),float(r['duration'])):h,ch=c[3:]
 else:
  if p not in cache_files:cache_files[p]=sha(p)
  h=cache_files[p];ch=pcm(r)
 if h in protected_sha:reasons.append('source_file_sha256')
 if ch in protected_pcm:reasons.append('crop_pcm_sha256')
 return reasons,h,ch
ancestry={}
for k,p in sources.items():
 count=0;overlap=0;why=Counter();overlap_ids=[]
 for r in read(p):
  reasons,_,_=identity(r);count+=1
  if reasons:overlap+=1;why.update(reasons);overlap_ids.append(r['id'])
  if count%50000==0:print('ANCESTRY_PROGRESS',k,count,flush=True)
 ancestry[k]={'rows':count,'overlap_rows':overlap,'reasons':dict(why),'manifest_sha256':sha(p),'matching_ids_by_protected_split':{split:len(set(overlap_ids)&{r['id'] for r in rows}) for split,rows in [('common_test',[r for r in held if r['release_split']=='test']),('extension',[r for r in held if r['release_split']=='extension']),('development',dev)]}}
 print('ANCESTRY',k,ancestry[k],flush=True)
 if k!='validation':assert overlap==0,'P3 adaptation history overlaps protected audio'
 save(R/'parent-ancestry-audit.json',ancestry)
heldids={r['id'] for r in held};candidates=[r for p in [G/'inputs/gold-community-test.jsonl',G/'inputs/gold-train.jsonl'] for r in read(p) if r['id'] not in heldids];assert len(candidates)==798
retained=[];filtered=Counter();seen=set();styles=Counter()
for r in candidates:
 reasons,h,ch=identity(r);assert not reasons,(r['id'],reasons)
 if not .1<=float(r['duration'])<=30:filtered['duration_outside_0.1_30']+=1;continue
 if not normalize(r['text']) or re.search(r'\[unk\]|<unk>|IGNORE_TIME_SEGMENT_IN_SCORING',r['text'],re.I):filtered['invalid_reference']+=1;continue
 if ch in seen:filtered['duplicate_pcm']+=1;continue
 seen.add(ch);styles['raw_uppercase']+=int(any(c.isupper() for c in r['text']));styles['raw_digits']+=int(any(c.isdigit() for c in r['text']));styles['normalized_digits']+=int(any(c.isdigit() for c in normalize(r['text'])));styles['normalization_changed']+=int(normalize(r['text'])!=r['text'])
 retained.append({**r,'text':normalize(r['text']),'source_audio_sha256':h,'pcm_sha256':ch,'domain':'gold_atco2'})
random.Random(42).shuffle(retained);lines(I/'gold-train.jsonl',retained)
english=read(C/'english-train.json');english=[{**r,'domain':'general_english'} for r in english if .1<=float(r['duration'])<=30];lines(I/'english-train.jsonl',english)
# Prevent the English guard from entering replay; its original audit also excludes guard speakers.
guardpcm=set();guardfiles=set()
for r in guard:
 rr={'audio_filepath':r['audio_filepath'],'offset':r['start_sample']/16000,'duration':(r['end_sample']-r['start_sample'])/16000};guardpcm.add(pcm(rr));guardfiles.add(r['audio_sha256'])
assert all(r['pcm_sha256'] not in guardpcm and r['source_audio_sha256'] not in guardfiles for r in english)
# Train on normalized labels; retain raw development references for raw-versus-lexical diagnosis.
lines(I/'development.jsonl',[{**r,'text':normalize(r['text'])} for r in dev]);lines(I/'development-raw.jsonl',dev)
refs=json.loads((C/'references.json').read_text())
lines(I/'english-guard.jsonl',[{'id':r['id'],'audio_filepath':r['audio_filepath'],'offset':r['start_sample']/16000,'duration':(r['end_sample']-r['start_sample'])/16000,'text':normalize(refs[r['id']]['text'])} for r in guard])
for name,rows in [('common-test',[r for r in held if r['release_split']=='test']),('extension',[r for r in held if r['release_split']=='extension'])]:lines(I/(name+'.jsonl'),rows)
models={'base':P/'base-model/parakeet-tdt-0.6b-v3-no-fused.nemo','silver':P/'artifacts/full/32830103/best-validation.nemo'}
expected={'base':'1effc181f031d045e17e2ed6e3d7d22e502e3bacf637c1e023ab1626fdba6460','silver':'a5e8653b2e5306259ffd6264705ef5c401c389b57d6de7463f96989ccde38d0a'}
for name,p in models.items():assert sha(p)==expected[name]
contract={'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),'experiment':R.name,'protected_evaluation_clips':1503,'protected_development_clips':100,'protected_evaluation_sha256':sha(release),'airport_date_overlap_allowed':True,'source_and_exact_audio_overlap_allowed':False,'ancestry':ancestry,'eligible_training_arms':['p_g2'],'excluded_training_arms':{'p_g3':'Silver parent checkpoint selection used protected audio; no clean evaluation claim'},'models':{k:{'path':str(p),'sha256':expected[k]} for k,p in models.items()},'gold':{'candidate_clips':798,'retained_clips':len(retained),'minutes':sum(r['duration'] for r in retained)/60,'filtered':dict(filtered),'style':dict(styles),'unique_pcm':len(seen)},'english':{'clips':len(english),'hours':sum(r['duration'] for r in english)/3600},'source_weights':{'gold_atco2':.85,'general_english':.15},'training':{'gpus':8,'precision':'32-true','batch_per_gpu':2,'global_batch':16,'seed':42,'max_steps':2000,'lr':1e-5,'warmup_steps':100,'min_lr':1e-6,'keep_steps':[250,500,1000,1500,2000],'validation_interval':250},'selection':{'primary':'standalone lexical development WER','english_guard_max_absolute_increase':.002,'english_guard_reference':'same arm initial checkpoint','tie':'earlier step','test_used':False},'synthetic':{'planned':True,'error_analysis_source':'development only; no reference sentences copied into generation','controls':['real Gold baseline','generic ATC synthetic','development-category-targeted synthetic'],'initial_max_accepted_utterances_per_arm':1000,'training_gold_weight':.70,'training_synthetic_weight':.15,'training_english_weight':.15,'final_test_after_all_selected_models_locked':True},'limitations':['Available adaptation/selection manifests audited; full pretrained ancestry unavailable.','Shared airport/date context is allowed.','Development has 100 clips; target-category findings need uncertainty and replication.','No synthetic audio generated by this preparation.']}
contract['manifest_sha256']={p.name:sha(p) for p in I.glob('*.jsonl')};save(R/'contract.json',contract)
print('PREPARATION_PASSED',json.dumps({'gold':contract['gold'],'ancestry':ancestry}),flush=True)
