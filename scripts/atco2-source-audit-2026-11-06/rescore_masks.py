from pathlib import Path
import json,sys,hashlib,collections,io,wave
import numpy as np
import pyarrow.parquet as pq
R=Path(__file__).absolute().parent;C=R.parent/'community-atc-eval-2026-11-06'
sys.path.insert(0,str(C))
from evaluate_community_atc import read,counts,normalize,edit_distance

test=read(C/'test.jsonl');matches=read(R/'private-match-evidence.jsonl')
train={m['test_id'] for m in matches if m['public_split']=='train' and m['test_segment_pcm_identical_at_source_offset']}
validation={m['test_id'] for m in matches if m['public_split']=='validation' and m['test_segment_pcm_identical_at_source_offset']}
jacktol={t['id'] for t in test if not t['known_jacktol_disjoint']}
assert train.isdisjoint(validation)
# Independent verification with Python's WAV reader, without the SoundFile decoding path.
lookup=collections.defaultdict(list)
for m in matches:
 if m['repo_id']=='jlvdoorn/atco2-asr':lookup[(m['public_file'],m['public_row'])].append(m)
verified=set();source_verified=set()
for file in {k[0] for k in lookup}:
 idx=0
 for batch in pq.ParquetFile(R/'public-data/jlvdoorn__atco2-asr'/file).iter_batches(batch_size=32):
  for row in batch.to_pylist():
   hits=lookup.get((file,idx),[]);idx+=1
   if not hits:continue
   with wave.open(io.BytesIO(row['audio']['bytes']),'rb') as f:
    assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2)
    raw=f.readframes(f.getnframes())
   for m in hits:
    with wave.open(m['test_source_audio'],'rb') as f:
     assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2)
     source=f.readframes(f.getnframes())
    assert raw==source
    t=next(t for t in test if t['id']==m['test_id'])
    with wave.open(t['audio_filepath'],'rb') as f:clip=f.readframes(f.getnframes())
    start=int(t['source_offset']*16000)*2;end=start+int(t['duration']*16000)*2
    assert clip==raw[start:end]
    verified.add(t['id']);source_verified.add(m['test_source_audio'])
assert verified==train|validation
masks={'exclude_jacktol_and_audited_atco2_training':jacktol|train,'exclude_jacktol_and_audited_atco2_training_and_validation':jacktol|train|validation}
private={'train_ids':sorted(train),'validation_ids':sorted(validation),'jacktol_ids':sorted(jacktol),'excluded_by_mask':{k:sorted(v) for k,v in masks.items()}}
(R/'private-exclusion-masks.json').write_text(json.dumps(private,indent=2))
summary={'matched_train_segments':len(train),'matched_validation_segments':len(validation),'training_matches_remaining_in_jacktol_disjoint':len(train-jacktol),'validation_matches_remaining_in_jacktol_disjoint':len(validation-jacktol),'independently_verified_segments':len(verified),'independently_verified_complete_recordings':len(source_verified),'private_mask_sha256':hashlib.sha256((R/'private-exclusion-masks.json').read_bytes()).hexdigest(),'views':{},'scores':{},'post_hoc':True,'interpretation':'These fixed common masks exclude only the audited sources. They are not independent blind tests and do not certify full training ancestry.'}
for label,excluded in masks.items():
 rows=[t for t in test if t['id'] not in excluded]
 summary['views'][label]={'excluded_utterances':len(excluded),'retained_utterances':len(rows),'reference_words':sum(len(normalize(t['text']).split()) for t in rows),'hours':sum(t['duration'] for t in rows)/3600,'source_recordings':len({t['source_audio_filepath'] for t in rows})}
print('MASKS',summary['views'],flush=True)
a=json.loads((C/'aggregate.json').read_text());paths={}
for k,r in a['results'].items():paths[k]=C/'baseline-predictions'/(k+'.jsonl') if k in ['qenneth','p4','n4'] else Path(r['raw_prediction_path'])
paths['our-jacktol-p2']=R.parent/'jacktol-p2-eval-2026-11-06/results/our-jacktol-p2/attempt-38096604/test/predictions.jsonl'
for k,path in paths.items():
 rows=read(path);assert len(rows)==len(test)
 for t,r in zip(test,rows):assert t['id']==r['id'] and normalize(t['text'])==normalize(r['text'])
 errors={}
 for r in rows:
  ref=normalize(r['text']).split();hyp=normalize(r['pred_text']).split();c=counts(ref,hyp);assert c[0]==edit_distance(ref,hyp)
  errors[r['id']]=(*c,len(ref))
 expected=a['results'][k]['scores']['full']['word_errors'] if k in a['results'] else 4030
 assert sum(v[0] for v in errors.values())==expected
 result={}
 for label,excluded in masks.items():
  sums=np.zeros(5,dtype=np.int64);byrec={}
  for r in rows:
   if r['id'] in excluded:continue
   c=errors[r['id']];sums+=np.array(c)
   rec=r['source_audio_filepath'];v=byrec.setdefault(rec,[0,0]);v[0]+=c[0];v[1]+=c[4]
  values=np.array(list(byrec.values()));rng=np.random.default_rng(20261106);boot=values[rng.integers(0,len(values),size=(2000,len(values)))].sum(axis=1)
  lo,hi=np.quantile(boot[:,0]/boot[:,1],[.025,.975]);e,s,d,i,w=map(int,sums)
  assert w==summary['views'][label]['reference_words']
  result[label]={'word_errors':e,'substitutions':s,'deletions':d,'insertions':i,'reference_words':w,'wer':e/w,'wer_95pct_recording_cluster_bootstrap':[float(lo),float(hi)]}
 summary['scores'][k]=result;print(k,{label:round(v['wer']*100,3) for label,v in result.items()},flush=True)
 (R/'common-mask-scores.json').write_text(json.dumps(summary,indent=2))
print('COMPLETE',flush=True)
