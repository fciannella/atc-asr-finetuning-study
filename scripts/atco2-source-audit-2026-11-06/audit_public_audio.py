"""Verify public training audio against a frozen private test; keep row evidence private."""
from pathlib import Path
import json,io,hashlib,collections,sys
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
R=Path(__file__).absolute().parent
C=R.parent/'community-atc-eval-2026-11-06'
sys.path.insert(0,str(C))
from score_wer import normalize
read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
sha=lambda b:hashlib.sha256(b).hexdigest()
test=read(C/'test.jsonl')
assert len(test)==1908
# Original complete recording identity, not merely the utterance crop identity.
source={};byname=collections.defaultdict(list);byhash=collections.defaultdict(list)
for r in test:
 path=r['source_audio_filepath']
 if path not in source:
  a,sr=sf.read(path,dtype='int16',always_2d=True)
  assert sr==16000 and a.shape[1]==1
  source[path]={'pcm':a[:,0].copy(),'hash':sha(a.astype('<i2').tobytes()),'frames':len(a),'rows':[]}
  byname[Path(path).name].append(path);byhash[source[path]['hash']].append(path)
 source[path]['rows'].append(r)
receipts=json.loads((R/'download-receipts.json').read_text());evidence=[];reports=[]
for receipt in receipts:
 repo=receipt['repo_id'];rev=receipt['revision'];folder=R/'public-data'/repo.replace('/','__')
 groups={}
 for record in receipt['files']:
  file=record['path'];split='validation' if '/validation-' in file else 'train'
  g=groups.setdefault(split,{'rows':0,'hours':0,'formats':collections.Counter(),'names':set(),'hashes':set(),'whole_sources':set(),'name_sources':set(),'contained_ids':set(),'partial_ids':set(),'matched_public_rows':set(),'decode_failures':0,'rasr_eligible_rows':0,'rasr_eligible_ids':set(),'whisper_30s_ids':set(),'maximum_duration':0,'non_16khz_rows':0})
  index=0
  for batch in pq.ParquetFile(folder/file).iter_batches(batch_size=64):
   for row in batch.to_pylist():
    public_id=file+':'+str(index);index+=1;g['rows']+=1
    audio=row['audio'];name=Path(audio['path']).name if audio.get('path') else ''
    a,sr=sf.read(io.BytesIO(audio['bytes']),dtype='int16',always_2d=True)
    g['hours']+=len(a)/sr/3600;g['formats'][(sr,a.shape[1])]+=1
    duration=len(a)/sr;g['maximum_duration']=max(g['maximum_duration'],duration)
    eligible=(0.1<=duration<=18 and bool((row.get('text') or '').strip()))
    g['rasr_eligible_rows']+=int(eligible)
    g['non_16khz_rows']+=int(sr!=16000)
    if sr!=16000 or a.shape[1]!=1:continue
    pcm=a[:,0];h=sha(a.astype('<i2').tobytes());g['names'].add(name);g['hashes'].add(h)
    candidates=set(byname.get(name,[]))|set(byhash.get(h,[]))
    for path in candidates:
     s=source[path];same=s['hash']==h
     if path in byname.get(name,[]):g['name_sources'].add(path)
     if same:g['whole_sources'].add(path);g['matched_public_rows'].add(public_id)
     for t in s['rows']:
      start=int(t['source_offset']*16000);count=int(t['duration']*16000)
      exact=(start+count<=len(pcm) and np.array_equal(pcm[start:start+count],s['pcm'][start:start+count]))
      if same and not exact:raise RuntimeError('Whole match but invalid test crop '+t['id'])
      if exact:
       cropped,csr=sf.read(t['audio_filepath'],dtype='int16');assert csr==16000 and np.array_equal(cropped,pcm[start:start+count])
       g['contained_ids'].add(t['id']);g['matched_public_rows'].add(public_id)
       if eligible:g['rasr_eligible_ids'].add(t['id'])
       if start+count<=30*16000:g['whisper_30s_ids'].add(t['id'])
      if same or exact:
       evidence.append({'repo_id':repo,'revision':rev,'public_split':split,'public_file':file,'public_row':index-1,'public_audio_path':audio.get('path'),'public_pcm_sha256':h,'public_duration':duration,'public_text_nonempty':bool((row.get('text') or '').strip()),'rasr_recipe_duration_eligible':eligible,'within_whisper_first_30s':start+count<=30*16000,'test_id':t['id'],'test_source_audio':path,'test_source_offset':t['source_offset'],'test_duration':t['duration'],'whole_recording_pcm_identical':same,'test_segment_pcm_identical_at_source_offset':exact})
  print('READ',repo,split,file,'rows',index,flush=True)
 for split,g in groups.items():
  ids=g['contained_ids'];report={'repo_id':repo,'revision':rev,'split':split,'rows':g['rows'],'hours':g['hours'],'audio_formats':{str(k):v for k,v in g['formats'].items()},'filename_linked_test_sources':len(g['name_sources']),'identical_complete_test_source_recordings':len(g['whole_sources']),'matching_public_rows':len(g['matched_public_rows']),'exactly_contained_test_segments':len(ids),'contained_test_hours':sum(t['duration'] for t in test if t['id'] in ids)/3600,'contained_test_words':sum(len(normalize(t['text']).split()) for t in test if t['id'] in ids),'matched_in_existing_jacktol_disjoint':sum(t['known_jacktol_disjoint'] for t in test if t['id'] in ids),'unmatched_test_segments':len(test)-len(ids),'decode_failures':g['decode_failures'],'maximum_public_duration':g['maximum_duration'],'non_16khz_rows_not_matched':g['non_16khz_rows'],'rasr_recipe_eligible_public_rows':g['rasr_eligible_rows'],'rasr_recipe_eligible_contained_test_segments':len(g['rasr_eligible_ids']),'within_whisper_first_30s_test_segments':len(g['whisper_30s_ids'])}
  reports.append(report);print('RESULT',json.dumps(report),flush=True)
(R/'private-match-evidence.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in evidence))
(R/'audio-audit.json').write_text(json.dumps({'source_test_sha256':json.loads((C/'data-contract.json').read_text())['test']['source_sha256'],'method':'Exact decoded signed 16-bit mono PCM equality at 16kHz, full-source SHA256 identity and sample-for-sample containment of frozen test crops; candidate links via source basename or full-recording PCM hash. No transcript-only match accepted.','limitations':'Negative results do not exclude arbitrary recropping, retiming, re-encoding or other acoustic transforms of unmatched recordings. Public dataset snapshots are pinned; model authors did not pin all training dataset revisions.','datasets':reports,'download_receipts':receipts,'private_evidence_sha256':sha((R/'private-match-evidence.jsonl').read_bytes())},indent=2))
print('COMPLETE',flush=True)
