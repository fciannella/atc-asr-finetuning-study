"""Check Gold source recordings against Jacktol, including non-Gold transmissions."""
from pathlib import Path
import collections, hashlib, json, os, wave
import numpy as np
import soundfile as sf

R=Path(__file__).resolve().parent
B=R.parent.parent
G=B/'experiments/atco2-gold-community-ablation-v1/inputs/gold-train.jsonl'
J=B/'evaluation/jacktol-atc-asr-075e736/release-v1'
read=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
sha=lambda b:hashlib.sha256(b).hexdigest()
gold=read(G)
def pcm(path):
    a,sr=sf.read(path,dtype='int16',always_2d=True)
    assert sr==16000 and a.shape[1]==1
    data=a.astype('<i2').tobytes()
    with wave.open(str(path),'rb') as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000)
        assert data==w.readframes(w.getnframes())
    return data
sources={p:pcm(p) for p in sorted({r['audio_filepath'] for r in gold})}
evidence=[]
seen=0
for split in ('train','validation','test'):
    for row in read(J/'manifests'/f'{split}.jsonl'):
        path=Path(row['audio_filepath'])
        assert sha(path.read_bytes())==row['audio_sha256']
        public=pcm(path)
        if len(public)>=16000 and any(public):
            for source,data in sources.items():
                at=data.find(public)
                while at>=0 and at%2: at=data.find(public,at+1)
                if at>=0:
                    evidence.append(dict(method='exact_public_clip_in_gold_source',gold_source=source,
                        jacktol_id=row['id'],jacktol_split=split,source_start_frame=at//2,match_frames=len(public)//2))
        seen+=1
        if seen%1000==0:print('SOURCE_SCAN',seen,'matches',len(evidence),flush=True)
# Preserve existing transformed-audio links only after proving identical complete
# Gold/Silver source PCM; a shared filename alone is not accepted.
prior_path=B/'datasets/fullhq-jacktol-disjoint-v1/confirmed_overlap.jsonl'
prior=read(prior_path)
names=collections.defaultdict(list)
for source in sources:names[Path(source).stem].append(source)
cached={}
for row in prior:
    path=row['source_audio_filepath']
    for source in names.get(Path(path).stem,[]):
        if path not in cached: cached[path]=pcm(path)
        if cached[path]==sources[source]:
            assert row['confirmed']
            evidence.append(dict(method='previous_acoustic_link_identical_complete_source',gold_source=source,
                jacktol_id=row['jacktol_id'],jacktol_split=row['jacktol_split'],
                prior_envelope_correlation=row['envelope_correlation'],
                prior_waveform_correlation=row['waveform_correlation']))
(R/'private-recording-evidence.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in evidence))
def aggregate(events):
    paths={e['gold_source'] for e in events}
    rows=[r for r in gold if r['audio_filepath'] in paths]
    return dict(source_recordings=len(paths),gold_clips=len(rows),gold_minutes=sum(r['duration'] for r in rows)/60)
report=dict(job_id=os.getenv('SLURM_JOB_ID'),jacktol_rows_checked=seen,gold_source_recordings=len(sources),
    by_split={s:aggregate([e for e in evidence if e['jacktol_split']==s]) for s in ('train','validation','test')},
    exact_recording_links=aggregate([e for e in evidence if e['method'].startswith('exact')]),
    prior_acoustic_recording_links=aggregate([e for e in evidence if e['method'].startswith('previous')]),
    recording_link_union=aggregate(evidence),
    prior_evidence_sha256=sha(prior_path.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
    private_evidence_sha256=sha((R/'private-recording-evidence.jsonl').read_bytes()))
(R/'recording-links-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE',json.dumps(report),flush=True)
