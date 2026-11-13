"""Verify the tagged private release through the Hub and the datasets consumer API."""
from pathlib import Path
import csv,hashlib,json
import soundfile as sf
from datasets import load_dataset,Audio
from huggingface_hub import HfApi,hf_hub_download
R=Path(__file__).resolve().parent
receipt=json.loads((R/'publication-receipt.json').read_text())
repo=receipt['repo'];revision=receipt['tag'];api=HfApi()
info=api.dataset_info(repo,revision=revision)
assert info.private and info.sha==receipt['release_revision']
remote={}
for split,count in [('test',1269),('extension',234)]:
    path=Path(hf_hub_download(repo,f'data/{split}/metadata.csv',repo_type='dataset',revision=revision))
    local=R/'package/data'/split/'metadata.csv'
    assert path.read_bytes()==local.read_bytes()
    rows=list(csv.DictReader(path.open()));assert len(rows)==count
    remote[split]={r['id']:r for r in rows}
    sample=rows[len(rows)//2]
    audio=Path(hf_hub_download(repo,f'data/{split}/'+sample['file_name'],repo_type='dataset',revision=revision))
    assert audio.read_bytes()==(local.parent/sample['file_name']).read_bytes()
    pcm,sr=sf.read(audio,dtype='int16',always_2d=True)
    assert sr==16000 and pcm.shape[1]==1
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==sample['pcm_sha256']
default=load_dataset(repo,'default',revision=revision,streaming=True)
assert set(default)=={'test','extension'}
for split in default:
    row=next(iter(default[split].cast_column('audio',Audio(decode=False))))
    assert row['text']==remote[split][row['id']]['text']
    assert bool(row['previously_used_for_p4_n4_training'])==(split=='extension')
    if isinstance(row['audio'],dict):
        assert row['audio'].get('path') or row['audio'].get('bytes')
    else:
        decoded=row['audio'].get_all_samples()
        assert decoded.sample_rate==16000 and decoded.data.shape[0]==1
        assert abs(decoded.data.shape[1]/16000-float(remote[split][row['id']]['duration']))<1e-9
combined=load_dataset(repo,'combined',split='test',revision=revision,streaming=True)
row=next(iter(combined.cast_column('audio',Audio(decode=False))))
assert row['id'] in (remote['test'].keys()|remote['extension'].keys())
result=dict(status='passed',repo=repo,private=True,tag=revision,revision=info.sha,
    remote_metadata_counts={s:len(rows) for s,rows in remote.items()},
    remote_metadata_bytes_match=True,remote_sample_audio_verified_per_split=1,
    remote_streaming_configurations_verified=['default/test','default/extension','combined/test'])
(R/'published-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('REMOTE_VERIFIED',json.dumps(result),flush=True)
