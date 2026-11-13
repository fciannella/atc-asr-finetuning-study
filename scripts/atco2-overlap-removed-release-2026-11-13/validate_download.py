"""Validate transported release bytes and actual Hugging Face dataset configurations."""
from pathlib import Path
import csv,hashlib,json,tarfile,wave
from datasets import load_dataset,Audio
from huggingface_hub import DatasetCard
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((R/'build-receipt.json').read_text())
assert sha(R/'package.tar.gz')==receipt['archive_sha256']
package=R/'package'
if not package.exists():
    package.mkdir()
    with tarfile.open(R/'package.tar.gz') as tar:tar.extractall(package,filter='data')
assert sha(package/'SHA256SUMS')==receipt['checksums_sha256']
expected=set()
for line in (package/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1);expected.add(name)
    assert sha(package/name)==digest,name
assert {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}==expected|{'SHA256SUMS'}
assert len(expected)+1==receipt['files']==1509
assert sha(package/'release.json')==receipt['release_json_sha256']
rows_by_split={}
for split,count in [('test',1269),('extension',234)]:
    rows=list(csv.DictReader((package/'data'/split/'metadata.csv').open()))
    assert len(rows)==count
    rows_by_split[split]=rows
    for r in rows:
        assert '/lustre/' not in json.dumps(r)
        path=package/'data'/split/r['file_name']
        with wave.open(str(path)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
            assert abs(w.getnframes()/16000-float(r['duration']))<1e-9
            assert hashlib.sha256(w.readframes(w.getnframes())).hexdigest()==r['pcm_sha256']
        assert (r['previously_used_for_p4_n4_training']=='True')==(split=='extension')
card=DatasetCard.load(package/'README.md')
assert len(card.data.configs)==2
default=load_dataset(str(package),name='default')
assert {s:len(d) for s,d in default.items()}=={'test':1269,'extension':234}
combined=load_dataset(str(package),name='combined',split='test')
assert len(combined)==1503
for split,d in default.items():
    assert set(d['id'])=={r['id'] for r in rows_by_split[split]}
    assert dict(zip(d['id'],d['text']))=={r['id']:r['text'] for r in rows_by_split[split]}
    audio=d.cast_column('audio',Audio(decode=False))[0]['audio']
    assert audio.get('path') or audio.get('bytes')
assert set(combined['id'])=={r['id'] for rows in rows_by_split.values() for r in rows}
result=dict(status='passed',package_files=1509,wav_files=1503,
    configs={'default':{'test':1269,'extension':234},'combined':{'test':1503}},
    all_file_checksums_verified=True,all_wav_pcm_hashes_verified=True,all_transcripts_match=True,
    datasets_version=__import__('datasets').__version__,checksums_sha256=receipt['checksums_sha256'])
(R/'local-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('VALIDATED',json.dumps(result),flush=True)
