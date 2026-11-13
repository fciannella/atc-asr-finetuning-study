"""Materialize the frozen audited ATCO2 pool; never overwrite an existing release."""
from pathlib import Path
import csv, gzip, hashlib, json, math, tarfile, wave
import numpy as np
import soundfile as sf

R=Path(__file__).resolve().parent
B=R.parent.parent
OUT=R/'package'
assert not OUT.exists(), 'Frozen package already exists; use a new release version'
read=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=B/'experiments/atco2-gold-community-ablation-v1/inputs'
test_path=inputs/'gold-community-test.jsonl'
gold_path=inputs/'gold-train.jsonl'
mask_path=B/'experiments/atco2-ancestry-audit-2026-11-06/private-exclusion-masks.json'
gold_mask_path=B/'experiments/gold-public-overlap-audit-2026-11-13/private-gold-exclusion-mask.json'
expected={test_path:'4af85f51d32051ee84c841c92c8e5e1eb69ee07201149197f905a7edc9c13082',
gold_path:'e73737113cea302971ac1e5f46e7800b5863d9e4cf9d3a109b8cf6730408c106',
mask_path:'11b88d3aeb7f2f89b401d3147ea905b0aab6b2d2ef8a98a0bc9a9dc0823f5b61',
gold_mask_path:'476b5d22a4867568c1682a3421249fc74715e9c93ea47651b07a2ebacec7a7f8'}
# Explicit immutable hashes for both the selected and excluded inventories.
for path,digest in expected.items():assert sha(path)==digest,(path,'input hash changed')
test=read(test_path);gold=read(gold_path)
mask=json.loads(mask_path.read_text())['excluded_by_mask']['exclude_jacktol_and_audited_atco2_training_and_validation']
gold_mask=json.loads(gold_mask_path.read_text())
assert len(test)==1908 and len(gold)==393 and len(mask)==639 and len(gold_mask['excluded_ids'])==159
assert set(mask)<={r['id'] for r in test}
assert set(gold_mask['excluded_ids'])|set(gold_mask['unmatched_ids'])=={r['id'] for r in gold}
assert set(gold_mask['excluded_ids']).isdisjoint(gold_mask['unmatched_ids'])
parts={'test':[r for r in test if r['id'] not in set(mask)],
       'extension':[r for r in gold if r['id'] in set(gold_mask['unmatched_ids'])]}
assert len(parts['test'])==1269 and len(parts['extension'])==234
assert math.isclose(sum(r['duration'] for r in parts['test'])/60,75.76146666666664)
assert math.isclose(sum(r['duration'] for r in parts['extension'])/60,14.879633333333343)
for key in ('id','source_record_id','audio_filepath','gold_group_key'):
    assert {r[key] for r in parts['test']}.isdisjoint({r[key] for r in parts['extension']}),key
evaluated={r['id']:r for r in read(B/'experiments/community-atc-eval-2026-11-06/test.jsonl')}
OUT.mkdir()
pcm_hashes=set();ids=set();private=[];stats={};source_cache={}
fields=['file_name','id','text','duration','airport','recorded_date','source_record_id',
        'source_offset_seconds','origin_partition','previously_used_for_p4_n4_training','pcm_sha256']
for split,rows in parts.items():
    folder=OUT/'data'/split;folder.mkdir(parents=True)
    seconds=0;words=0
    with (folder/'metadata.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            assert row['id'] not in ids;ids.add(row['id'])
            assert row['gold_acceptance']=='human_rechecker_accepted' and row['text'].strip()
            src=row['audio_filepath']
            if src not in source_cache:
                with wave.open(src,'rb') as w:
                    assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
                    source_cache[src]=w.readframes(w.getnframes())
            raw=source_cache[src]
            start=int(row['offset']*16000);count=int(row['duration']*16000)
            assert start>=0 and count>0 and 2*(start+count)<=len(raw)
            crop=raw[2*start:2*(start+count)]
            digest=hashlib.sha256(crop).hexdigest()
            assert digest not in pcm_hashes,('duplicate complete PCM',row['id'])
            pcm_hashes.add(digest)
            name=hashlib.sha256(row['id'].encode()).hexdigest()[:20]+'.wav'
            dest=folder/name
            with wave.open(str(dest),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(crop)
            independent,sr=sf.read(dest,dtype='int16',always_2d=True)
            assert sr==16000 and independent.shape==(count,1)
            assert independent.astype('<i2').tobytes()==crop
            if split=='test':
                old=evaluated[row['id']]
                assert old['known_jacktol_disjoint'] and old['text']==row['text']
                with wave.open(old['audio_filepath'],'rb') as w:assert w.readframes(w.getnframes())==crop
            duration=count/16000
            record=dict(file_name=name,id=row['id'],text=row['text'],duration=duration,
                airport=row['airport'],recorded_date=row['recorded_date'],source_record_id=row['source_record_id'],
                source_offset_seconds=start/16000,origin_partition='community_test' if split=='test' else 'gold_train',
                previously_used_for_p4_n4_training=split=='extension',pcm_sha256=digest)
            writer.writerow(record)
            private.append(dict(row,release_split=split,release_audio_filepath=str(dest),release_pcm_sha256=digest))
            seconds+=duration;words+=len(row['text'].split())
    stats[split]=dict(clips=len(rows),audio_seconds=seconds,audio_minutes=seconds/60,
        manifest_minutes=sum(r['duration'] for r in rows)/60,source_recordings=len({r['source_record_id'] for r in rows}),
        metadata_sha256=sha(folder/'metadata.csv'),previously_used_for_p4_n4_training=split=='extension')
    print('VERIFIED',split,json.dumps(stats[split]),flush=True)
assert len(ids)==len(pcm_hashes)==1503
(R/'private-release-source-manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in private))
(OUT/'README.md').write_text((R/'README.dataset.md').read_text())
(OUT/'.gitattributes').write_text('*.wav filter=lfs diff=lfs merge=lfs -text\n')
release=dict(version='atco2-known-overlap-removed-v2-2026-11-13',clips=1503,
    audio_minutes=sum(s['audio_minutes'] for s in stats.values()),splits=stats,
    selection='Remove the union of known Jacktol source-recording links and audited public ATCO2 train/validation matches; extend with Gold clips surviving the November 13 exact, acoustic and source-recording audit.',
    inputs={p.name:sha(p) for p in expected},builder_sha256=sha(Path(__file__)),
    crop_policy='start=int(offset_seconds*16000); frames=int(manifest_duration_seconds*16000), matching the audited/evaluated PCM; preserve original signed-int16 mono samples and transcript text.',
    verification=dict(input_hashes_passed=True,excluded_ids_absent=True,unique_ids=1503,unique_complete_pcm_hashes=1503,
        all_audio_independently_decoded=True,original_test_audio_matches_evaluated_pcm=1269,
        shared_source_records_between_splits=0,shared_airport_date_groups_between_splits=0),
    scope='Known overlap removed against the frozen audited snapshots; no certification of every public corpus, arbitrary transformed audio or full model ancestry.',
    existing_model_eligibility='P4/N4 use only default/test (1269 clips). All 234 extension clips were in their Gold training pool. Combined evaluation requires a lineage excluding the extension and linked recordings throughout training and selection.')
(OUT/'release.json').write_text(json.dumps(release,indent=2)+'\n')
files=sorted(p for p in OUT.rglob('*') if p.is_file())
(OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(OUT).as_posix()+'\n' for p in files))
seal=dict(version=release['version'],files=len(files)+1,release_json_sha256=sha(OUT/'release.json'),
    checksums_sha256=sha(OUT/'SHA256SUMS'),builder_sha256=sha(Path(__file__)),
    private_source_manifest_sha256=sha(R/'private-release-source-manifest.jsonl'))
with (R/'package.tar.gz').open('wb') as f:
    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for p in sorted(p for p in OUT.rglob('*') if p.is_file()):
                item=tar.gettarinfo(str(p),arcname=p.relative_to(OUT).as_posix())
                item.uid=item.gid=0;item.uname=item.gname='';item.mtime=0;item.mode=0o644
                with p.open('rb') as source:tar.addfile(item,source)
seal['archive_sha256']=sha(R/'package.tar.gz');seal['archive_bytes']=(R/'package.tar.gz').stat().st_size
(R/'build-receipt.json').write_text(json.dumps(seal,indent=2)+'\n')
print('COMPLETE',json.dumps(seal),flush=True)
