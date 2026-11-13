"""Read-only Gold/public overlap audit. Row-level evidence stays restricted."""
from pathlib import Path
import collections, hashlib, io, json, os, wave
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from audit_jacktol_overlap import collect_jacktol, candidate_pairs, verify_candidates

R = Path(__file__).resolve().parent
B = R.parent.parent
A = B / 'experiments/atco2-ancestry-audit-2026-11-06'
G = B / 'experiments/atco2-gold-community-ablation-v1/inputs/gold-train.jsonl'
J = B / 'evaluation/jacktol-atc-asr-075e736/release-v1'
read = lambda p: [json.loads(l) for l in p.read_text().splitlines()]
sha = lambda data: hashlib.sha256(data).hexdigest()
gold = read(G)
assert len(gold) == 393
assert sha(G.read_bytes()) == 'e73737113cea302971ac1e5f46e7800b5863d9e4cf9d3a109b8cf6730408c106'

def wav_pcm(file):
    with wave.open(file, 'rb') as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        return w.readframes(w.getnframes())

def load_target(row, kind):
    path = row['audio_filepath']
    if path not in sources:
        a, sr = sf.read(path, dtype='int16', always_2d=True)
        assert sr == 16000 and a.shape[1] == 1
        data = a.astype('<i2').tobytes()
        assert data == wav_pcm(path)  # independent input decoder
        sources[path] = (data, sha(data))
    full, h = sources[path]
    start = int(row.get('offset', 0) * 16000)
    count = int(row['duration'] * 16000)
    crop = full[2*start:2*(start+count)]
    assert len(crop) == 2*count and count > 0 and any(crop), (row['id'],len(crop),count)
    return dict(id=row['id'], kind=kind, duration=row['duration'], path=path,
                source_hash=h, source_start=start, crop=crop, frames=count)

sources = {}
targets = [load_target(r, 'gold') for r in gold]
# Recover two previously confirmed public train/validation matches as controls.
old = read(A / 'private-match-evidence.jsonl')
test = {r['id']: r for r in read(B/'experiments/community-atc-eval-2026-11-06/test.jsonl')}
control_ids = set()
for split in ('train', 'validation'):
    e = next(x for x in old if x['repo_id']=='jlvdoorn/atco2-asr' and x['public_split']==split and x['test_segment_pcm_identical_at_source_offset'])
    t = test[e['test_id']]
    targets.append(load_target(dict(id=t['id'], audio_filepath=t['source_audio_filepath'],
                                   offset=t['source_offset'], duration=t['duration']), 'control'))
    control_ids.add(t['id'])

def first_aligned(haystack, needle):
    at = haystack.find(needle)
    while at >= 0 and at % 2:
        at = haystack.find(needle, at + 1)
    return at

evidence = []
cache = {}
stats = collections.defaultdict(collections.Counter)
def compare(pcm, metadata, independent_reader):
    h = sha(pcm)
    if h not in cache:
        matches = []
        for t in targets:
            start = first_aligned(pcm, t['crop'])
            if start >= 0:
                matches.append((t, 'full_gold_crop_contained', start//2, 0, t['frames']))
            elif len(pcm) >= 16000 and len(pcm) < len(t['crop']) and any(pcm):
                # Also detect shorter public clips contained within a Gold crop.
                at = first_aligned(t['crop'], pcm)
                if at >= 0:
                    matches.append((t, 'public_crop_contained_in_gold', 0, at//2, len(pcm)//2))
            elif h == t['source_hash']:
                raise RuntimeError('Identical whole source without exact target crop')
        cache[h] = matches
    matches = cache[h]
    if matches:
        assert independent_reader() == pcm
    for t, method, public_start, gold_start, frames in matches:
        assert pcm[2*public_start:2*(public_start+frames)] == t['crop'][2*gold_start:2*(gold_start+frames)]
        evidence.append(dict(**metadata, target_id=t['id'], target_kind=t['kind'],
            gold_duration=t['duration'], gold_source=t['path'], method=method,
            public_start_frame=public_start, gold_start_frame=gold_start, match_frames=frames,
            complete_source_identical=h==t['source_hash'], public_pcm_sha256=h,
            independent_wave_verified=True))

receipts = json.loads((A/'download-receipts.json').read_text())
for receipt in receipts:
    repo = receipt['repo_id']
    for f in receipt['files']:
        path = A/'public-data'/repo.replace('/', '__')/f['path']
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''): digest.update(block)
        assert digest.hexdigest() == f['sha256']
        split = 'validation' if '/validation-' in f['path'] else 'train'
        s = stats[(repo,split)]
        rowidx = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                raw = row['audio']['bytes']
                audio, sr = sf.read(io.BytesIO(raw), dtype='int16', always_2d=True)
                s['rows_decoded'] += 1
                if sr == 16000 and audio.shape[1] == 1:
                    s['rows_compared'] += 1
                    compare(audio.astype('<i2').tobytes(), dict(repo_id=repo,
                        revision=receipt['revision'], split=split, file=f['path'], row=rowidx),
                        lambda: wav_pcm(io.BytesIO(raw)))
                else:
                    s['rows_skipped_other_format'] += 1
                rowidx += 1
        print('PUBLIC_DONE', repo, f['path'], rowidx, 'evidence', len(evidence), flush=True)

jacktol, exact, grams = collect_jacktol(J)
manifest_hashes = {s:sha((J/'manifests'/f'{s}.jsonl').read_bytes()) for s in ('train','validation','test')}
for i, row in enumerate(jacktol):
    path = row['audio_filepath']
    assert sha(Path(path).read_bytes()) == row['audio_sha256']
    audio, sr = sf.read(path, dtype='int16', always_2d=True)
    s = stats[('jacktol/ATC-ASR-Dataset',row['_split'])]
    s['rows_decoded'] += 1
    assert sr == 16000 and audio.shape[1] == 1
    s['rows_compared'] += 1
    compare(audio.astype('<i2').tobytes(), dict(repo_id='jacktol/ATC-ASR-Dataset',
        revision=row['source_revision'], split=row['_split'], row_id=row['id']), lambda: wav_pcm(path))
    if (i+1)%1000 == 0: print('JACKTOL_EXACT', i+1, 'evidence',len(evidence),flush=True)

candidates, candidate_stats = candidate_pairs(G, 'gold_train', jacktol, exact, grams, .65)
print('JACKTOL_CANDIDATES', len(candidates), flush=True)
verified = verify_candidates(candidates)
(R/'private-jacktol-candidates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in verified))
(R/'private-exact-evidence.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in evidence))
found_controls = {e['target_id'] for e in evidence if e['target_kind']=='control'}
assert control_ids <= found_controls, (control_ids, found_controls)

def aggregate(events):
    ids = {e['target_id'] for e in events}
    full = {e['target_id'] for e in events if e['method']=='full_gold_crop_contained'}
    intervals = collections.defaultdict(list)
    for e in events:
        intervals[e['target_id']].append((e['gold_start_frame'],e['gold_start_frame']+e['match_frames']))
    frames = 0
    for spans in intervals.values():
        end = 0
        for left,right in sorted(spans):
            frames += max(0,right-max(left,end)); end=max(end,right)
    return dict(affected_clips=len(ids), affected_clip_minutes=sum(r['duration'] for r in gold if r['id'] in ids)/60,
                fully_contained_clips=len(full), fully_contained_minutes=sum(r['duration'] for r in gold if r['id'] in full)/60,
                matched_audio_minutes=frames/16000/60,
                identical_complete_source_recordings=len({e['gold_source'] for e in events if e['complete_source_identical']}))

gold_e = [e for e in evidence if e['target_kind']=='gold']
datasets = []
for (repo,split), s in stats.items():
    events = [e for e in gold_e if e['repo_id']==repo and e['split']==split]
    datasets.append(dict(repo_id=repo,split=split,**dict(s),**aggregate(events)))
acoustic = [e for e in verified if e['confirmed']]
acoustic_ids = {e['source_id'] for e in acoustic}
exact_ids = {e['target_id'] for e in gold_e}
report = dict(date='2026-11-13', job_id=os.getenv('SLURM_JOB_ID'),
    gold_manifest_sha256=sha(G.read_bytes()), gold_clips=len(gold), gold_minutes=sum(r['duration'] for r in gold)/60,
    gold_source_recordings=len({r['audio_filepath'] for r in gold}),
    method='Exact 16-kHz mono signed-int16 decoded PCM containment in both directions, independent WAV verification for every positive; no filename or transcript requirement. All eligible rows searched. Additional transcript-candidate acoustic check against Jacktol.',
    datasets=datasets, exact_union=aggregate(gold_e),
    exact_train_union=aggregate([e for e in gold_e if e['split']=='train']),
    exact_validation_union=aggregate([e for e in gold_e if e['split']=='validation']),
    positive_controls=dict(expected=len(control_ids),recovered=len(found_controls)),
    jacktol_candidate_audit=dict(candidate_pairs=len(verified),confirmed_pairs=len(acoustic),
       confirmed_clips=len(acoustic_ids),confirmed_clip_minutes=sum(r['duration'] for r in gold if r['id'] in acoustic_ids)/60,
       additional_clips_beyond_exact=len(acoustic_ids-exact_ids), fuzzy_threshold=.65),
    exact_unmatched_clips=len(gold)-len(exact_ids),
    exact_unmatched_minutes=sum(r['duration'] for r in gold if r['id'] not in exact_ids)/60,
    limitations=['Negative results do not exclude partial overlaps where neither full clip contains the other, arbitrary audio transformations, missing public snapshots or inaccessible pretraining corpora.',
       '9559 ATCOSIM 32-kHz rows decoded but not compared in the exact 16-kHz pass.',
       'Jacktol acoustic candidate pass is transcript/duration filtered and uses the existing heuristic correlation criteria; it is not a complete transformed-audio search.',
       'These Gold clips were used by P4/N4; public overlap absence would not make them held-out evaluation for those checkpoints.'],
    public_download_receipts=receipts,jacktol_manifest_sha256=manifest_hashes,
    script_sha256=sha(Path(__file__).read_bytes()), jacktol_helper_sha256=sha((R/'audit_jacktol_overlap.py').read_bytes()),
    private_exact_evidence_sha256=sha((R/'private-exact-evidence.jsonl').read_bytes()),
    private_jacktol_candidates_sha256=sha((R/'private-jacktol-candidates.jsonl').read_bytes()))
(R/'audit-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('public_download_receipts',)}),flush=True)
