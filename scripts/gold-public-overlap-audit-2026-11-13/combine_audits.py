"""Combine the completed audits without changing any training/evaluation split."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
B=R.parent.parent
read=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
G=B/'experiments/atco2-gold-community-ablation-v1/inputs/gold-train.jsonl'
gold=read(G)
direct=json.loads((R/'audit-summary.json').read_text())
recordings=json.loads((R/'recording-links-summary.json').read_text())
exact=[e for e in read(R/'private-exact-evidence.jsonl') if e['target_kind']=='gold']
acoustic=[e for e in read(R/'private-jacktol-candidates.jsonl') if e['confirmed']]
links=read(R/'private-recording-evidence.jsonl')
exact_ids={e['target_id'] for e in exact}
acoustic_ids={e['source_id'] for e in acoustic}
sources={e['gold_source'] for e in exact+links}|{e['source_audio_filepath'] for e in acoustic}
excluded={r['id'] for r in gold if r['audio_filepath'] in sources}
assert exact_ids|acoustic_ids <= excluded
remaining={r['id'] for r in gold}-excluded
def stats(ids):
    rows=[r for r in gold if r['id'] in ids]
    return dict(clips=len(rows),minutes=sum(r['duration'] for r in rows)/60,
                source_recordings=len({r['audio_filepath'] for r in rows}))
original_path=B/'experiments/atco2-ancestry-audit-2026-11-06/common-mask-scores.json'
old=json.loads(original_path.read_text())
baseline=old['views']['exclude_jacktol_and_audited_atco2_training_and_validation']
baseline_minutes=baseline['hours']*60
combined=baseline_minutes+stats(remaining)['minutes']
private_mask={'excluded_ids':sorted(excluded),'unmatched_ids':sorted(remaining),
              'purpose':'Audit evidence only; no training or evaluation assignment changed.'}
(R/'private-gold-exclusion-mask.json').write_text(json.dumps(private_mask,indent=2)+'\n')
out=dict(date='2026-11-13',gold_manifest_sha256=sha(G),gold=stats({r['id'] for r in gold}),
    exact_clip_overlap=stats(exact_ids),additional_acoustic_clip_overlap=stats(acoustic_ids-exact_ids),
    additional_source_recording_exclusions=stats(excluded-exact_ids-acoustic_ids),
    conservative_exclusion_union=stats(excluded),no_identified_overlap=stats(remaining),
    evaluation_budget=dict(existing_known_overlap_removed_minutes=baseline_minutes,
        potential_additional_gold_minutes=stats(remaining)['minutes'],
        conditional_combined_minutes=combined,remaining_to_120_minutes=120-combined,
        eligible_added_minutes_for_existing_p4_n4=0,
        conditions='Only for future checkpoints whose entire training/selection ancestry excludes these Gold clips and linked recordings, including Silver exposure. No full-ancestry certification is established.'),
    gold_vs_original_test_identity_check=json.loads((R/'gold-test-identity-check.json').read_text()),
    direct_audio_audit=direct,recording_audit=recordings,
    prior_test_rescoring_sha256=sha(original_path),combine_script_sha256=sha(Path(__file__)),
    private_exclusion_mask_sha256=sha(R/'private-gold-exclusion-mask.json'))
(R/'gold-public-overlap-audit-2026-11-13.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('direct_audio_audit','recording_audit')},indent=2))
