"""Paired final Silver evaluation after development-only checkpoint selection."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
from native_baseline import digest,save
R=Path(sys.argv[1]);plan=json.loads((R/'plan.json').read_text());sel=json.loads((R/'selection.json').read_text())
assert sel['locked_before_test']
for n,h in sel['selection_inputs_sha256'].items():assert digest(R/'evaluations'/n/'scores.json')==h
names=['base-test',sel['checkpoint']+'-test'];scores={};predictions={}
for name in names:
 d=json.loads((R/'evaluations'/name/'scores.json').read_text());assert d['completed']==d['total']==1269 and d['plan_sha256']==digest(R/'plan.json')
 m=next(x for x in d['groups'] if x['split']=='test' and x['profile']=='atc_lexical_v1' and all(x[k]=='all' for k in ['cantica','narrator','canto']))
 assert m['examples']==1269 and m['reference_words']==13851
 scores[name]={**m,'token_limit_hits':d['token_limit_hits']}
 predictions[name]={p.stem:json.loads(p.read_text()) for p in (R/'evaluations'/name/'predictions').glob('*.json')}
a,b=[predictions[n] for n in names];assert a.keys()==b.keys()
for k in a:assert a[k]['pcm_sha256']==b[k]['pcm_sha256'] and a[k]['context']==b[k]['context']
save(R/'silver-final-report.json',{'status':'complete','completed_at':datetime.now(timezone.utc).isoformat(),'selection':sel,'test':scores,'test_prediction_pairs_verified':1269,'plan_sha256':digest(R/'plan.json'),'gold_status':'held_by_user_pending_additional_separate_human_labeled_data','limitations':plan['limitations']})
print('SILVER FINAL EVALUATION COMPLETE',flush=True)
