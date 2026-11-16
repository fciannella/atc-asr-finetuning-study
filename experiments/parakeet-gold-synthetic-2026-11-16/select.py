from pathlib import Path
import json
R=Path(__file__).resolve().parent;c=json.loads((R/'contract.json').read_text())
base=json.loads((R/'evaluations/base/scores.json').read_text())
limit=base['splits']['english_guard']['wer']+c['selection']['english_guard_max_absolute_increase']
results=[]
for step in c['training']['keep_steps']:
 x=json.loads((R/'arms/p_g2/evaluations'/('step-%d'%step)/'scores.json').read_text());assert x['status']=='complete'
 x['english_guard_pass']=x['splits']['english_guard']['wer']<=limit;results.append(x)
passing=[x for x in results if x['english_guard_pass']]
best=min(passing,key=lambda x:(x['splits']['development']['wer'],x['step'])) if passing else None
out={'status':'selected' if best else 'no_candidate_passed_english_guard','selected':best,'candidates':results,'english_guard_limit':limit,'test_used':False,'test_pending':'Lock synthetic comparisons before final evaluation'}
(R/'arms/p_g2/selection.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
