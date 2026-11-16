"""Development-only lexical error attribution; never a source of target sentences."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,re
from score_wer import normalize,edit_distance
NUMBERS=set('zero oh one two three four five six seven eight nine niner fife tree hundred thousand decimal point double triple'.split())
NATO=set('alpha alfa bravo charlie delta echo foxtrot golf hotel india juliet juliett kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray x ray yankee zulu'.split())
COMMANDS=set('climb descend maintain cleared clearance runway heading flight level contact squawk taxi hold short approach departure tower ground center centre speed knots turn left right direct report landing takeoff wind traffic'.split())
AIRLINES=set('speedbird lufthansa swiss easy ryanair wizz klm airfrance air france austrian delta united american emirates qantas'.split())
def category(w):
 if w in NUMBERS or w.isdigit():return 'number_terms'
 if w in NATO:return 'phonetic_alphabet'
 if w in COMMANDS:return 'atc_commands'
 if w in AIRLINES:return 'callsign_related_terms'
 return 'other'
def align(ref,hyp):
 n,m=len(ref),len(hyp);d=[[0]*(m+1) for _ in range(n+1)]
 for i in range(n+1):d[i][0]=i
 for j in range(m+1):d[0][j]=j
 for i in range(1,n+1):
  for j in range(1,m+1):d[i][j]=min(d[i-1][j-1]+(ref[i-1]!=hyp[j-1]),d[i-1][j]+1,d[i][j-1]+1)
 ops=[];i,j=n,m
 while i or j:
  if i and j and d[i][j]==d[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
   ops.append(('C' if ref[i-1]==hyp[j-1] else 'S',ref[i-1],hyp[j-1]));i-=1;j-=1
  elif i and d[i][j]==d[i-1][j]+1:ops.append(('D',ref[i-1],''));i-=1
  else:ops.append(('I','',hyp[j-1]));j-=1
 return list(reversed(ops))
def main():
 p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--raw-references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert 'development' in str(a.predictions),'Only development error analysis is authorized for generation targeting'
 raw={r['id']:r for r in map(json.loads,a.raw_references.read_text().splitlines())};rows=[json.loads(l) for l in a.predictions.open()];assert len(rows)==len(raw)==100
 totals=Counter();categories=defaultdict(Counter);private=[];raw_errors=raw_words=0
 for r in rows:
  assert r['id'] in raw
  reference=normalize(raw[r['id']]['text']).split();hyp=normalize(r['pred_text']).split();ops=align(reference,hyp);errors=sum(op!='C' for op,_,_ in ops);assert errors==edit_distance(reference,hyp)
  totals['reference_words']+=len(reference);totals['word_errors']+=errors
  raw_ref=raw[r['id']]['text'].split();raw_hyp=r['pred_text'].split();raw_errors+=edit_distance(raw_ref,raw_hyp);raw_words+=len(raw_ref)
  for op,x,y in ops:
   if op!='I':categories[category(x)]['reference_words']+=1
   if op!='C':totals[op]+=1;categories[category(x or y)][op]+=1;categories[category(x or y)]['attributed_errors']+=1
  totals['rows_with_digit_format_difference']+=int(any(c.isdigit() for c in raw[r['id']]['text'])!=any(c.isdigit() for c in r['pred_text']))
  private.append({'id':r['id'],'audio_filepath':r['audio_filepath'],'offset':r.get('offset',0),'duration':r['duration'],'reference':raw[r['id']]['text'],'hypothesis':r['pred_text'],'word_errors':errors,'reference_words':len(reference),'operations':ops})
 assert totals['S']+totals['D']+totals['I']==totals['word_errors']
 result={'scope':'development_only','examples':len(rows),'lexical_wer':totals['word_errors']/totals['reference_words'],'raw_wer':raw_errors/raw_words,'counts':dict(totals),'category_counts':dict(categories),'generation_policy':'Use aggregate categories only. Do not copy development references, hypotheses, specific callsigns, numeric sequences or recordings into synthetic examples.','category_caveats':['Categories are heuristic token classes, not verified entity spans.','Calls sign terms do not measure full callsign accuracy.','Digit-format mismatches require review before treating numeric errors as acoustic failures.','Noise, accent, clipping and reference defects require audio review and cannot be inferred from token counts alone.'],'predictions_sha256':hashlib.sha256(a.predictions.read_bytes()).hexdigest()}
 a.output.write_text(json.dumps(result,indent=2)+'\n');a.output.with_name('private-error-review.json').write_text(json.dumps(sorted(private,key=lambda x:x['word_errors'],reverse=True),indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
