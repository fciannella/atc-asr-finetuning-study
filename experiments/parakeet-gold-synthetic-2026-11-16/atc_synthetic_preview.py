# /// script
# dependencies = ["data-designer==0.10.0"]
# ///
"""Controlled fictional ATC text; no held-out sentences or model paraphrasing."""
import json,os,random
from pathlib import Path
from collections import Counter
import data_designer.config as dd
R=Path(__file__).resolve().parent
WORDS='zero one two three four five six seven eight niner'.split()
def digits(n,width):return ' '.join(WORDS[int(c)] for c in str(n).zfill(width))
COMMANDS={}
def add(focus,intent,phase,command):
 key='command_%03d'%len(COMMANDS);COMMANDS[key]={'focus':focus,'intent':intent,'phase':phase,'command':command}
for n in [50,90,120,180,230,270,310,350]:add('number_terms','heading','enroute','turn right heading '+digits(n,3))
for n in [180,230,270,310,350]:add('number_terms','flight_level','enroute','climb and maintain flight level '+digits(n,3))
for n in ['0421','1203','2345','4601','5124','6203','7012']:
 assert len(n)==4 and all(c in '01234567' for c in n) and n not in ['7500','7600','7700']
 add('number_terms','squawk','enroute','squawk '+' '.join(WORDS[int(c)] for c in n))
for runway in ['zero nine','two seven','one eight','three six']:
 for route in ['alpha','bravo','charlie','delta','alpha bravo','charlie delta']:
  add('phonetic_alphabet','taxi','taxi','taxi via '+route+' hold short of runway '+runway)
 for action in ['cleared to land runway ','hold short of runway ']:add('atc_commands','runway_instruction','approach' if action.startswith('cleared') else 'taxi',action+runway)
for name,freq in [('tower','one one eight decimal seven'),('ground','one two one decimal seven'),('departure','one two four decimal six')]:
 add('callsign_related_terms','frequency_change','departure','contact '+name+' '+freq)
for command in ['say again last transmission','request descent to flight level two seven zero','request direct routing when able','unable due to weather request vectors']:
 add('other','pilot_request','enroute',command)

@dd.custom_column_generator(required_columns=['command_id','callsign'],side_effect_columns=['focus','phase','speaker_role'])
def render(row):
 c=COMMANDS[row['command_id']];text=row['callsign']+' '+c['command']
 row['utterance']={'spoken_text':text,'intent':c['intent']};row['focus']=c['focus'];row['phase']=c['phase'];row['speaker_role']='pilot' if c['focus']=='other' else 'controller'
 return row

def load_config_builder():
 arm=os.environ.get('ATC_SYNTHETIC_ARM','generic');assert arm in ['generic','targeted']
 categories=list(dict.fromkeys(x['focus'] for x in COMMANDS.values()));mass={c:1 for c in categories}
 if arm=='targeted':
  x=json.loads((R/'development-error-analysis.json').read_text());assert x['scope']=='development_only' and x['examples']==100
  mass={c:x['category_counts'].get(c,{}).get('attributed_errors',0) if c!='other' else 0 for c in categories}
 counts=Counter(x['focus'] for x in COMMANDS.values());ids=[k for k,v in COMMANDS.items() if mass[v['focus']]>0]
 weights=[mass[COMMANDS[k]['focus']]/counts[COMMANDS[k]['focus']] for k in ids]
 b=dd.DataDesignerConfigBuilder()
 def cat(name,values,weights=None):b.add_column(dd.SamplerColumnConfig(name=name,sampler_type='category',params=dd.CategorySamplerParams(values=values,weights=weights)))
 cat('arm',[arm]);cat('command_id',ids,weights)
 rng=random.Random(16942);callsigns=[rng.choice(['speedbird','lufthansa','united','american','ryanair','swiss'])+' '+digits(rng.randrange(100,1000),3) for _ in range(300)]
 cat('callsign',callsigns);b.add_column(dd.CustomColumnConfig(name='utterance',generator_function=render))
 return b
