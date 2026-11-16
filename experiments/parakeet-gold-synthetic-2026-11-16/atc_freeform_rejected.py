# /// script
# dependencies = ["data-designer==0.10.0", "pydantic"]
# ///
"""Independent ATC text; never ingest development/test sentences."""
import json,os,random
from pathlib import Path
from pydantic import BaseModel,Field
import data_designer.config as dd
R=Path(__file__).resolve().parent
class Utterance(BaseModel):
 spoken_text: str=Field(description='One realistic English ATC radio transmission, eight to thirty five words. Lowercase spoken words only, no digits, punctuation, stage directions, dialogue labels or explanatory text. Include the exact supplied callsign.')
 intent: str=Field(description='Short intent label describing the clearance, readback or request.')

def load_config_builder():
 arm=os.environ.get('ATC_SYNTHETIC_ARM','generic');assert arm in ['generic','targeted']
 categories=['number_terms','phonetic_alphabet','atc_commands','callsign_related_terms','other']
 weights=None
 if arm=='targeted':
  x=json.loads((R/'development-error-analysis.json').read_text());assert x['scope']=='development_only' and x['examples']==100
  weights=[max(1,x['category_counts'].get(c,{}).get('attributed_errors',0)) for c in categories]
 b=dd.DataDesignerConfigBuilder()
 def cat(name,values,weights=None):b.add_column(dd.SamplerColumnConfig(name=name,sampler_type='category',params=dd.CategorySamplerParams(values=values,weights=weights)))
 cat('arm',[arm]);cat('focus',categories,weights)
 cat('speaker_role',['controller','pilot'])
 cat('phase',['departure','approach','enroute','taxi'])
 rng=random.Random(16942);digits='zero one two three four five six seven eight niner'.split()
 callsigns=[rng.choice(['speedbird','lufthansa','united','american','ryanair','swiss'])+' '+' '.join(rng.choices(digits,k=3)) for _ in range(300)]
 cat('callsign',callsigns)
 cat('number_prompt',[' '.join(rng.choices(digits,k=3)) for _ in range(300)])
 b.add_column(dd.LLMStructuredColumnConfig(name='utterance',model_alias='atc-nemotron',output_format=Utterance,
 system_prompt='You write fictional but plausible English air traffic control training utterances. Use standard civil aviation phraseology, realistic clearances, and spoken digit-by-digit callsigns. No transcript or recording is provided. Invent every sentence. These are synthetic ASR research examples, not operational instructions.',
 prompt='Create one {{ speaker_role }} transmission for {{ phase }}. Research arm: {{ arm }}. Emphasize {{ focus }}. Include this exact spoken callsign: {{ callsign }}. An independently sampled number sequence is {{ number_prompt }}; use it only if operationally plausible. For phonetic focus include a spelled registration or taxiway using NATO words. For numbers include a plausible frequency, flight level, heading or squawk. For commands include one clear ATC instruction or readback. For other focus vary ordinary request language. Return spoken_text and intent in the requested schema. Avoid greeting, commentary, bracketed annotations and invented aviation terms.'))
 return b
