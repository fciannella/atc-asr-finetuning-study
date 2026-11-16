from pathlib import Path
from collections import Counter
import json,re,hashlib
from score_wer import normalize,edit_distance
R=Path(__file__).resolve().parent
DIGITS='zero one two three four five six seven eight nine'.split()
def numeric_view(text):
 text=re.sub(r'(?<=\d)\.(?=\d)',' decimal ',text)
 text=re.sub(r'\d',lambda m:' '+DIGITS[int(m[0])]+' ',text)
 return normalize(text).replace('niner','nine')
out={'status':'preview_only_not_training_eligible','freeform_attempt':'Rejected for phraseology defects and unhelpful category coverage; ten synthesized files quarantined.','text_generator':'Data Designer category samplers plus constrained command templates. No LLM paraphrasing in retained preview.','tts':'Magpie Multilingual stock voices','independent_qc_asr':'Whisper large v3','holdout_content_used':False,'review_pending':['listening to verify spoken labels','aviation phraseology review','residual-development targeting after Gold baseline selection','larger corpus and matched-duration ablation'],'arms':{}}
for arm in ['generic','targeted']:
 p=R/('synthetic-'+arm+'-audio-preview');report=json.loads((p/'report.json').read_text());rows=[json.loads(l) for l in (p/'preview.jsonl').read_text().splitlines()];metrics={}
 for name,normalizer in [('primary_lexical',normalize),('numeric_format_diagnostic',numeric_view)]:
  words=sum(len(normalizer(x['text']).split()) for x in rows);errors=sum(edit_distance(normalizer(x['text']).split(),normalizer(x['qc_hypothesis']).split()) for x in rows);metrics[name]={'reference_words':words,'word_errors':errors,'wer':errors/words}
 out['arms'][arm]={'clips':len(rows),'minutes_clean':sum(x['duration'] for x in rows)/60,'paired_radio_variants':len(rows),'category_counts':dict(Counter(x['focus'] for x in rows)),'voices':dict(Counter(x['voice'] for x in rows)),'automatic_audio_checks_passed':len(rows),'independent_asr_agreement':metrics,'not_model_benchmark':True,'provenance':{k:rows[0][k] for k in ['tts_function','tts_version','qc_asr_function','qc_asr_version']},'manifest_sha256':hashlib.sha256((p/'preview.jsonl').read_bytes()).hexdigest(),'input_parquet_sha256':report['input_parquet_sha256']}
out['numeric_format_diagnostic']='Supplementary QC only: expand decimal digits individually, decimal point between digits to decimal, and niner to nine. Does not correct homophones or change benchmark scoring; ASR agreement does not prove audio correctness.'
(R/'synthetic-preview-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({arm:{k:v for k,v in x.items() if k not in ['voices','provenance','manifest_sha256','input_parquet_sha256']} for arm,x in out['arms'].items()},indent=2))
