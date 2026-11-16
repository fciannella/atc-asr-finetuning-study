"""Stock-voice TTS preview with independent Whisper checks; no private audio uploads."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,json,hashlib,re,urllib.request
import numpy as np
import pandas as pd
import soundfile as sf
import riva.client
from riva.client.proto import riva_tts_pb2,riva_asr_pb2
from scipy.signal import butter,sosfilt
from score_wer import normalize,edit_distance
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('parquet',type=Path);p.add_argument('--arm',choices=['generic','targeted'],required=True);a=p.parse_args()
rows=pd.read_parquet(a.parquet).to_dict('records');assert 1<=len(rows)<=20
catalog=json.loads((R/'tts-catalog.json').read_text());params=catalog['config']['modelConfig'][0]['parameters']
voices=[params['voice_name']+'.'+x.split(':')[0] for x in params['subvoices'].split(',') if x.startswith('EN-US.') and x.split(':')[0].count('.')==1]
assert len(voices)>=4
key=Path.home().joinpath('.config/nvidia/adlr_ngc_api_key').read_text().strip()
request=urllib.request.Request('https://api.nvcf.nvidia.com/v2/nvcf/functions?visibility=public,authorized',headers={'Authorization':'Bearer '+key})
functions=json.load(urllib.request.urlopen(request,timeout=30))['functions'];whisper=next(f for f in functions if f['status']=='ACTIVE' and f['name']=='ai-whisper-large-v3')
def auth(fid):return riva.client.Auth(uri='grpc.nvcf.nvidia.com:443',use_ssl=True,metadata_args=[['function-id',fid],['authorization','Bearer '+key]])
ttsauth=auth(catalog['function_id']);tts=riva.client.SpeechSynthesisService(ttsauth)
asrauth=auth(whisper['id']);asr=riva.client.ASRService(asrauth)
out=R/('synthetic-'+a.arm+'-audio-preview');out.mkdir(exist_ok=True);manifest=[];rejects=[]
for i,r in enumerate(rows):
 u=r['utterance'];u=json.loads(u) if isinstance(u,str) else u;text=u['spoken_text'];ref=normalize(text)
 if not re.fullmatch('[a-z ]+',text) or not 8<=len(ref.split())<=35 or r['callsign'] not in text:
  rejects.append({'index':i,'reason':'text_format_length_or_callsign','text':text});continue
 voice=voices[i%len(voices)]
 req=riva_tts_pb2.SynthesizeSpeechRequest(text=text,voice_name=voice,language_code='en-US',sample_rate_hz=16000,encoding=riva.client.AudioEncoding.LINEAR_PCM)
 try:
  response=tts.stub.Synthesize(req,metadata=ttsauth.get_auth_metadata(),timeout=90)
  audio=np.frombuffer(response.audio,dtype='<i2');floating=audio.astype(np.float32)/32768;duration=len(audio)/16000
  quality={'duration_seconds':duration,'rms':float(np.sqrt(np.mean(floating**2))),'clipped_fraction':float(np.mean(np.abs(audio.astype(np.int32))>=32760))}
  assert 1<=duration<=30 and quality['rms']>.001 and quality['clipped_fraction']<.005,quality
  audio_path=out/('%02d-clean.wav'%i);sf.write(audio_path,audio,16000,subtype='PCM_16')
  # Artificial radio response; never use evaluation recordings/noise.
  radio=sosfilt(butter(5,[300,3400],btype='bandpass',fs=16000,output='sos'),floating)
  rng=np.random.default_rng(9000+i);noise=rng.standard_normal(len(radio));noise*=np.sqrt(np.mean(radio**2))/(10**(20/20));radio=np.clip(radio+noise,-.99,.99)
  sf.write(out/('%02d-radio.wav'%i),radio,16000,subtype='PCM_16')
  cfg=riva_asr_pb2.RecognitionConfig(encoding=riva.client.AudioEncoding.LINEAR_PCM,sample_rate_hertz=16000,language_code='en-US',audio_channel_count=1,max_alternatives=1,enable_automatic_punctuation=False)
  recognition=asr.stub.Recognize(riva_asr_pb2.RecognizeRequest(config=cfg,audio=audio.tobytes()),metadata=asrauth.get_auth_metadata(),timeout=90)
  hyp=' '.join(x.alternatives[0].transcript for x in recognition.results if x.alternatives)
  wer=edit_distance(ref.split(),normalize(hyp).split())/len(ref.split())
  record={'id':'synthetic-%s-%02d'%(a.arm,i),'text':ref,'intent':u['intent'],'focus':r['focus'],'voice':voice,'audio_filepath':str(audio_path.resolve()),'duration':duration,'tts_function':catalog['function_id'],'tts_version':catalog['function_version'],'qc_asr_function':whisper['id'],'qc_asr_version':whisper['versionId'],'qc_hypothesis':hyp,'qc_lexical_wer':wer,'quality':quality,'pcm_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'status':'preview_needs_listening_and_phraseology_review','training_eligible':False}
  manifest.append(record);print('SYNTHETIC_PREVIEW',i,'duration',round(duration,2),'independent_ASR_WER',round(wer,4),flush=True)
 except Exception as e:rejects.append({'index':i,'reason':type(e).__name__,'detail':str(e)[:500]});print('PREVIEW_FAILURE',i,type(e).__name__,flush=True)
 (out/'preview.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in manifest))
 (out/'report.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'arm':a.arm,'requested_rows':len(rows),'synthesized_and_qc_checked':len(manifest),'minutes':sum(x['duration'] for x in manifest)/60,'rejected_or_failed':rejects,'input_parquet_sha256':hashlib.sha256(a.parquet.read_bytes()).hexdigest(),'training_eligible':False,'notes':['Whisper disagreement flags review, not automatic rejection: numeric text formatting and ASR mistakes can inflate WER.','Artificial radio variant: 300-3400 Hz bandpass and Gaussian noise at 20 dB SNR. No test audio used.','Stock voices only; no voice cloning.','Text quality and spoken label correctness still require listening/phraseology review before training.']},indent=2)+'\n')
