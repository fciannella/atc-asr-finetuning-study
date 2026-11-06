#!/usr/bin/env python3
"""Pinned, audio-only community ASR evaluation. Raw predictions stay in --run-root."""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,os,subprocess,sys,time,traceback
from pathlib import Path
from score_wer import normalize,edit_distance

def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def save(p,obj): Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def counts(ref,hyp):
 # Deterministic optimal alignment: substitution, deletion, insertion tie order.
 prev=[(j,0,0,j) for j in range(len(hyp)+1)]
 for i,a in enumerate(ref,1):
  cur=[(i,0,i,0)]
  for j,b in enumerate(hyp,1):
   c,s,d,n=prev[j-1];sub=(c+(a!=b),s+(a!=b),d,n)
   c,s,d,n=prev[j];dele=(c+1,s,d+1,n)
   c,s,d,n=cur[j-1];ins=(c+1,s,d,n+1)
   cur.append(min([sub,dele,ins],key=lambda v:v[0]))
  prev=cur
 return prev[-1]
def score(rows):
 out={}
 for label,part in [('full',rows),('known_jacktol_disjoint',[r for r in rows if r['known_jacktol_disjoint']])]:
  sums=[0,0,0,0];words=0;byrec={}
  for r in part:
   ref=normalize(r['text']).split();hyp=normalize(r['pred_text']).split();v=counts(ref,hyp)
   assert v[0]==edit_distance(ref,hyp)
   sums=[a+b for a,b in zip(sums,v)];words+=len(ref)
   g=byrec.setdefault(r['source_audio_filepath'],[0,0]);g[0]+=v[0];g[1]+=len(ref)
  import numpy as np
  a=np.array(list(byrec.values()),dtype=np.int64);rng=np.random.default_rng(20261106)
  samples=a[rng.integers(0,len(a),size=(2000,len(a)))].sum(axis=1);ci=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist()
  out[label]={'utterances':len(part),'reference_words':words,'word_errors':sums[0],'substitutions':sums[1],'deletions':sums[2],'insertions':sums[3],'wer':sums[0]/words,'source_recordings':len(byrec),'wer_95pct_recording_cluster_bootstrap':ci}
 return out

def nemo_infer(model,root,rows,split,out):
 directory=out/split;directory.mkdir(exist_ok=True)
 manifest=directory/'input.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 pred=directory/'predictions.jsonl'
 cmd=[sys.executable,'/opt/prompt_model/NeMo/examples/asr/speech_to_text_eval.py',f'model_path={root/"models"/model["slug"]/model["weight_files"][0]}',f'dataset_manifest={manifest}',f'output_filename={pred}','batch_size=8','amp=false','compute_dtype=bfloat16','matmul_precision=high','use_cer=false','rnnt_decoding.strategy=greedy_batch','rnnt_decoding.greedy.max_symbols_per_step=10','rnnt_decoding.greedy.use_cuda_graph_decoder=true']
 if model['kind']=='nemo-tdt':cmd+=['rnnt_decoding.model_type=tdt']
 else:cmd+=['decoder_type=rnnt']
 save(directory/'command.json',cmd)
 with (directory/'inference.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
 preds=read(pred);assert len(preds)==len(rows)
 for a,b in zip(rows,preds):
  assert a['text']==b['text'] and a['audio_filepath']==b['audio_filepath']
 return [{**a,'pred_text':b['pred_text']} for a,b in zip(rows,preds)]

def hf_loader(model,root):
 import torch
 from transformers import AutoProcessor,WhisperForConditionalGeneration,AutoModelForCTC,Wav2Vec2Processor
 local=root/'models'/model['slug'];kind=model['kind'];meta={}
 if kind=='ctc':
  processor=Wav2Vec2Processor.from_pretrained(local,local_files_only=True)
  net=AutoModelForCTC.from_pretrained(local,local_files_only=True,torch_dtype=torch.bfloat16)
 else:
  base=root/'models'/'whisper-turbo-base' if kind=='whisper-lora' else local
  processor=AutoProcessor.from_pretrained(base,local_files_only=True)
  net=WhisperForConditionalGeneration.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
  if kind=='whisper-lora':
   from peft import PeftModel,LoraConfig
   cfg=json.loads((local/'adapter_config.json').read_text())
   fields=inspect.signature(LoraConfig).parameters
   unknown={k:v for k,v in cfg.items() if k not in fields}
   # New-version optional/default-only metadata may be removed; never drop an active adapter option.
   allowed_metadata={'peft_version','auto_mapping'}
   assert all(v is None or v is False or k in allowed_metadata for k,v in unknown.items()),unknown
   clean={k:v for k,v in cfg.items() if k in fields}
   adapter=LoraConfig(**clean)
   net=PeftModel.from_pretrained(net,str(local),config=adapter,local_files_only=True,is_trainable=False)
   net=net.merge_and_unload();meta['adapter_default_metadata_omitted']=unknown
  net.generation_config.forced_decoder_ids=None
 net=net.to('cuda').eval()
 return net,processor,meta

def hf_infer(model,net,processor,rows,out,split):
 import torch,soundfile as sf
 directory=out/split;directory.mkdir(exist_ok=True);pred=directory/'predictions.jsonl';results=[]
 cfg={'batch_size':8,'dtype':'bfloat16','sampling_rate':16000,'greedy':True,'external_lm':False,'external_context':False}
 if model['kind']!='ctc':cfg.update(language='en',task='transcribe',num_beams=1,do_sample=False,max_new_tokens=440,return_timestamps=False)
 save(directory/'decoding.json',cfg)
 with pred.open('w') as f,torch.inference_mode():
  for start in range(0,len(rows),8):
   batch=rows[start:start+8];audios=[]
   for r in batch:
    audio,sr=sf.read(r['audio_filepath'],dtype='float32');assert sr==16000 and audio.ndim==1;audios.append(audio)
   if model['kind']=='ctc':
    inputs=processor(audios,sampling_rate=16000,return_tensors='pt',padding=True,return_attention_mask=True)
    inputs={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}
    logits=net(**inputs).logits;ids=logits.argmax(dim=-1)
    lengths=net._get_feat_extract_output_lengths(inputs['attention_mask'].sum(-1)).tolist()
    texts=[processor.decode(ids[i,:int(n)].cpu().tolist()) for i,n in enumerate(lengths)]
    capped=[False]*len(batch)
   else:
    inputs=processor(audios,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
    feats=inputs.input_features.to('cuda',dtype=torch.bfloat16);mask=inputs.attention_mask.to('cuda')
    ids=net.generate(input_features=feats,attention_mask=mask,language='en',task='transcribe',do_sample=False,num_beams=1,max_new_tokens=440,return_timestamps=False)
    texts=processor.batch_decode(ids,skip_special_tokens=True)
    capped=[net.generation_config.eos_token_id not in seq.tolist() for seq in ids]
   for r,text,cap in zip(batch,texts,capped):
    item={**r,'pred_text':text,'eos_absent_in_returned_sequence':cap};results.append(item);f.write(json.dumps(item)+'\n')
   f.flush()
   if start%80==0:print(model['slug'],split,min(start+8,len(rows)),'/',len(rows),flush=True)
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--slug',required=True);p.add_argument('--attempt',default='attempt-1');args=p.parse_args()
 root=args.run_root;cfg=json.loads((root/'models.json').read_text());model=next(m for m in cfg['models'] if m['slug']==args.slug)
 out=root/'results'/args.slug/args.attempt;out.mkdir(parents=True,exist_ok=False)
 try:
  import torch,importlib.metadata
  torch.manual_seed(20261106);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
  versions={x:importlib.metadata.version(x) for x in ['torch','transformers','nemo_toolkit','peft','safetensors']}
  hardware={'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'versions':versions}
  save(out/'environment.json',hardware)
  receipt=json.loads((root/'models'/args.slug/'download-receipt.json').read_text())
  for name,f in receipt['weights'].items():assert sha(root/'models'/args.slug/name)==f['sha256']
  test=read(root/'test.jsonl');dev=read(root/'dev.jsonl')[:8]
  assert len(test)==1908 and sum(len(normalize(r['text']).split()) for r in test)==21930
  loading=time.monotonic();meta={}
  if model['kind'].startswith('nemo'):
   nemo_infer(model,root,dev,'smoke',out);load_seconds=time.monotonic()-loading
   begin=time.monotonic();preds=nemo_infer(model,root,test,'test',out);seconds=time.monotonic()-begin
  else:
   net,processor,meta=hf_loader(model,root);hf_infer(model,net,processor,dev,out,'smoke');torch.cuda.synchronize();load_seconds=time.monotonic()-loading
   torch.cuda.reset_peak_memory_stats();begin=time.monotonic();preds=hf_infer(model,net,processor,test,out,'test');torch.cuda.synchronize();seconds=time.monotonic()-begin
   meta['peak_allocated_gpu_bytes']=torch.cuda.max_memory_allocated()
  assert len(preds)==1908 and all(a['id']==b['id'] and a['text']==b['text'] for a,b in zip(test,preds))
  audio_seconds=sum(r['duration'] for r in test);metrics=score(preds)
  result={'status':'complete','model':model,'receipt':receipt,'environment':hardware,'scores':metrics,'load_and_smoke_seconds':load_seconds,'test_wall_seconds':seconds,'test_audio_seconds':audio_seconds,'test_rtf':seconds/audio_seconds,'timing_scope':'NeMo CLI includes test model reload; HF excludes load after dev smoke. Concurrent 8-model node run.','eos_absent_returned_sequence_count':sum(r.get('eos_absent_in_returned_sequence',False) for r in preds) if model['kind'].startswith('whisper') else None,'prediction_sha256':sha(out/'test/predictions.jsonl'),'data_contract':json.loads((root/'data-contract.json').read_text()),'scorer_sha256':sha(Path(__file__).with_name('score_wer.py')),'runner_sha256':sha(__file__),'job_id':os.environ.get('SLURM_JOB_ID'),'metadata':meta}
  save(out/'summary.json',result);print('COMPLETE',args.slug,json.dumps(metrics),flush=True)
 except Exception:
  save(out/'failure.json',{'status':'failed','traceback':traceback.format_exc(),'job_id':os.environ.get('SLURM_JOB_ID')});raise
if __name__=='__main__':main()
