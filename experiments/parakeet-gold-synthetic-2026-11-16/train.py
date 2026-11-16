"""Pinned NeMo TDT refinement with finite-gradient and checkpoint gates."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,json,os,time
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf,open_dict
from nemo.collections.asr.models import ASRModel
R=Path(__file__).resolve().parent

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def domain_weights(model):
 out={}
 for domain in ['encoder','decoder','joint']:
  digest=hashlib.sha256();count=0
  for name,p in model.named_parameters():
   if name.startswith(domain+'.') and p.requires_grad:
    digest.update(name.encode());digest.update(p.detach().float().cpu().contiguous().numpy().tobytes());count+=p.numel()
  assert count>0
  out[domain]={'elements_hashed':count,'sha256':digest.hexdigest()}
 return out
class Checks(Callback):
 def __init__(self,root,end,keeps,resume):
  self.root=root;self.end=end;self.keeps=keeps;self.resume=resume;self.finite_steps=0;self.validations=[];self.saved=[]
 def state_dict(self):return {'finite_steps':self.finite_steps,'validations':self.validations,'saved':self.saved}
 def load_state_dict(self,state):
  for k,v in state.items():setattr(self,k,v)
 def on_train_start(self,trainer,model):
  assert trainer.world_size==8
  assert trainer.global_step==self.resume and self.finite_steps==self.resume
  if self.resume:assert len(trainer.optimizers[0].state)>0 and trainer.lr_scheduler_configs
  self.initial=domain_weights(model);self.started=time.monotonic()
  if trainer.is_global_zero:save(self.root/('resume-%d.json'%self.resume),{'status':'passed','global_step':trainer.global_step,'world_size':trainer.world_size,'optimizer_states':len(trainer.optimizers[0].state)})
 def on_before_optimizer_step(self,trainer,model,optimizer):
  flags=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
  assert flags
  ok=torch.stack(flags).all().to(torch.int32);torch.distributed.all_reduce(ok,op=torch.distributed.ReduceOp.MIN)
  if not ok.item():raise RuntimeError('Non-finite gradient on at least one rank')
  self.finite_steps+=1
 def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
  loss=outputs['loss'] if isinstance(outputs,dict) else outputs
  if loss is not None:
   loss=loss.detach().float();torch.distributed.all_reduce(loss);loss/=trainer.world_size
   assert torch.isfinite(loss).item()
  if trainer.is_global_zero and trainer.global_step%10==0:
   save(self.root/'status.json',{'phase':'training','job':os.environ.get('SLURM_JOB_ID'),'step':trainer.global_step,'target':self.end,'world_size':8,'finite_gradient_steps':self.finite_steps,'mean_rank_loss':float(loss) if loss is not None else None,'updated_at':datetime.now(timezone.utc).isoformat()})
 def on_validation_end(self,trainer,model):
  if trainer.sanity_checking:return
  metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if k.startswith('val_') and torch.is_tensor(v) and v.numel()==1}
  self.validations.append({'step':trainer.global_step,'metrics':metrics})
  if trainer.global_step in self.keeps and trainer.global_step not in self.saved:
   assert 'val_wer' in metrics,metrics
   self.saved.append(trainer.global_step)
   trainer.save_checkpoint(str(self.root/'checkpoints'/('step=%d.ckpt'%trainer.global_step)))
   trainer.strategy.barrier()
   if trainer.is_global_zero:save(self.root/'checkpoints.json',{'steps':self.saved,'validations':self.validations})
 def on_train_end(self,trainer,model):
  assert trainer.global_step==self.end==self.finite_steps
  assert self.end in self.saved and self.validations
  final=domain_weights(model)
  changed={k:self.initial[k]['sha256']!=final[k]['sha256'] for k in final}
  assert all(changed.values()),changed
  if trainer.is_global_zero:save(self.root/('complete-%d.json'%self.end),{'status':'passed','step':self.end,'world_size':8,'finite_gradient_steps':self.finite_steps,'checkpoint_saved':True,'changed_parameter_domains':changed,'validation_events':self.validations,'elapsed_seconds':time.monotonic()-self.started,'quality_claim':None})
 def on_exception(self,trainer,model,exception):
  if trainer.is_global_zero:save(self.root/'failure.json',{'error':repr(exception),'step':trainer.global_step,'job':os.environ.get('SLURM_JOB_ID')})

def main():
 p=argparse.ArgumentParser();p.add_argument('arm',choices=['p_g3','p_g2']);p.add_argument('--end',type=int,default=2000);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=int,default=0);a=p.parse_args()
 contract=json.loads((R/'contract.json').read_text());assert contract['status']=='passed';assert a.arm in contract['eligible_training_arms']
 role='silver' if a.arm=='p_g3' else 'base';parent=contract['models'][role]['path']
 root=R/'arms'/a.arm/('smoke' if a.smoke else 'train');(root/'checkpoints').mkdir(parents=True,exist_ok=True)
 full=40 if a.smoke else 2000;keeps=[20,40] if a.smoke else contract['training']['keep_steps']
 assert a.end<=full and (a.end in keeps)
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']));pl.seed_everything(42,workers=True);torch.set_float32_matmul_precision('high')
 trainer=pl.Trainer(accelerator='gpu',devices=8,num_nodes=1,strategy=DDPStrategy(timeout=timedelta(minutes=30)),precision='32-true',max_steps=a.end,max_epochs=-1,limit_train_batches=1000,val_check_interval=10 if a.smoke else 250,check_val_every_n_epoch=None,num_sanity_val_steps=0,use_distributed_sampler=False,sync_batchnorm=True,gradient_clip_val=1.0,accumulate_grad_batches=1,enable_checkpointing=False,enable_progress_bar=False,log_every_n_steps=10,logger=CSVLogger(str(root),'metrics',version='v1'),callbacks=[Checks(root,a.end,keeps,a.resume)])
 model=ASRModel.restore_from(parent,map_location='cpu');model.set_trainer(trainer)
 assert model.cfg.loss.loss_name=='tdt' and model.cfg.decoding.model_type=='tdt'
 assert list(model.cfg.loss.tdt_kwargs.durations)==list(model.cfg.decoding.durations)
 model.joint.set_fuse_loss_wer(False)
 with open_dict(model.cfg):model.cfg.joint.fuse_loss_wer=False;model.cfg.skip_nan_grad=False
 config=OmegaConf.load('/opt/prompt_model/NeMo/examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml')
 train=OmegaConf.create({'manifest_filepath':None,'input_cfg':[{'type':'nemo','manifest_filepath':str(R/'inputs/gold-train.jsonl'),'weight':.85,'tags':{'domain':'gold_atco2'}},{'type':'nemo','manifest_filepath':str(R/'inputs/english-train.jsonl'),'weight':.15,'tags':{'domain':'general_english'}}],'tarred_audio_filepaths':None,'use_lhotse':True,'use_bucketing':False,'batch_size':2,'batch_duration':None,'quadratic_duration':None,'bucket_duration_bins':None,'bucket_batch_size':None,'batch_tokens':None,'shuffle_buffer_size':2048,'seed':42,'shard_seed':42,'sample_rate':16000,'max_duration':30.,'min_duration':.1,'text_field':'text','num_workers':2,'shuffle':True,'pin_memory':True})
 val=OmegaConf.create({'manifest_filepath':str(R/'inputs/development.jsonl'),'use_lhotse':True,'use_bucketing':False,'batch_size':16,'batch_duration':None,'quadratic_duration':None,'bucket_duration_bins':None,'bucket_batch_size':None,'batch_tokens':None,'shuffle':False,'sample_rate':16000,'text_field':'text','num_workers':2,'min_duration':0.,'max_duration':None})
 optim=OmegaConf.create({'name':'adamw','lr':1e-5,'betas':[.9,.98],'weight_decay':.001,'sched':{'name':'CosineAnnealing','max_steps':full,'warmup_steps':2 if a.smoke else 100,'min_lr':1e-6}})
 model.setup_training_data(train);model.setup_multiple_validation_data(val);model.setup_optimization(optim)
 model.spec_augmentation=ASRModel.from_config_dict(OmegaConf.create({'_target_':'nemo.collections.asr.modules.SpectrogramAugmentation','freq_masks':2,'time_masks':10,'freq_width':27,'time_width':.05}))
 if trainer.is_global_zero:save(root/'config.json',{'parent':parent,'parent_sha256':contract['models'][role]['sha256'],'seed':42,'scheduler_steps':full,'training':OmegaConf.to_container(train,resolve=True),'validation':OmegaConf.to_container(val,resolve=True),'optimizer':OmegaConf.to_container(optim,resolve=True),'keep_steps':keeps,'precision':'32-true','world_size':8})
 checkpoint=str(root/'checkpoints'/('step=%d.ckpt'%a.resume)) if a.resume else None
 trainer.fit(model,ckpt_path=checkpoint)
if __name__=='__main__':main()
