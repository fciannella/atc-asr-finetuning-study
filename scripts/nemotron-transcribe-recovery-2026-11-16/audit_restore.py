import copy,hashlib,json,sys,time
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_sampler_from_config,make_structured_with_schema_warnings
from lhotse.utils import fix_random_seed
from torchdata.stateful_dataloader import StatefulDataLoader
from packing_fast import install as packing_install
packing_install()
R=Path(__file__).parent;C=Path(sys.argv[1]);rank=int(sys.argv[2]);count=40
cfg=OmegaConf.load(C/'silver/full.yaml')
tokenizer=AutoTokenizer(cfg.model.tokenizer_path,use_fast=True,trust_remote_code=False,pad_token=cfg.model.get('pad_token',None))
tokenizer.add_special_tokens({'additional_special_tokens':[cfg.model.audio_locator_tag]})
class Identity:
    def __getitem__(self,index):return index

def build(packed=True):
    fix_random_seed(42)
    data=make_structured_with_schema_warnings(OmegaConf.create(OmegaConf.to_container(cfg.data.train_ds,resolve=True)))
    if not packed:
        data.use_packed_sequence_sampling=False;data.batch_tokens=None;data.batch_size=2
    sampler,iterable=get_lhotse_sampler_from_config(config=data,global_rank=rank,world_size=8,tokenizer=tokenizer)
    assert not iterable
    loader=StatefulDataLoader(Identity(),sampler=sampler,batch_size=None,num_workers=0,collate_fn=lambda x:x)
    return sampler,loader

def fingerprint(batch):
    rows=[]
    for x in batch:
        audio=[]
        for turn in getattr(x,'turns',[]):
            cut=getattr(turn,'cut',None)
            if cut is not None:audio.append([cut.recording.id,cut.start,cut.duration,cut.channel])
        tokens={k:getattr(x,k).tolist() for k in ['input_ids','context_ids','answer_ids'] if getattr(x,k,None) is not None}
        rows.append([x.id,audio,tokens])
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
def take(it,n):return [fingerprint(next(it)) for _ in range(n)]
# Observe the real bug before patching, without spending GPUs or loading audio.
s,l=build();it=iter(l);prefix=take(it,count);saved=copy.deepcopy(l.state_dict());expected=take(it,20)
u,ul=build();ul.load_state_dict(copy.deepcopy(saved));ui=iter(ul)
assert u._needs_fast_forward is True,'Expected pinned-runtime duplicate restore was not reproduced'
assert take(ui,20)==expected
assert u._needs_fast_forward is True
from sampler_restore import install
install()
reports=[]
for packed in [True,False]:
    s,l=build(packed);it=iter(l);prefix=take(it,count);saved=copy.deepcopy(l.state_dict());expected=take(it,20)
    r,rl=build(packed);calls=[];original=r._fast_forward
    def replay():
        calls.append(time.monotonic());return original()
    r._fast_forward=replay
    rl.load_state_dict(copy.deepcopy(saved));ri=iter(rl)
    assert len(calls)==1 and not r._needs_fast_forward
    assert take(ri,20)==expected
    assert not r._needs_fast_forward
    r.set_epoch(1);assert r.epoch==1
    take(iter(rl),5);assert len(calls)==1
    # Unequal dual states must fail closed.
    bad=copy.deepcopy(saved);bad['_sampler_iter_state']['epoch']+=1
    bs,bl=build(packed);bl.load_state_dict(bad)
    try:iter(bl)
    except RuntimeError as e:assert 'unequal serialized states' in str(e)
    else:raise AssertionError('Expected ambiguous-state rejection')
    reports.append({'packed':packed,'exact_continuation_batches':20,'restore_count':len(calls),'epoch_transition_replays':0,'prefix_sha256':hashlib.sha256(json.dumps(prefix).encode()).hexdigest()})
# Ordinary distinct sampler/iterator remains handled by TorchData.
plain=StatefulDataLoader(list(range(100)),batch_size=3,num_workers=0,shuffle=False)
pit=iter(plain);next(pit);st=copy.deepcopy(plain.state_dict());want=next(pit)
p2=StatefulDataLoader(list(range(100)),batch_size=3,num_workers=0,shuffle=False);p2.load_state_dict(st)
assert next(iter(p2)).equal(want)
report={'status':'passed','rank':rank,'world_size':8,'bug_reproduced':True,'packed_and_unpacked':reports,'distinct_iterator_regression':'passed','unequal_state_guard':'passed'}
(R/('restore-audit-rank-%d.json'%rank)).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
