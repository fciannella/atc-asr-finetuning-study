"""Restore an aliased Lhotse sampler/iterator once in pinned TorchData.

Zero-worker loaders serialize the same stateful sampler twice. Deserializing
it a second time schedules a duplicate Lhotse replay. Preserve the first,
complete restore, with a strict equality guard on the redundant payload.
"""
import os

def equal(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    import torch
    import numpy as np
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    return a==b

def install():
    import torchdata.stateful_dataloader.stateful_dataloader as sd
    from lhotse.dataset.sampling.base import CutSampler
    cls=sd._StatefulSingleProcessDataLoaderIter
    if getattr(cls,'_atc_alias_fix',False):return
    original=cls.load_state_dict
    def restore(self,state):
        if self._index_sampler is self._sampler_iter and isinstance(self._index_sampler,CutSampler):
            a=state.get(sd._INDEX_SAMPLER_STATE);b=state.get(sd._SAMPLER_ITER_STATE)
            if a is not None and b is not None:
                if not equal(a,b):raise RuntimeError('Aliased Lhotse sampler has unequal serialized states; refusing ambiguous resume')
                state=dict(state);state[sd._SAMPLER_ITER_STATE]=None
                print('SAMPLER_ALIAS_RESTORE_ONCE',os.environ.get('RANK','cpu'),flush=True)
        return original(self,state)
    cls.load_state_dict=restore
    cls._atc_alias_fix=True
