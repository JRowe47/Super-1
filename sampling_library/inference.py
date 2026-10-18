"""Cached inference over the existing learned weight curve and token router.

Only attention storage/readout is delegated to a memory backend. Projection
bases, coefficient curves, normalization blocks, router and MLP stay intact.
"""
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Tuple
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InferenceCache:
    layers: Tuple[Any, ...]
    offset: int
    depth: int
    batch: int
    policy: tuple
    signature: tuple
    owner: object


def cache_bytes(value):
    """Logical tensor bytes, excluding model weights and allocator overhead."""
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,InferenceCache):return cache_bytes(value.layers)
    if is_dataclass(value):return sum(cache_bytes(getattr(value,f.name)) for f in fields(value))
    if isinstance(value,(tuple,list)):return sum(cache_bytes(x) for x in value)
    return 0


class ExactKV(nn.Module):
    """Full causal KV history; a reference backend with no learned parameters."""
    def forward(self,q,k,v,state=None):
        offset=0 if state is None else state[0].shape[2]
        if state is not None:k=torch.cat((state[0],k),2);v=torch.cat((state[1],v),2)
        if offset:
            mask=torch.arange(k.shape[2],device=q.device)[None,:]<=torch.arange(offset,offset+q.shape[2],device=q.device)[:,None]
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        else:out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return out,(k,v)


def rotary(x,offset):
    # Match the original interleaved RoPE, including its float32 frequency path.
    d=x.shape[-1]
    inv=1.0/(10000.0**(torch.arange(0,d,2,device=x.device).float()/d))
    positions=torch.arange(offset,offset+x.shape[2],device=x.device).float()
    angles=torch.outer(positions,inv)
    cos,sin=angles.cos().to(x.dtype)[None,None],angles.sin().to(x.dtype)[None,None]
    result=torch.empty_like(x);a,b=x[...,0::2],x[...,1::2]
    result[...,0::2]=a*cos-b*sin;result[...,1::2]=a*sin+b*cos
    return result


class CachedSampledCurve(nn.Module):
    """Wrap a loaded SampledCurve without replacing or copying its weights.

    The loaded model/profile selects the executed depth for a request. The
    learned token-conditioned sampling coordinates remain dynamic within it.
    Changing depth, router weights or policy requires starting with cache=None
    and prefilling the complete prefix again.
    """
    def __init__(self,model,memory_factory=None):
        super().__init__()
        for name in ('base','router','projection','depth','mode'):
            if not hasattr(model,name):raise TypeError('Expected a function-weight SampledCurve model')
        if model.mode not in ('fixed','global','token'):raise ValueError('Unknown sampling policy')
        for block in model.base.blocks:
            for name in ('q','k','v','o','f1','f2'):
                projection=getattr(block,name)
                if not all(hasattr(projection,k) for k in ('B','p','DEG')):
                    raise TypeError('All projections must retain their learned function-weight bases')
        self.model=model
        self.memory=nn.ModuleList([ExactKV() if memory_factory is None else memory_factory(i,b.nh,b.d//b.nh) for i,b in enumerate(model.base.blocks)])
        self._cache_owner=object()
        self.train(model.training)
    def _signature(self):
        return tuple((id(p),p._version,str(p.device),str(p.dtype)) for p in self.parameters())
    def forward(self,ids,cache=None,*,positions=False,uniform=False):
        if ids.ndim!=2 or ids.shape[1]==0:raise ValueError('Expected a nonempty [batch,tokens] tensor')
        model=self.model;depth=model.depth
        if not isinstance(depth,int) or not 2<=depth<=len(model.base.blocks):raise ValueError('Invalid executed sampling depth')
        if hasattr(model,'execution_depth') and model.execution_depth!=depth:raise ValueError('Profile depth metadata is inconsistent')
        policy=(model.mode,bool(uniform));signature=self._signature()
        if cache is not None:
            if not isinstance(cache,InferenceCache) or cache.owner is not self._cache_owner:raise ValueError('Cache belongs to a different runtime; prefill the prefix again')
            if cache.depth!=depth or cache.policy!=policy or cache.signature!=signature:raise ValueError('Depth, policy or weights changed; prefill the complete prefix again')
            if cache.batch!=ids.shape[0] or len(cache.layers)!=depth:raise ValueError('Cache batch or layer count mismatch')
        offset=0 if cache is None else cache.offset
        previous=(None,)*depth if cache is None else cache.layers
        x=model.base.tok(ids);t=x.new_zeros(ids.shape);sampled=[];states=[]
        for i,block in enumerate(model.base.blocks[:depth]):
            h=block.ln1(x)
            if 0<i<depth-1:
                condition=h if model.mode=='token' else torch.zeros_like(h)
                delta=model.router(torch.cat((condition,t.unsqueeze(-1)),dim=-1)).squeeze(-1)
                if model.mode=='fixed' or uniform:delta=delta*0
                fraction=torch.sigmoid(delta-x.new_tensor(depth-i-1).log())
                t=t+(1-t)*fraction
            elif i==depth-1:t=torch.ones_like(t)
            sampled.append(t)
            batch,length,width=x.shape
            def heads(projection):return model.projection(projection,h,t).view(batch,length,block.nh,-1).transpose(1,2)
            q,k,v=heads(block.q),heads(block.k),heads(block.v)
            q,k=rotary(q,offset),rotary(k,offset)
            a,state=self.memory[i](q,k,v,previous[i]);states.append(state)
            x=x+model.projection(block.o,a.transpose(1,2).reshape(batch,length,width),t)
            x=x+model.projection(block.f2,F.gelu(model.projection(block.f1,block.ln2(x),t)),t)
        logits=model.base.head(model.base.lnf(x))
        updated=InferenceCache(tuple(states),offset+ids.shape[1],depth,ids.shape[0],policy,signature,self._cache_owner)
        return (logits,updated,torch.stack(sampled,dim=-1)) if positions else (logits,updated)
