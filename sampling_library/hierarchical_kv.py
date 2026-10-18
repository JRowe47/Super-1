"""Experimental hierarchical KV storage for the unchanged function-weight model.

Recent tokens remain exact. Older keys/values are merged into bounded levels
using learned, content-dependent pooling. This approximates full attention;
it does not replace function-weight projections or introduce recurrent blocks.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class KVNode:
    key: torch.Tensor
    value: torch.Tensor
    count: int


@dataclass(frozen=True)
class HierarchicalState:
    recent: tuple
    levels: tuple
    configuration: tuple


class HierarchicalKV(nn.Module):
    def __init__(self,heads,head_dim,*,recent_capacity,level_capacities,merge_factor=2):
        super().__init__()
        if not isinstance(merge_factor,int) or merge_factor<2:raise ValueError('merge_factor must be an integer >=2')
        if not isinstance(recent_capacity,int) or recent_capacity<merge_factor:raise ValueError('recent_capacity must allow a complete merge group')
        levels=tuple(level_capacities)
        if not levels or any(not isinstance(x,int) or x<merge_factor for x in levels):raise ValueError('Each hierarchy level must allow a complete merge group')
        if heads<1 or head_dim<1:raise ValueError('Head dimensions must be positive')
        self.heads=heads;self.head_dim=head_dim;self.recent_capacity=recent_capacity;self.level_capacities=levels;self.merge_factor=merge_factor
        # Learned per-head scoring directions; zero is count-weighted mean pooling.
        self.pool_query=nn.Parameter(torch.zeros(len(levels)+1,heads,head_dim))
    def configuration(self):return (self.heads,self.head_dim,self.recent_capacity,tuple(self.level_capacities),self.merge_factor)
    @property
    def capacity(self):return self.recent_capacity+sum(self.level_capacities)
    def _merge(self,nodes,level):
        keys=torch.cat([n.key for n in nodes],2);values=torch.cat([n.value for n in nodes],2)
        count_dtype=torch.float64 if keys.dtype==torch.float64 else torch.float32
        log_counts=torch.tensor([n.count for n in nodes],device=keys.device,dtype=count_dtype).log().to(keys.dtype)
        scores=(keys*self.pool_query[level][None,:,None,:]).sum(-1)/math.sqrt(self.head_dim)+log_counts[None,None,:]
        weights=scores.softmax(-1).unsqueeze(-1)
        return KVNode((keys*weights).sum(2,keepdim=True),(values*weights).sum(2,keepdim=True),sum(n.count for n in nodes))
    def _append(self,recent,levels,node):
        recent.append(node)
        if len(recent)>self.recent_capacity:
            group=recent[:self.merge_factor];del recent[:self.merge_factor];levels[0].append(self._merge(group,0))
        for level,limit in enumerate(self.level_capacities):
            if len(levels[level])>limit:
                group=levels[level][:self.merge_factor];del levels[level][:self.merge_factor]
                merged=self._merge(group,level+1)
                if level+1<len(levels):levels[level+1].append(merged)
                else:levels[level].insert(0,merged)
    def forward(self,q,k,v,state=None):
        if q.shape!=k.shape or k.shape!=v.shape or q.ndim!=4:raise ValueError('Expected equal [batch,head,time,head_dim] Q/K/V shapes')
        if q.shape[1]!=self.heads or q.shape[-1]!=self.head_dim:raise ValueError('Memory and projection head dimensions differ')
        config=self.configuration()
        if state is not None and (not isinstance(state,HierarchicalState) or state.configuration!=config):raise ValueError('Hierarchy configuration changed; prefill the complete prefix again')
        recent=list(state.recent) if state is not None else []
        levels=[list(x) for x in state.levels] if state is not None else [[] for _ in self.level_capacities]
        outputs=[]
        # Update/read chronologically even during prefill: future tokens cannot
        # alter summaries used for earlier queries. Chunk boundaries are inert.
        for i in range(q.shape[2]):
            self._append(recent,levels,KVNode(k[:,:,i:i+1].clone(),v[:,:,i:i+1].clone(),1))
            nodes=[n for level in reversed(levels) for n in level]+recent
            keys=torch.cat([n.key for n in nodes],2);values=torch.cat([n.value for n in nodes],2)
            count_dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
            mass=torch.tensor([n.count for n in nodes],device=q.device,dtype=count_dtype).log().to(q.dtype)[None,None,None,:]
            outputs.append(F.scaled_dot_product_attention(q[:,:,i:i+1],keys,values,attn_mask=mass))
        return torch.cat(outputs,2),HierarchicalState(tuple(recent),tuple(tuple(x) for x in levels),config)
