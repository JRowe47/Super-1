"""Matched tiny RoPE blocks with attention or a source-checked gated delta recurrence.

The chunk solver is an exact algebraic rearrangement of the sequential rule,
not a nonlinear GRU scan. State matrices have shape [batch, head, key, value].
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def delta_sequential(q,k,v,log_decay,beta,state=None):
    state=k.new_zeros(k.shape[0],k.shape[1],k.shape[-1],v.shape[-1]) if state is None else state
    outputs=[]
    for i in range(k.shape[2]):
        state=state*log_decay[:,:,i,None,None].exp()
        residual=(v[:,:,i]-(k[:,:,i,:,None]*state).sum(-2))*beta[:,:,i,None]
        state=state+k[:,:,i,:,None]*residual[:,:,None,:]
        outputs.append((q[:,:,i,:,None]*state).sum(-2))
    return torch.stack(outputs,2),state


def delta_chunk(q,k,v,log_decay,beta,state=None,chunk_size=32):
    state=k.new_zeros(k.shape[0],k.shape[1],k.shape[-1],v.shape[-1]) if state is None else state
    output=[]
    for start in range(0,k.shape[2],chunk_size):
        qc,kc,vc=q[:,:,start:start+chunk_size],k[:,:,start:start+chunk_size],v[:,:,start:start+chunk_size]
        b=beta[:,:,start:start+chunk_size];g=log_decay[:,:,start:start+chunk_size].cumsum(-1)
        size=kc.shape[2];lower=torch.ones(size,size,dtype=torch.bool,device=k.device).tril()
        ratios=(g[:,:,:,None]-g[:,:,None,:]).masked_fill(~lower,-torch.inf).exp()
        gram=(kc@kc.transpose(-1,-2))*ratios*b[:,:,:,None]
        matrix=gram.tril(-1)+torch.eye(size,device=k.device,dtype=k.dtype)
        rhs=b[:,:,:,None]*(vc-g.exp()[:,:,:,None]*(kc@state))
        updates=torch.linalg.solve_triangular(matrix,rhs,upper=False,unitriangular=True)
        output.append(g.exp()[:,:,:,None]*(qc@state)+((qc@kc.transpose(-1,-2))*ratios)@updates)
        state=g[:,:,-1,None,None].exp()*state+(kc*(g[:,:,-1,None]-g).exp()[:,:,:,None]).transpose(-1,-2)@updates
    return torch.cat(output,2),state


def rope(x,offset):
    half=x.shape[-1]//2
    frequency=torch.exp(-math.log(10000)*torch.arange(half,device=x.device,dtype=x.dtype)/half)
    angle=torch.arange(offset,offset+x.shape[2],device=x.device,dtype=x.dtype)[:,None]*frequency[None,:]
    a,b=x[...,:half],x[...,half:]
    return torch.cat((a*angle.cos()-b*angle.sin(),a*angle.sin()+b*angle.cos()),-1)


class Mixer(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.kind=kind;self.heads=c['heads'];self.chunk=c['chunk_size']
        self.qkv=nn.Linear(c['width'],3*c['width'],bias=False);self.out=nn.Linear(c['width'],c['width'],bias=False)
        if kind=='D':
            self.gates=nn.Linear(c['width'],2*c['heads']);nn.init.zeros_(self.gates.weight)
            with torch.no_grad():self.gates.bias[:c['heads']].fill_(c['decay_bias']);self.gates.bias[c['heads']:].fill_(0)
            self.norm=nn.RMSNorm(c['width']//c['heads'])
    def forward(self,x,state=None,offset=0,sequential=False):
        b,t,w=x.shape;q,k,v=[a.view(b,t,self.heads,-1).transpose(1,2) for a in self.qkv(x).chunk(3,-1)]
        if self.kind=='A':
            q,k=rope(q,offset),rope(k,offset)
            if state is not None:k=torch.cat((state[0],k),2);v=torch.cat((state[1],v),2)
            if offset:
                mask=torch.arange(k.shape[2],device=x.device)[None,:]<=torch.arange(offset,offset+t,device=x.device)[:,None]
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            else:y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            state=(k,v)
        else:
            q,k=F.normalize(q,dim=-1),F.normalize(k,dim=-1)
            gates=self.gates(x).transpose(1,2);g=F.logsigmoid(gates[:,:self.heads]);beta=gates[:,self.heads:].sigmoid()
            if sequential:y,state=delta_sequential(q,k,v,g,beta,state)
            else:y,state=delta_chunk(q,k,v,g,beta,state,self.chunk)
            y=self.norm(y)
        return self.out(y.transpose(1,2).reshape(b,t,w)),state


class Block(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.norm1=nn.LayerNorm(c['width']);self.norm2=nn.LayerNorm(c['width'])
        self.mixer=Mixer(c,kind);self.ffn=nn.Sequential(nn.Linear(c['width'],c['width']*c['ffn_multiplier'],bias=False),nn.GELU(),nn.Linear(c['width']*c['ffn_multiplier'],c['width'],bias=False))
    def forward(self,x,state=None,offset=0,sequential=False):
        y,state=self.mixer(self.norm1(x),state,offset,sequential);x=x+y;return x+self.ffn(self.norm2(x)),state


class Network(nn.Module):
    def __init__(self,c,pattern):
        super().__init__();self.pattern=pattern;self.embedding=nn.Embedding(c['vocab'],c['width']);self.blocks=nn.ModuleList([Block(c,k) for k in pattern]);self.norm=nn.LayerNorm(c['width'])
        for name,p in self.named_parameters():
            if p.ndim>=2 and 'gates' not in name:nn.init.normal_(p,0,.02)
    def forward(self,ids,state=None,sequential=False):
        states,offset=state if state is not None else ([None]*len(self.blocks),0)
        x=self.embedding(ids);result=[]
        for block,s in zip(self.blocks,states):x,s=block(x,s,offset,sequential);result.append(s)
        return F.linear(self.norm(x),self.embedding.weight),(result,offset+ids.shape[1])


def build(c,arm,seed):
    # Copy every shared parameter from the same initialized attention model.
    torch.manual_seed(seed);baseline=Network(c,c['patterns']['transformer'])
    if arm=='transformer':return baseline
    torch.manual_seed(seed);m=Network(c,c['patterns'][arm]);state=m.state_dict()
    for name,value in baseline.state_dict().items():
        if name in state and state[name].shape==value.shape:state[name]=value
    m.load_state_dict(state);return m


def tensor_bytes(tree):
    if isinstance(tree,torch.Tensor):return tree.numel()*tree.element_size()
    if isinstance(tree,(tuple,list)):return sum(tensor_bytes(v) for v in tree)
    return 0
