"""Matched native-block candidates and an exact three-sample curve lift."""
import copy,math
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from common import ROOT,WINDOWS,KEEP
from curve_model import HermiteBank,CurveBlock,key,coordinates

TIMES=coordinates(3)

def phase4_banks(original):
    banks=nn.ModuleDict()
    for start in WINDOWS:
        bank=HermiteBank(original[start:start+4])
        bank.slopes.load_state_dict(load_file(str(ROOT/'reference/phase4_checkpoints'/f'window_{start}.safetensors')))
        bank.requires_grad_(False);banks[str(start)]=bank
    return banks

def candidate(original,kind,scale=1.,trainable=False,checkpoint=None):
    banks=phase4_banks(original) if kind=='interpolated' else None
    blocks=nn.ModuleDict()
    for start in WINDOWS:
        row=nn.ModuleList()
        for i,t in enumerate(TIMES):
            block=copy.deepcopy(original[start+KEEP[i]])
            with torch.no_grad():
                if banks is not None:
                    sampled=banks[str(start)].sample(t)
                    for name,p in block.named_parameters():p.copy_(sampled[name])
                if scale!=1:
                    block.self_attn.o_proj.weight.mul_(scale)
                    block.mlp.down_proj.weight.mul_(scale)
            block.requires_grad_(trainable);row.append(block)
        blocks[str(start)]=row
    if checkpoint is not None:blocks.load_state_dict(load_file(str(checkpoint)))
    return blocks

def install(model,original,blocks,only=None):
    active=WINDOWS if only is None else only
    layers=[];i=0
    while i<len(original):
        if i in active:
            layers.extend(blocks[str(i)]);i+=4
        else:layers.append(original[i]);i+=1
    for j,block in enumerate(layers):block.self_attn.layer_idx=j
    model.model.layers=nn.ModuleList(layers)
    model.config.num_hidden_layers=len(layers)
    return layers

def restore(model,original):
    model.model.layers=nn.ModuleList(original);model.config.num_hidden_layers=len(original)
    for j,block in enumerate(original):block.self_attn.layer_idx=j

def save_blocks(blocks,path):
    save_file({name:p.detach().contiguous() for name,p in blocks.state_dict().items()},str(path))

def derivative_basis(t):
    a=np.zeros(4,dtype=np.float64)
    if abs(t-round(t))<1e-12:return a
    j=math.floor(t);u=t-j
    a[j]=u**3-2*u**2+u;a[j+1]=u**3-u**2
    return a

SAMPLE_MATRIX=np.stack([derivative_basis(t) for t in TIMES])
RIGHT_INVERSE=np.linalg.pinv(SAMPLE_MATRIX)

def cardinal_basis(t):
    return derivative_basis(t)@RIGHT_INVERSE

class LiftedBank(nn.Module):
    """C(t)=phase4(t)+sum_i l_i(t) D_i; l_i(native knots)=0.

    A A+ = I_3, so each D_i is the target-minus-base tensor at sample i.
    Three full-rank sample constraints need no fifth mode in this phase,
    which studies only N=3. N=2 quality is not assumed or exposed by loader.
    """
    def __init__(self,initial,blocks):
        super().__init__();object.__setattr__(self,'initial',initial)
        self.names=initial.names;self.delta=nn.ParameterDict()
        with torch.no_grad():
            base=[initial.sample(t) for t in TIMES]
            target=[dict(block.named_parameters()) for block in blocks]
            for name in self.names:
                self.delta[key(name)]=nn.Parameter(torch.stack([target[i][name]-base[i][name] for i in range(3)]),requires_grad=False)

    def sample(self,t):
        if not 0<=t<=3:raise ValueError('Depth outside the four original knots')
        if abs(t-round(t))<1e-12:return self.initial.knots[round(t)]
        result={};coeff=cardinal_basis(t);base=self.initial.sample(t)
        for name in self.names:
            value=base[name]
            for i,c in enumerate(coeff):
                if abs(c)>1e-12:value=value+float(c)*self.delta[key(name)][i]
            result[name]=value
        return result

def lift(original,blocks):
    initial=phase4_banks(original)
    return nn.ModuleDict({str(s):LiftedBank(initial[str(s)],blocks[str(s)]) for s in WINDOWS})

def install_curve(model,original,banks,steps=3):
    if steps not in [3,4]:raise ValueError('Only trained three-step and original four-step schedules are supported')
    layers=[];i=0
    while i<len(original):
        if i not in WINDOWS:
            block=original[i];block.self_attn.layer_idx=len(layers);layers.append(block);i+=1
        else:
            for t in coordinates(steps):layers.append(CurveBlock(model.config,banks[str(i)],t,1.,len(layers)))
            i+=4
    model.model.layers=nn.ModuleList(layers);model.config.num_hidden_layers=len(layers)
    return layers

def load_candidate(kind='retained',as_curve=False,steps=3):
    if kind not in ['interpolated','retained']:raise ValueError('Unknown initialization')
    model=load_model_for_loader();original=list(model.model.layers)
    blocks=candidate(original,kind,checkpoint=ROOT/'checkpoints'/f'{kind}.safetensors')
    if as_curve:
        banks=lift(original,blocks);install_curve(model,original,banks,steps)
        object.__setattr__(model,'phase6_original_blocks',original)
        object.__setattr__(model,'phase6_banks',banks)
    elif steps==3:install(model,original,blocks)
    elif steps!=4:raise ValueError('Unsupported schedule')
    return model

def load_model_for_loader():
    from common import load_model
    return load_model()
