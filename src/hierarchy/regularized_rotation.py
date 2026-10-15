"""Separate energy-regularized candidate; does not alter prior experiment."""
import importlib.util
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('regularized_reference', ROOT/'packed_rotation.py')
reference=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=reference
spec.loader.exec_module(reference)

class RegularizedRotation(nn.Module):
    def __init__(self,base,cross_scale=True):
        super().__init__()
        if base.width % 2 or base.state_scales < 2:
            raise ValueError('Even width and multiple scales required')
        self.base=base
        self.cross_scale=cross_scale
        self.rotation_gain=nn.Parameter(torch.zeros(base.state_scales))
        self.log_radius=nn.Parameter(torch.full((base.state_scales,),math.log(math.expm1(1.0))))

    def rotate(self,state):
        pairs=state.reshape(*state.shape[:-1],-1,2)
        other=pairs.roll(1,dims=1) if self.cross_scale else pairs
        numerator=pairs[...,0]*other[...,1]-pairs[...,1]*other[...,0] if self.cross_scale else (pairs*other).sum(-1)
        # At initialization radius=1. Positivity is structural; a learned
        # radius can still become small, so no global stability claim follows.
        radius=F.softplus(self.log_radius)[None,:,None]
        query=numerator/(radius+pairs.square().sum(-1)+other.square().sum(-1))
        angle=torch.tanh(self.rotation_gain)[None,:,None]*query
        c,s=torch.cos(angle),torch.sin(angle)
        return torch.stack([c*pairs[...,0]-s*pairs[...,1],s*pairs[...,0]+c*pairs[...,1]],-1).flatten(-2)

    def forward(self,tokens,state=None,return_state=False):
        core=self.base.core
        embedded=core.embedding(tokens)
        if state is None:
            state=self.base.initial_state(tokens.shape[0])
        self.base._validate_state(state,embedded)
        input_terms=F.linear(embedded,torch.cat([core.input_update.weight,core.input_gate.weight]),torch.cat([core.input_update.bias,core.input_gate.bias]))
        inputs_update,inputs_gate=input_terms.split(self.base.width,-1)
        recurrent_weight=torch.cat([core.state_update.weight,core.state_gate.weight])
        scale=core.scale[None]
        means=[]
        for index in range(tokens.shape[1]):
            ru,rg=F.linear(state,recurrent_weight).split(self.base.width,-1)
            coupled=core.coupling(state.mean(1,keepdim=True))
            update=torch.tanh(inputs_update[:,index,None]+ru+coupled+scale)
            gate=torch.sigmoid(inputs_gate[:,index,None]+rg-scale)
            state=self.rotate(gate*state+(1-gate)*update)
            means.append(state.mean(1))
        logits=self.base.output(core.norm(torch.stack(means,1)))
        return (logits,state) if return_state else logits
