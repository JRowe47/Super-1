"""Exact full-dimensional weight-norm profiles, using small Gram matrices."""
import math
import numpy as np
import torch
from common import OUT,dump
from curve_model import key
from joint_model import derivative_basis,completed_basis

@torch.no_grad()
def profiles(banks):
    result={};ts=np.linspace(0,3,151)
    for label in ['initial','orthogonal_curve','lifted_dense']:
        result[label]={}
        for start,b in banks[label].items():
            rank=next(iter(b.delta.values())).shape[0];dim=8+rank
            gram=torch.zeros((dim,dim),dtype=torch.float64)
            for name in b.names:
                vectors=[b.initial.knots[i][name].flatten() for i in range(4)]
                vectors += [b.initial.slopes[key(name)][i].flatten() for i in range(4)]
                vectors += [b.delta[key(name)][i].flatten() for i in range(rank)]
                x=torch.stack(vectors).double();gram+=x@x.T
            rows=[]
            for t in ts:
                c=np.zeros(dim)
                if abs(t-round(t))<1e-12:c[round(t)]=1
                else:
                    j=math.floor(t);u=t-j;c[j]=2*u**3-3*u**2+1;c[j+1]=-2*u**3+3*u**2
                    c[4:8]=derivative_basis(t)
                    c[8:]=completed_basis(t) if b.kind=='complete_curve' else derivative_basis(t)@b.transform
                ct=torch.tensor(c,dtype=torch.float64)
                rows.append({'depth':float(t)+int(start),'parameter_vector_norm':float((ct@gram@ct).clamp_min(0).sqrt())})
            result[label][start]=rows
    dump(OUT/'weight_profiles.json',{'curves':result,'meaning':'Exact norm of all nine parameter tensors; a scalar projection of the full curve, not an activation trajectory or proof of smooth language behavior.'})
