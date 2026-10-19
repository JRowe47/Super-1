"""Exact full-dimensional curve norms from small parameter Gram matrices."""
import math
import numpy as np
import torch
from common import OUT,dump
from curve_model import key
from models import RIGHT_INVERSE,TIMES

def coefficients(t,order=0):
    j=min(math.floor(t),2);u=t-j;c=np.zeros(11)
    if order==0:
        c[j]=2*u**3-3*u**2+1;c[j+1]=-2*u**3+3*u**2
        c[4+j]=u**3-2*u**2+u;c[5+j]=u**3-u**2
    elif order==1:
        c[j]=6*u*u-6*u;c[j+1]=-6*u*u+6*u
        c[4+j]=3*u*u-4*u+1;c[5+j]=3*u*u-2*u
    else:
        c[j]=12*u-6;c[j+1]=-12*u+6
        c[4+j]=6*u-4;c[5+j]=6*u-2
    c[8:]=c[4:8]@RIGHT_INVERSE
    return torch.tensor(c,dtype=torch.float64)

@torch.no_grad()
def curve_profiles(banks):
    results={};summaries={};ts=np.unique(np.r_[np.linspace(0,3,151),TIMES])
    for kind,windows in banks.items():
        results[kind]={};summaries[kind]={}
        for start,bank in windows.items():
            gram=torch.zeros((11,11),dtype=torch.float64)
            for name in bank.names:
                vectors=[bank.initial.knots[j][name].flatten() for j in range(4)]
                vectors += [bank.initial.slopes[key(name)][j].flatten() for j in range(4)]
                vectors += [bank.delta[key(name)][j].flatten() for j in range(3)]
                matrix=torch.stack(vectors).double();gram+=matrix@matrix.T
            rows=[]
            for t in ts:
                c=coefficients(t);dc=coefficients(t,1);ddc=coefficients(t,2)
                increment=c.clone();increment[:8]=0
                rows.append({'depth':float(t)+int(start),
                    'parameter_vector_norm':float((c@gram@c).clamp_min(0).sqrt()),
                    'departure_from_phase4_norm':float((increment@gram@increment).clamp_min(0).sqrt()),
                    'speed_norm':float((dc@gram@dc).clamp_min(0).sqrt()),
                    'second_derivative_squared_norm':float((ddc@gram@ddc).clamp_min(0))})
            results[kind][start]=rows
            summaries[kind][start]={'parameter_space_arc_length':float(np.trapezoid([r['speed_norm'] for r in rows],ts)),
                'integrated_second_derivative_energy':float(np.trapezoid([r['second_derivative_squared_norm'] for r in rows],ts)),
                'max_departure_from_phase4_norm':max(r['departure_from_phase4_norm'] for r in rows)}
    dump(OUT/'weight_profiles.json',{'curves':results,'summaries':summaries,
        'meaning':'Norms and derivatives use all nine parameter tensors per block. Scalar projections of weight curves; no assertion that arc length measures useful information or language behavior.',
        'numerics':'Float64 Gram matrices; 151 evenly spaced depth samples plus fitted sample points. Arc length and second-derivative energy use trapezoidal quadrature; piecewise second derivatives may jump at knots.'})
