"""Function-matched joint-window controls; same pretrained starting functions.

Raw and orthogonal banks span exactly the same Hermite derivative space.
Dense controls hold five independent sampled tensors (three-step + two-step).
"""
import math
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from common import ROOT,WINDOWS
from curve_model import HermiteBank,key,coordinates,install,restore

TIMES=coordinates(3)+coordinates(2)

def derivative_basis(t):
    a=np.zeros(4,dtype=np.float64)
    if abs(t-round(t))<1e-12:return a
    j=math.floor(t);u=t-j;a[j]=u**3-2*u**2+u;a[j+1]=u**3-u**2
    return a

def basis_analysis():
    a=np.stack([derivative_basis(t) for t in TIMES]);u,s,vh=np.linalg.svd(a,full_matrices=True)
    # Fix SVD signs for a deterministic, readable basis convention.
    for j in range(4):
        if u[np.argmax(np.abs(u[:,j])),j]<0:u[:,j]*=-1;vh[j]*=-1
    transform=vh.T@np.diag(1/s)
    return a,transform,u[:,:4],s,u[:,4]

def bubble(t):
    return float(np.prod([t-j for j in range(4)])**2)

def completed_basis(t):
    """Five sample-orthonormal functions, all zero at the four fixed knots.

    Four span the original derivative space. The fifth is a polynomial bubble
    with its projection onto that space subtracted; on the five tested samples
    it equals the missing left-null direction.
    """
    a,b,u,_,left=basis_analysis();q=np.asarray([bubble(x) for x in TIMES])
    denominator=float(left@q)
    r=(bubble(t)-derivative_basis(t)@np.linalg.pinv(a)@q)/denominator
    return np.r_[derivative_basis(t)@b,r]


class DeltaBank(nn.Module):
    def __init__(self,original,start,kind):
        super().__init__();self.kind=kind;self.start=start
        initial=HermiteBank(original[start:start+4]);initial.slopes.load_state_dict(
            load_file(str(ROOT/'reference/phase4_checkpoints'/f'window_{start}.safetensors')))
        initial.requires_grad_(False)
        object.__setattr__(self,'initial',initial)
        self.names=initial.names
        with torch.no_grad():self.base=[{n:p.detach().clone() for n,p in initial.sample(t).items()} for t in TIMES]
        a,b,u,s,null=basis_analysis()
        self.transform=np.eye(4) if kind=='raw_curve' else b
        self.delta=nn.ParameterDict()
        if kind=='dense':
            # Separate Parameters are essential: inactive resolutions must have
            # grad=None, so Adam does not advance their moments on other steps.
            for i in range(5):
                for name in self.names:self.delta[f'{key(name)}__sample{i}']=nn.Parameter(torch.zeros_like(self.base[i][name]))
        else:
            rank=5 if kind=='complete_curve' else 4
            for name in self.names:self.delta[key(name)]=nn.Parameter(torch.zeros((rank,*self.base[0][name].shape)))

    def sample(self,t):
        if abs(t-round(t))<1e-12:return self.initial.knots[round(t)]
        idx=next((i for i,x in enumerate(TIMES) if abs(t-x)<1e-10),None)
        if self.kind=='dense':
            if idx is None:raise ValueError('Dense control only defines the prespecified 3/2-step samples')
            return {n:self.base[idx][n]+self.delta[f'{key(n)}__sample{idx}'] for n in self.names}
        base=self.base[idx] if idx is not None else self.initial.sample(t)
        coeff=completed_basis(t) if self.kind=='complete_curve' else derivative_basis(t)@self.transform
        out={}
        for name in self.names:
            value=base[name]
            for j,c in enumerate(coeff):
                if abs(c)>1e-14:value=value+float(c)*self.delta[key(name)][j]
            out[name]=value
        return out


def make_banks(original,kind):
    return nn.ModuleDict({str(s):DeltaBank(original,s,kind) for s in WINDOWS})

def set_model(model,original,banks,n,materialized=False,only=None):
    starts=WINDOWS if only is None else only
    return install(model,original,{s:{'bank':banks[str(s)],'steps':n} for s in starts},materialized=materialized)

def save_banks(banks,path):
    save_file({f'{s}.{name}':p.detach().contiguous() for s,b in banks.items() for name,p in b.delta.items()},str(path))

def load_banks(banks,path):
    state=load_file(str(path))
    for s,b in banks.items():b.delta.load_state_dict({k[len(s)+1:]:v for k,v in state.items() if k.startswith(s+'.')})

@torch.no_grad()
def sampled_delta_rms(banks):
    ss=0.;count=0
    for b in banks.values():
        for i,t in enumerate(TIMES):
            current=b.sample(t)
            for name in b.names:
                d=current[name]-b.base[i][name];ss+=float(d.double().square().sum());count+=d.numel()
    return math.sqrt(ss/count)

@torch.no_grad()
def reset_delta(banks):
    for b in banks.values():
        for p in b.delta.values():p.zero_()

@torch.no_grad()
def project_dense(original,dense):
    projected=make_banks(original,'orthogonal_curve')
    _,_,u,_,left_null=basis_analysis();u=torch.tensor(u,dtype=torch.float64)
    result={};total_ss=0.;discarded_ss=0.;gram=torch.zeros((5,5),dtype=torch.float64)
    for s,b in dense.items():
        families={}
        for name in b.names:
            d=torch.stack([b.delta[f'{key(name)}__sample{i}'] for i in range(5)]).double()
            flat=d.flatten(1);gram+=flat@flat.T
            c=torch.tensordot(u.T,d,dims=([1],[0]));p=torch.tensordot(u,c,dims=([1],[0]))
            projected[s].delta[key(name)].copy_(c.float())
            ss=float(d.square().sum());ds=float((d-p).square().sum());total_ss+=ss;discarded_ss+=ds
            families[name]={'increment_squared_norm':ss,'discarded_squared_norm':ds,'discarded_energy_fraction':ds/ss if ss else 0.}
        result[s]=families
    eigenvalues=torch.linalg.eigvalsh(gram).flip(0)
    return projected,{'per_window_family':result,'total_discarded_energy_fraction':discarded_ss/total_ss,
        'dense_increment_depth_gram':gram.tolist(),'dense_increment_depth_eigenvalues':eigenvalues.tolist(),
        'best_unconstrained_rank4_increment_energy_residual_fraction':float(eigenvalues[-1]/eigenvalues.sum()),
        'isotropic_random_one_direction_expected_energy_fraction':.2,
        'meaning':'Fraction of learned dense increment squared norm outside the rank-4 shared sample space; original weight knots remain fixed.'}

@torch.no_grad()
def lift_dense(original,dense):
    lifted=make_banks(original,'complete_curve')
    h=torch.tensor(np.stack([completed_basis(t) for t in TIMES]),dtype=torch.float64)
    assert float((h.T@h-torch.eye(5,dtype=torch.float64)).abs().max())<1e-12
    for s,b in dense.items():
        for name in b.names:
            d=torch.stack([b.delta[f'{key(name)}__sample{i}'] for i in range(5)]).double()
            c=torch.tensordot(h.T,d,dims=([1],[0]))
            lifted[s].delta[key(name)].copy_(c.float())
    return lifted,{'sample_basis':h.tolist(),'gram_max_abs_error':float((h.T@h-torch.eye(5,dtype=torch.float64)).abs().max()),
        'bubble':'q(t) = product_{j=0..3} (t-j)^2',
        'fifth_mode':'r(t)=[q(t)-a(t) A^+ q_samples] / [left_null^T q_samples]',
        'coefficients':'H^T times the five learned dense increments, where H=[U,left_null] is orthogonal.',
        'claim_scope':'Exact interpolation algebra at the five 3/2-step samples, and unchanged original four knots. FP32 end-to-end differences are measured separately; other depths are not validated.'}

def export_native(model,original,banks,n):
    """Sample once, fold residual multipliers into o/down projections.

    The resulting ModuleList contains ordinary native LlamaDecoderLayers. Float
    rounding changes when scaling is folded; equivalence is checked numerically.
    """
    import copy
    layers=[];i=0
    while i<len(original):
        if i not in WINDOWS:
            b=original[i];b.self_attn.layer_idx=len(layers);layers.append(b);i+=1;continue
        for t in coordinates(n):
            b=copy.deepcopy(original[i]);b.self_attn.layer_idx=len(layers)
            with torch.no_grad():
                params=banks[str(i)].sample(t)
                for name,p in b.named_parameters():p.copy_(params[name])
                b.self_attn.o_proj.weight.mul_(4/n);b.mlp.down_proj.weight.mul_(4/n)
            layers.append(b)
        i+=4
    model.model.layers=nn.ModuleList(layers)
    return layers
