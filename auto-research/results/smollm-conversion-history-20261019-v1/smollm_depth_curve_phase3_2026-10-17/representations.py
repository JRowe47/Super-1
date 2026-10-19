"""Structured coefficient curves and compute-reducing first-block replacements."""
from __future__ import annotations
import copy, math
import numpy as np
import pywt
from scipy.interpolate import CubicHermiteSpline
import torch
from torch import nn
from torch.nn import functional as F
from curve_layer import CardinalCurve

def make_curve(**tensors):
    parts=[];layout={};offset=0
    for name,t in tensors.items():
        t=t.detach().float().contiguous()
        layout[name]={'offset':offset,'shape':list(t.shape),'count':t.numel()}
        offset+=t.numel();parts.append(t.flatten())
    return CardinalCurve(torch.cat(parts),layout)

class ExactCurveLinear(nn.Module):
    def __init__(self,weight):
        super().__init__();self.curve=make_curve(weight=weight)
    @property
    def weight(self):return self.curve.tensor('weight')
    def forward(self,x):return F.linear(x,self.weight)

class ExactCurveNorm(nn.Module):
    def __init__(self,source):
        super().__init__();self.curve=make_curve(weight=source.weight)
        self.variance_epsilon=source.variance_epsilon
    @property
    def weight(self):return self.curve.tensor('weight')
    def forward(self,x):
        dtype=x.dtype;x=x.float()
        x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.variance_epsilon)
        return self.weight*x.to(dtype)

def wrap_unchanged(block):
    for parent_name in ['self_attn','mlp']:
        parent=getattr(block,parent_name)
        for name,child in list(parent.named_children()):
            if isinstance(child,nn.Linear):setattr(parent,name,ExactCurveLinear(child.weight))
    for name in ['input_layernorm','post_attention_layernorm']:
        setattr(block,name,ExactCurveNorm(getattr(block,name)))

class FactorCurveLinear(nn.Module):
    """Each retained singular component has two vector curves; exact knot views."""
    def __init__(self,left,right):
        super().__init__();self.curve=make_curve(left=left,right=right)
        self.out_features,self.in_features=left.shape[0],right.shape[1]
    def forward(self,x):
        return F.linear(F.linear(x,self.curve.tensor('right')),self.curve.tensor('left'))
    @property
    def weight(self):return self.curve.tensor('left')@self.curve.tensor('right')

class JointQKCurve(nn.Module):
    """One shared input projection; both q and k remain in their original axes.

    The reference eager attention calls q then k on the SAME normalized tensor.
    The temporary factor output is cleared by k; caching is inference-only.
    No cached token-to-token attention or position-dependent values are reused.
    """
    def __init__(self,left,right,q_rows):
        super().__init__();self.curve=make_curve(left=left,right=right)
        self.q_rows=q_rows;self._x=None;self._projected=None
    def project(self,x,role):
        right=self.curve.tensor('right');left=self.curve.tensor('left')
        if role=='q':
            self._x=x;self._projected=F.linear(x,right)
            return F.linear(self._projected,left[:self.q_rows])
        z=self._projected if self._x is x and self._projected is not None else F.linear(x,right)
        self._x=None;self._projected=None
        return F.linear(z,left[self.q_rows:])

class JointQKView(nn.Module):
    def __init__(self,bank,role):
        super().__init__();object.__setattr__(self,'bank',bank);self.role=role
    def forward(self,x):return self.bank.project(x,self.role)

class ReconstructedCurveLinear(nn.Module):
    """Curve coefficients are compact, but a dense matrix is materialized once.

    This is explicitly a storage/functional control, not a matrix-FLOP saving.
    The cached dense reconstruction is counted separately as working memory.
    """
    def __init__(self,weight,coefficients,method,metadata):
        super().__init__();self.method=method;self.metadata=metadata
        self.coefficient_curve=make_curve(coefficients=coefficients.reshape(-1))
        self.register_buffer('materialized_weight',weight.float().contiguous(),persistent=False)
        self.out_features,self.in_features=weight.shape
    @property
    def weight(self):return self.materialized_weight
    def forward(self,x):return F.linear(x,self.materialized_weight)

class GroupedCurveMLP(nn.Module):
    """Evaluate selected groups of original gated neurons, keeping their coupling."""
    def __init__(self,source,indices):
        super().__init__();indices=torch.sort(indices.long()).values
        self.register_buffer('selected_neurons',indices)
        self.curve=make_curve(gate=source.gate_proj.weight[indices],
                             up=source.up_proj.weight[indices],
                             down=source.down_proj.weight[:,indices])
        self.act_fn=source.act_fn
    def forward(self,x):
        g=F.linear(x,self.curve.tensor('gate'));u=F.linear(x,self.curve.tensor('up'))
        return F.linear(self.act_fn(g)*u,self.curve.tensor('down'))

class ZeroMLP(nn.Module):
    def forward(self,x):return torch.zeros_like(x)

def svd_basis(weight,chol=None):
    w=weight.detach().double()
    transformed=w if chol is None else w@chol
    u,s,vh=torch.linalg.svd(transformed,full_matrices=False)
    right=vh if chol is None else torch.linalg.solve_triangular(chol.T,vh.T,upper=True).T
    return {'left':(u*s).float(),'right':right.float(),'singular_values':s.cpu()}

def rank_for_budget(weight,budget,joint=False):
    m,n=weight.shape
    return max(1,min(m,n,int(math.floor(budget*m*n/(m+n)))))

def reconstruct_haar(weight,budget):
    w=weight.detach().cpu().numpy().reshape(-1).astype(np.float64);n=len(w)
    power_two=(n&-n).bit_length()-1;level=min(10,power_two)
    coeffs=pywt.wavedec(w,'haar',mode='periodization',level=level)
    packed=np.concatenate(coeffs);assert len(packed)==n
    keep=max(1,int(math.floor(n*budget)))
    sparse=packed.copy();sparse[keep:]=0
    parts=[];offset=0
    for c in coeffs:parts.append(sparse[offset:offset+len(c)]);offset+=len(c)
    restored=pywt.waverec(parts,'haar',mode='periodization')[:n]
    return torch.from_numpy(restored.reshape(weight.shape)).float(),torch.from_numpy(packed[:keep]).float(),{'wavelet':'haar','level':level,'order':'coarsest-to-finest; prefix of coefficient array','full_count':n}

def reconstruct_knots(weight,budget):
    w=weight.detach().cpu().numpy().reshape(-1).astype(np.float64);n=len(w)
    keep=max(2,int(math.floor(n*budget)))
    indices=np.rint(np.linspace(0,n-1,keep)).astype(np.int64)
    values=w[indices]
    slopes=np.gradient(values,indices.astype(float),edge_order=1)
    interpolant=CubicHermiteSpline(indices,values,slopes)
    restored=interpolant(np.arange(n,dtype=float))
    return torch.from_numpy(restored.reshape(weight.shape)).float(),torch.from_numpy(values).float(),{'indices':'round(linspace(0,N-1,K))','slopes':'numpy.gradient on retained knots','full_count':n}

def install(model,original_block,spec,bases,orders):
    """Reset/replace block 0 only. Return exact accounting of representation cost."""
    block=copy.deepcopy(original_block);model.model.layers[0]=block
    total=sum(p.numel() for p in original_block.parameters())
    meta={'changed_block_indices':[0],'original_block_values':total,'changed_dense_values':0,
          'retained_changed_values':0,'dense_working_cache_values':0,'projection_mac_reduction_per_token':0,
          'ranks':{},'spec':spec}
    method=spec['method'];budget=spec.get('budget',1.0)
    if method=='original':
        meta.update({'block_values':total,'block_projection_macs_per_token':3538944});return meta
    if method=='exact_curves':
        pass
    elif method=='mlp_skip':
        removed=sum(p.numel() for p in block.mlp.parameters());block.mlp=ZeroMLP()
        meta['changed_dense_values']=removed;meta['projection_mac_reduction_per_token']=removed
    elif method=='mlp_groups':
        groups=spec['groups'];keep=groups*96
        indices=orders[spec['ordering']][:keep]
        source=block.mlp;removed=sum(p.numel() for p in source.parameters())
        block.mlp=GroupedCurveMLP(source,indices)
        retained=3*576*keep
        meta.update({'groups':groups,'retained_neurons':keep,'changed_dense_values':removed,
                     'retained_changed_values':retained,'projection_mac_reduction_per_token':removed-retained})
    elif method.startswith('joint_'):
        basis=bases[method]['qk'];w=torch.cat([block.self_attn.q_proj.weight,block.self_attn.k_proj.weight])
        rank=min(w.shape) if spec.get('full_rank') else rank_for_budget(w,budget)
        bank=JointQKCurve(basis['left'][:,:rank],basis['right'][:rank],576)
        block.add_module('joint_qk_curve',bank)
        block.self_attn.q_proj=JointQKView(bank,'q');block.self_attn.k_proj=JointQKView(bank,'k')
        retained=rank*sum(w.shape)
        meta.update({'changed_dense_values':w.numel(),'retained_changed_values':retained,
                     'projection_mac_reduction_per_token':w.numel()-retained,'ranks':{'qk':rank}})
    else:
        names={'q':['q_proj'],'k':['k_proj'],'qk':['q_proj','k_proj'],
               'all':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}[spec['target']]
        for name in names:
            parent=block.self_attn if name in ['q_proj','k_proj','v_proj','o_proj'] else block.mlp
            w=getattr(parent,name).weight
            meta['changed_dense_values']+=w.numel()
            if method in ['svd','activation_svd']:
                basis=bases[method][name]
                rank=min(w.shape) if spec.get('full_rank') else rank_for_budget(w,budget)
                module=FactorCurveLinear(basis['left'][:,:rank],basis['right'][:rank])
                retained=rank*sum(w.shape);meta['ranks'][name]=rank
                meta['projection_mac_reduction_per_token']+=w.numel()-retained
            else:
                fn=reconstruct_haar if method=='haar' else reconstruct_knots
                restored,coeff,info=fn(w,budget)
                module=ReconstructedCurveLinear(restored,coeff,method,info)
                retained=coeff.numel();meta['dense_working_cache_values']+=w.numel()
            meta['retained_changed_values']+=retained
            setattr(parent,name,module)
    wrap_unchanged(block)
    meta['block_values']=total-meta['changed_dense_values']+meta['retained_changed_values']
    meta['block_projection_macs_per_token']=3538944-meta['projection_mac_reduction_per_token']
    meta['retained_changed_fraction']=meta['retained_changed_values']/max(1,meta['changed_dense_values'])
    return meta
