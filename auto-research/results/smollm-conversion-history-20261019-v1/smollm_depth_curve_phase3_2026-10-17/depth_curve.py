"""A learned C1 curve of SwiGLU weights and explicit Euler depth integration.

Only the first block's MLP residual transformation is replaced. Original
attention executes once. Knots index DEPTH here, not flattened weight entries.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F


def spline_basis(t,knots,dtype=torch.float32,device=None):
    """Clamped Catmull-Rom basis, including accumulated endpoint multiplicities."""
    if not 0<=float(t)<=1:raise ValueError('Depth must lie in [0,1].')
    if knots==1:return torch.ones(1,dtype=dtype,device=device)
    x=float(t)*(knots-1);j=min(int(x),knots-1);u=x-j
    u2=u*u;u3=u2*u
    weights=[-.5*u3+u2-.5*u,1.5*u3-2.5*u2+1,-1.5*u3+2*u2+.5*u,.5*u3-.5*u2]
    basis=torch.zeros(knots,dtype=dtype,device=device)
    for index,value in zip([j-1,j,j+1,j+2],weights):basis[min(knots-1,max(0,index))]+=value
    return basis


class DepthCurveMLP(nn.Module):
    def __init__(self,hidden_size,width,knots,norm_weight,epsilon,steps=1):
        super().__init__();self.hidden_size=hidden_size;self.width=width;self.knots=knots;self.steps=steps
        self.coefficients=nn.Parameter(torch.zeros(knots,3*hidden_size*width))
        self.register_buffer('norm_weight',norm_weight.detach().clone())
        self.epsilon=float(epsilon);self._sampled_weights=None

    def unpack(self,flat):
        n=self.hidden_size*self.width
        return flat[:n].view(self.width,self.hidden_size),flat[n:2*n].view(self.width,self.hidden_size),flat[2*n:].view(self.hidden_size,self.width)

    def weights_at(self,t):
        if self.knots==1:return self.unpack(self.coefficients[0])
        basis=spline_basis(t,self.knots,self.coefficients.dtype,self.coefficients.device)
        return self.unpack(basis@self.coefficients)

    def norm(self,x):
        x=x.float();return self.norm_weight*(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.epsilon))

    def field(self,x,weights):
        g,u,d=weights;x=self.norm(x)
        return F.linear(F.silu(F.linear(x,g))*F.linear(x,u),d)

    def forward(self,anchor,steps=None,return_path=False,frozen_state=False):
        n=self.steps if steps is None else steps
        if n<1:raise ValueError('Depth step count must be positive.')
        delta=torch.zeros_like(anchor);path=[anchor] if return_path else None
        cache=self._sampled_weights if steps is None else None
        for i in range(n):
            weights=cache[i] if cache is not None else self.weights_at(i/n)
            state=anchor if frozen_state else anchor+delta
            delta=delta+self.field(state,weights)/n
            if return_path:path.append(anchor+delta)
        return (delta,torch.stack(path)) if return_path else delta

    @torch.no_grad()
    def materialize(self,steps):
        """Optional inference cache; persistent coefficient storage is unchanged."""
        self.steps=steps
        self._sampled_weights=[tuple(w.detach().clone() for w in self.weights_at(i/steps)) for i in range(steps)]

    def stream(self,steps):
        self.steps=steps;self._sampled_weights=None

    def configuration(self):
        return {'hidden_size':self.hidden_size,'width':self.width,'knots':self.knots,'steps':self.steps,'epsilon':self.epsilon}


def install_student(model,source,student,steps,materialized=False):
    block=copy.deepcopy(source)
    block.post_attention_layernorm=nn.Identity()
    block.mlp=copy.deepcopy(student).eval()
    if materialized:block.mlp.materialize(steps)
    else:block.mlp.stream(steps)
    model.model.layers[0]=block
    return block


def make_from_state(config,state):
    model=DepthCurveMLP(config['hidden_size'],config['width'],config['knots'],
              state['norm_weight'],config['epsilon'],config.get('steps',1))
    model.load_state_dict(state,strict=True)
    return model


def accounting(student,steps,qk_reduced=False,materialized=False):
    per_step=3*student.hidden_size*student.width
    attention=884736-(387264 if qk_reduced else 0)
    values=attention+student.coefficients.numel()+1152
    return {'student_trainable_values':student.coefficients.numel(),'spline_knots':student.knots,
       'mlp_width':student.width,'field_evaluations':steps,'block_coefficient_values':values,
       'block_projection_macs_per_token':attention+steps*per_step,
       'block_projection_mac_fraction_saved':1-(attention+steps*per_step)/3538944,
       'sampled_matrix_cache_values':steps*per_step if materialized else 0,
       'streamed_weight_assembly_macs_per_forward':steps*student.knots*per_step if student.knots>1 and not materialized else 0,
       'norms_per_mlp':steps,'attention_evaluations':1,
       'scope':'Only block 0; learned linear MACs exclude attention scores, normalizations, activation functions, residual adds and spline assembly.'}
