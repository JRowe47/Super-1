
"""Identity-safe experimental wrapper around the recovered historical fast GRU.

This is an explicitly new candidate, NOT a replacement for historical source.
The baseline's transition and segment/document update logic are not copied.
No training or file access occurs on import.
"""
import torch
from torch import nn
class ConditionedFastRotation(nn.Module):
    def __init__(self,native_cell,query_kind="oriented_cross",epsilon=1e-6):
        super().__init__()
        if native_cell.hidden_size%2:raise ValueError("Even recurrent width required")
        if query_kind not in ("oriented_cross","alignment_dot"):raise ValueError("Unknown query control")
        if epsilon<=0:raise ValueError("Positive numerical epsilon required")
        self.native=native_cell;self.query_kind=query_kind;self.epsilon=float(epsilon)
        self.hidden_size=native_cell.hidden_size;self.input_size=native_cell.input_size
        if self.input_size!=3*self.hidden_size:raise ValueError("Expected historical token/segment/document input")
        self.phase=nn.Parameter(torch.zeros(self.hidden_size//2))
        self.gain=nn.Parameter(torch.zeros(self.hidden_size//2))
    def forward(self,inputs,previous):
        proposal=self.native(inputs,previous)
        width=self.hidden_size
        context=inputs[...,width:2*width]+inputs[...,2*width:3*width]
        p=proposal.reshape(*proposal.shape[:-1],-1,2);q=context.reshape(*context.shape[:-1],-1,2)
        numerator=p[...,0]*q[...,1]-p[...,1]*q[...,0] if self.query_kind=="oriented_cross" else (p*q).sum(-1)
        denominator=torch.sqrt(p.square().sum(-1)*q.square().sum(-1)+self.epsilon)
        angle=self.phase+torch.tanh(self.gain)*torch.tanh(numerator/denominator)
        c,s=angle.cos(),angle.sin()
        return torch.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),-1).flatten(-2)
def install_fast_rotation(historical_model,query_kind="oriented_cross",epsilon=1e-6):
    """Mutate only an explicitly cloned candidate; callers keep baseline separate."""
    if isinstance(historical_model.fast_cell,ConditionedFastRotation):raise ValueError("Already wrapped")
    historical_model.fast_cell=ConditionedFastRotation(historical_model.fast_cell,query_kind,epsilon)
    return historical_model
