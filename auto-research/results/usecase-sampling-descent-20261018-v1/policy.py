"""Reuse the verified four-step controller; optimize its parameters, not the backbone."""
import hashlib,importlib.util
from pathlib import Path
import torch
H=Path(__file__).resolve().parent; PARENT=H.parent/'adaptive-depth-shakespeare-20261018-v1'
spec=importlib.util.spec_from_file_location('prior_sampling',PARENT/'model.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

def build(c,seed,mode='fixed',base_state=None):
    torch.manual_seed(seed); m=prior.SampledCurve(c,mode)
    if base_state is not None:m.base.load_state_dict(base_state)
    return m

def freeze_base(model):
    for p in model.base.parameters():p.requires_grad_(False)
    return [p for p in model.router.parameters() if p.requires_grad]

def fingerprint(base):
    h=hashlib.sha256()
    for k,v in sorted(base.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
