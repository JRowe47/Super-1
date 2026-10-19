"""One-time provenance recipe; the resulting reference weights are bundled."""
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file

phase5=Path(sys.argv[1]).resolve()
destination=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(phase5))
from common import setup,load_model,ROOT
from joint_model import make_banks,load_banks,export_native

setup();model=load_model();original=list(model.model.layers)
banks=make_banks(original,'dense');load_banks(banks,ROOT/'checkpoints/dense.safetensors')
export_native(model,original,banks,3)
state={}
for start,new_start in [(4,4),(12,11)]:
    for i in range(3):
        for name,p in model.model.layers[new_start+i].named_parameters():
            state[f'{start}.{i}.{name}']=p.detach().contiguous()
save_file(state,str(destination))
print({'destination':str(destination),'values':sum(p.numel() for p in state.values())},flush=True)
