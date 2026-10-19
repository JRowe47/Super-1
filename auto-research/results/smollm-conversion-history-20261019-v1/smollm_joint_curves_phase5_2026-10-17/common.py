import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
WINDOWS=[4,12]
SEED=60517
ARMS=['raw_curve','orthogonal_curve','dense']

def dump(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def setup():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    for name in ['data','results','cache','checkpoints','reference']:(ROOT/name).mkdir(parents=True,exist_ok=True)

def model_dir():
    p=Path(os.environ.get('SMOLLM_MODEL_DIR',ROOT/'reference_model'))
    return p if p.exists() else ROOT.parent/'smollm_depth_curve_phase3/reference_model'

def load_model():
    return AutoModelForCausalLM.from_pretrained(model_dir(),local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)

def data():
    with np.load(ROOT/'data/token_windows.npz') as a:
        return {k:torch.tensor(a[k],dtype=torch.long) for k in a.files}

def hashes(model):
    return {n:hashlib.sha256(memoryview(p.detach().contiguous().numpy()).cast('B')).hexdigest()
            for n,p in model.named_parameters()}
