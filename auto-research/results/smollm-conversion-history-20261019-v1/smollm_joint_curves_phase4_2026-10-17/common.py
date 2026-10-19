import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
WINDOWS=[4,12]
SEED=50517

def dump(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)

def setup():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    for d in ['data','results','cache','checkpoints','reference']:(ROOT/d).mkdir(exist_ok=True)

def load_model():
    path=Path(os.environ.get('SMOLLM_MODEL_DIR',ROOT/'reference_model'))
    if not path.exists():path=ROOT.parent/'smollm_depth_curve_phase3/reference_model'
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager').eval()
    return model.requires_grad_(False)

def data():
    with np.load(ROOT/'data/token_windows.npz') as source:
        return {k:torch.tensor(source[k],dtype=torch.long) for k in source.files}

def hashes(model):
    return {n:hashlib.sha256(memoryview(p.detach().contiguous().numpy()).cast('B')).hexdigest()
            for n,p in model.named_parameters()}
