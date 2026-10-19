"""Check exported block states and a measured held-out window after reload."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from load_candidate import load_first_block
from prepare_data import model_dir

ROOT=Path(__file__).resolve().parent


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model=AutoModelForCausalLM.from_pretrained(model_dir(),local_files_only=True,
              torch_dtype=torch.float32,attn_implementation='eager').eval()
    original=copy.deepcopy(model.model.layers[0])
    results=json.loads((ROOT/'results/heldout_test.json').read_text())
    row=torch.from_numpy(np.load(ROOT/'data/token_windows.npz')['test'][0]).long()
    ids=row[:-1][None];target=row[1:][None]
    checked=[];checkpoints=[]
    for path in sorted((ROOT/'checkpoints').glob('*.safetensors')):
        result=next(r for r in results if r['spec']['id']==path.stem)
        model.model.layers[0]=copy.deepcopy(original)
        load_first_block(model,path,result['spec'])
        state=load_file(str(path));loaded=model.model.layers[0].state_dict()
        assert state.keys()==loaded.keys()
        assert all(torch.equal(t,loaded[k]) for k,t in state.items())
        logits=model(ids,use_cache=False).logits
        nll=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1)))
        expected=result['metrics']['per_window_nll'][0]
        assert abs(nll-expected)<5e-6,(path.name,nll,expected)
        checked.append({'candidate':path.stem,'state_tensors_exact':len(state),
                        'first_test_window_nll':nll,'expected_nll':expected,
                        'absolute_loss_difference':abs(nll-expected)})
        layouts={name:module.layout for name,module in model.model.layers[0].named_modules()
                 if hasattr(module,'layout')}
        checkpoints.append({'file':path.name,'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'spec':result['spec'],'accounting':result['accounting'],
                'curve_layouts':layouts,'state_tensors':{k:list(t.shape) for k,t in state.items()}})
        print('Verified',path.name,'loss difference',abs(nll-expected),flush=True)
    assert len(checked)==3
    (ROOT/'results/checkpoint_verification.json').write_text(json.dumps({'status':'PASS','checks':checked},indent=2)+'\n')
    manifest={'format':'smollm2-block0-structured-curves-v1','scope':'block 0 only; CPU FP32/eager reference',
              'model':json.loads((ROOT/'model_manifest.json').read_text()),'checkpoints':checkpoints}
    (ROOT/'checkpoints/checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
