"""Validate each exported student and reproduce its measured held-out loss."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, DynamicCache
from safetensors.torch import load_file
from load_candidate import restore
from prepare_data import model_dir

ROOT=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    model=AutoModelForCausalLM.from_pretrained(model_dir(),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    original=copy.deepcopy(model.model.layers[0]);results=json.loads((ROOT/'results/test.json').read_text())
    windows=torch.from_numpy(np.load(ROOT/'data/token_windows.npz')['test'][:2]).long()
    checks=[];manifest=[]
    for name in ['direct_192','direct_768','curve_fixed8','curve_mixed']:
        steps=1 if name.startswith('direct') else 8
        model.model.layers[0]=copy.deepcopy(original);restore(model,name,steps)
        path=ROOT/'checkpoints'/f'{name}.safetensors';state=load_file(str(path));actual=model.model.layers[0].mlp.state_dict()
        assert state.keys()==actual.keys() and all(torch.equal(v,actual[k]) for k,v in state.items())
        logits=model(windows[:,:-1],use_cache=False).logits
        nll=F.cross_entropy(logits.flatten(0,1),windows[:,1:].flatten(),reduction='none').view(2,-1).mean(1)
        expected=next(r['metrics']['per_window_nll'][:2] for r in results if r['configuration']['id']==f'{name}_n{steps}')
        difference=float((nll-torch.tensor(expected)).abs().max());assert difference<5e-6,difference
        info=json.loads((ROOT/'results'/f'training_{name}.json').read_text())
        checked={'student':name,'tested_steps':steps,'state_tensors_exact':len(state),'max_loss_difference':difference}
        if name.startswith('curve'):
            ids=windows[0:1,:32];full=model(ids,use_cache=False).logits;cache=DynamicCache();parts=[]
            for i in range(32):parts.append(model(ids[:,i:i+1],past_key_values=cache,use_cache=True).logits)
            error=float((full-torch.cat(parts,dim=1)).abs().max());assert error<1e-3,error
            checked['eight_step_cached_vs_prefill_max_logit']=error
        checks.append(checked)
        manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
                         'configuration':info['configuration'],'best_update':info['best_update']})
        print('Verified',name,'loss difference',difference,flush=True)
    (ROOT/'results/export_verification.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n')
    (ROOT/'checkpoints/checkpoint_manifest.json').write_text(json.dumps({'format':'depth-curve-mlp-v1','students':manifest},indent=2)+'\n')


if __name__=='__main__':main()
