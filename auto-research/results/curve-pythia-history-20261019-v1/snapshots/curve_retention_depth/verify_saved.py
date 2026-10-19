"""Re-load the prespecified lambda4 checkpoints and verify all evaluated scores."""
import argparse,importlib.metadata,json,platform
from pathlib import Path
import numpy as np
import torch
import scoring as S
from run_retention import lm_eval
ROOT=Path(__file__).resolve().parent
def main(checkpoints):
    torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);S.CHECKPOINTS=checkpoints
    m=S.model(143000);_,_,teacher=S.evaluate(m,'retention');data=np.load(ROOT/'data/phase_blocks.npz');rows=json.loads((ROOT/'results/evaluations.json').read_text());checks=[]
    for arm in ['native_lambda4','curve_lambda4']:
        with np.load(ROOT/'models'/f'{arm}_final_decoder.npz') as z,torch.no_grad():
            ps=dict(m.named_parameters());assert len(z.files)==72
            for n in z.files:ps[n].copy_(torch.from_numpy(z[n]))
        expected=next(r for r in rows if r['arm']==arm and r['step']==24);lm=lm_eval(m,data['validation']);ret,_,_=S.evaluate(m,'retention',teacher)
        le=float(np.max(np.abs(np.array(lm['block_nll'])-expected['validation']['block_nll'])))
        ce=max(abs(a[k]-b[k]) for kind in ['repeat','scrambled'] for a,b in zip(ret[kind]['sequences'],expected['retention'][kind]['sequences']) for k in ['accuracy','target_nll'])
        pe=max(abs(ret['prose'][k]-expected['retention']['prose'][k]) for k in ['nll','kl_from_baseline','argmax_changed'])
        assert max(le,ce,pe)<1e-6;checks.append(dict(arm=arm,validation_max_error=le,copy_max_error=ce,prose_max_error=pe))
    result=dict(checks=checks,python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['torch','numpy','transformers','safetensors','pyarrow','matplotlib']})
    (ROOT/'results/saved_model_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');main(p.parse_args().checkpoints)
