"""Ordinary full-model forward + direct target indexing, independent of evaluator."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import scoring as S
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
@torch.inference_mode()
def main(checkpoints):
    S.CHECKPOINTS=checkpoints;torch.set_num_threads(3);torch.set_num_interop_threads(1)
    data=S.read('probe_inputs.json');results=S.read('confirmation.json');meta=S.read('curve_metadata.json');z=np.load(ROOT/'native_local_curves.npz');checks=[]
    for step,direction,key,lam in [(1000,'early','selected_early',1),(8000,'late','selected_late',meta['late_to_early_norm_ratio'])]:
        m=S.model(step);p=S.param(m,3);base=p.detach().clone()
        for variant in [False,True]:
            if variant:
                p.copy_(base);flat=p.view(-1)
                flat[589824:655360]=torch.from_numpy((base.view(-1)[589824:655360].double().numpy()-z[key]).astype(np.float32))
                r=next(r for r in results if r['id']!='baseline' and r['step']==step and r['direction']==direction and r['path']=='aaaaad' and abs(r['lam']-lam)<1e-7)
            else:r=next(r for r in results if r['id']=='baseline' and r['step']==step)
            for n in [16,32,48]:
                row=next(r for r in data['sets']['confirmation'] if r['n']==n);seq=np.array(row['prefix'])
                for kind in ['repeat','scrambled']:
                    order=np.arange(n-1) if kind=='repeat' else np.array(row['order']);ids=torch.tensor(np.r_[0,seq,seq[order]])[None,:]
                    logits=m(ids,use_cache=False).logits[0];q=torch.arange(n+1,2*n);targets=torch.tensor(seq[order+1])
                    lp=F.log_softmax(logits[q],dim=-1);loss=-lp[torch.arange(n-1),targets]
                    stored=next(r for r in r[kind]['sequences'] if r['id']==row['id']);err=float(abs(loss.mean().item()-stored['target_nll']))
                    # Single-item and batched GEMMs can differ in their final float32 bits.
                    assert err<2e-5,(step,direction,variant,n,kind,err)
                    assert (logits[q].argmax(-1)==targets).tolist()==stored['position_correct']
                    checks.append(dict(step=step,variant=variant,kind=kind,n=n,nll_abs_error=err,top1_positions_match=True))
            p.copy_(base)
        del m
    out=dict(method='Ordinary GPTNeoXForCausalLM.forward, direct indexed targets, one sequence at a time',checks=checks,maximum_nll_abs_error=max(r['nll_abs_error'] for r in checks))
    (OUT/'direct_scoring_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');a=p.parse_args();main(a.checkpoints)
