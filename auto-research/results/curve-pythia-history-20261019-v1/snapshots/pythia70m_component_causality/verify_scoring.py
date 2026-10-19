"""Cross-check stored scores against ordinary full-model forward and direct indexing."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import causal_probe as cp

@torch.inference_mode()
def main(checkpoints):
    cp.CHECKPOINTS=checkpoints;torch.set_num_threads(3)
    m=cp.model(1000);probe=cp.read('probe_inputs.json')['sets']['confirmation'][0]
    n=probe['n'];seq=np.array(probe['prefix']);q=np.arange(n+1,2*n)
    cats={c['id']:c for c in cp.read('candidate_catalog.json')};rows={r['id']:r for r in cp.read('confirmation.json') if r['phase']=='rollback'}
    z=np.load(cp.ROOT/'native_qkv_curves.npz');delta=[z[f'step1000_layer{l}'].astype(np.float64)-z[f'step512_layer{l}'] for l in range(6)]
    orig=z['step1000_layer3'];checks=[];direct={};leaf=cp.read('selection_locked.json')['strongest_local_leaf']
    for cid in ['baseline','H6_root',leaf]:
        if cid!='baseline':cp.param(m,3).copy_(torch.from_numpy((orig.astype(np.float64)-cp.project(cats[cid],delta)).astype(np.float32)))
        for kind in ['repeat','scrambled']:
            order=np.arange(n-1) if kind=='repeat' else np.array(probe['order']);target=seq[order+1]
            ids=torch.tensor(np.r_[0,seq,seq[order]][None]);out=m(ids,use_cache=False,output_attentions=True)
            lp=F.log_softmax(out.logits[0],dim=-1);nll=float(-lp[q,torch.tensor(target)].mean())
            acc=float((out.logits[0,q].argmax(-1)==torch.tensor(target)).float().mean())
            recorded=rows[cid][kind]['sequences'][0]
            err=abs(nll-recorded['target_nll']);assert err<2e-5,(cid,kind,err)
            assert acc==recorded['accuracy']
            direct[cid,kind]=out.logits[0,q[-1]].double().numpy()
            checks.append(dict(id=cid,probe=kind,full_forward_nll=nll,recorded_nll=recorded['target_nll'],nll_difference=err,accuracy=acc))
        cp.param(m,3).copy_(torch.from_numpy(orig));assert torch.equal(cp.param(m,3),torch.from_numpy(orig))
    g=np.load(cp.OUT/'logit_gram_step1000.npy');grams=[]
    base=np.load(cp.RAW/'confirmation_rollback_baseline.npz')
    for cid in ['H6_root',leaf]:
        h=np.load(cp.RAW/f'confirmation_rollback_{cid}.npz')
        for kind in ['repeat','scrambled']:
            d=h[kind+'_hidden'][0].astype(np.float64)-base[kind+'_hidden'][0]
            estimated=float(np.sqrt(d@g@d));dl=direct[cid,kind]-direct['baseline',kind];dl-=dl.mean();actual=float(np.sqrt(np.mean(dl**2)))
            error=abs(actual-estimated);assert error<2e-5,(cid,kind,error)
            grams.append(dict(id=cid,probe=kind,direct_centered_logit_rms=actual,gram_rms=estimated,absolute_difference=error))
    cp.dump('direct_scoring_verification.json',dict(checks=checks,gram_checks=grams,
        explanation='Full GPTNeoXForCausalLM forward, direct target indexing, one confirmation sequence; small floating differences reflect batch/matmul rounding.'))
    print('Direct forward scores, parameter restores and Gram reconstruction verified.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=Path,default=cp.DEFAULT);main(ap.parse_args().checkpoints)
