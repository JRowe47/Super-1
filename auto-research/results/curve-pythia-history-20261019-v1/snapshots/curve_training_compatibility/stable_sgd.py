"""Disclosed additional stability control; retains the original oversized-step run."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
import pythia_train as T
from transforms import haar,inverse
ROOT=T.ROOT;OUT=T.OUT
def main(checkpoints):
    t=time.time();torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    T.S.CHECKPOINTS=checkpoints;T.REF=ROOT/'stable_references';T.REF.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'protocol_addendum.json').read_text());lr=cfg['value']
    data=np.load(ROOT/'data/lm_token_blocks.npz');m=T.S.model(143000)
    for p in m.parameters():p.requires_grad_(False)
    entries=[(n,p) for n,p in m.named_parameters() if n.startswith('gpt_neox.layers.')]
    names=[n for n,p in entries];shapes=[tuple(p.shape) for n,p in entries];initial=[p.detach().double().flatten().clone() for n,p in entries]
    _,_,teachers=T.S.evaluate(m,'retention');rows=[];equiv=[];losses=[]
    for arm,curve in [('native_sgd',False),('curve_sgd',True)]:
        masters=[torch.nn.Parameter(haar(v) if curve else v.clone()) for v in initial]
        opt=torch.optim.SGD(masters,lr=lr,foreach=False)
        def evaluate(step):
            T.install(m,names,shapes,masters,curve)
            with torch.inference_mode():
                lm=T.lm_eval(m,data['validation']);ret,hidden,_=T.S.evaluate(m,'retention',teachers)
                h=m.gpt_neox(torch.tensor(data['validation'][:1]),use_cache=False).last_hidden_state
                ck=T.compare(m,names,arm,step,h,None)
                if ck:equiv.append(ck)
            rows.append(dict(arm=arm,step=step,validation=lm,retention=ret));T.dump('stable_sgd_evaluations.json',rows);T.dump('stable_sgd_equivalence.json',equiv)
            np.savez_compressed(ROOT/'responses'/f'stable_{arm}_{step}.npz',**hidden)
            print('STABLE',arm,step,'validation',lm['nll'],'repeat',ret['repeat']['accuracy'],flush=True)
        evaluate(0)
        for step in range(1,25):
            opt.zero_grad(set_to_none=True);x=torch.tensor(data['train'][2*(step-1):2*step]);over=T.replacements(names,shapes,masters,curve)
            out=functional_call(m,over,(),dict(input_ids=x,labels=x,use_cache=False));out.loss.backward();opt.step()
            losses.append(dict(arm=arm,step=step,loss=float(out.loss.detach())))
            if step in [1,4,12,24]:evaluate(step)
            del out,over
        del opt,masters;gc.collect()
    T.dump('stable_sgd_training.json',losses);T.dump('stable_sgd_environment.json',dict(seconds=time.time()-t,lr=lr,training_steps=48,threads=3))
    print('STABLE COMPLETE',time.time()-t,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');main(p.parse_args().checkpoints)
