import time, numpy as np, torch
import torch.nn.functional as F
from common import *
from behavior import load_model

@torch.inference_mode()
def score(model, data):
    rows=[]; n=32
    for start in range(0,len(data),4):
        ids=torch.tensor(data[start:start+4]); logits=model(ids,use_cache=False).logits
        logp=F.log_softmax(logits,dim=-1)
        for b in range(len(ids)):
            p=torch.arange(1,n); q=p+n
            first=-logp[b,p,ids[b,p+1]]; second=-logp[b,q,ids[b,q+1]]
            rows.append({'first_nll':float(first.mean()),'second_nll':float(second.mean()),
              'advantage':float((first-second).mean()),
              'accuracy':float((logits[b,q].argmax(-1)==ids[b,q+1]).float().mean())})
    return {'sequences':rows,**{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}}

def main():
    begin=time.time(); rng=np.random.default_rng(SEED+1)
    seq=np.stack([rng.choice(np.arange(100,50000),32,replace=False) for _ in range(64)])
    data=np.concatenate([np.zeros((64,1),dtype=np.int64),seq,seq],axis=1)
    np.save(OUT/'transition_heldout_sequences.npy',data)
    results=[]
    for step in [1000,8000,143000]:
        model=load_model(step); reference=None
        for head in [None,6,5,7]:
            hook=None
            if head is not None:
                def zero_head(module,args,h=head):
                    x=args[0].clone(); x[...,h*64:(h+1)*64]=0
                    return (x,)+args[1:]
                hook=model.gpt_neox.layers[3].attention.dense.register_forward_pre_hook(zero_head)
            res=score(model,data)
            if hook is not None: hook.remove()
            if head is None: reference=res
            else:
                delta=np.array([a['second_nll']-b['second_nll'] for a,b in zip(res['sequences'],reference['sequences'])])
                boot=np.random.default_rng(SEED).choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
                res['paired_second_nll_increase']=float(delta.mean())
                res['paired_nll_95ci']=[float(x) for x in np.quantile(boot,[.025,.975])]
            results.append({'step':step,'ablated_layer':3 if head is not None else None,'ablated_head':head,**res})
            dump('head_ablation.json',results)
            print(step,head,res['accuracy'],res['second_nll'],flush=True)
        del model
    a=tensor_map(state(512)); b=tensor_map(state(1000)); rows=[]
    for l in range(6):
        for head in range(8):
            for fam in ['q','k','v','o']:
                d=b[(l,fam)]-a[(l,fam)]
                x=d[:,head*64:(head+1)*64] if fam=='o' else d[head*64:(head+1)*64]
                c=real_fft(x.ravel()); st=concentration(c)
                rows.append({'layer':l,'head':head,'family':fam,**st,'low10':energy(c[:int(c.size*.1)])/energy(c)})
    dump('transition_head_spectra.json',rows)
    dump('transition_environment.json',{'seconds':time.time()-begin,'new_sequences':64,'seed':SEED+1,'optimizer_steps':0})
    print('TRANSITION COMPLETE',flush=True)

if __name__=='__main__': main()
