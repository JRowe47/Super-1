"""Actual coefficient autograd and paired decoder continuation, with native controls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','3')
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
import scoring as S
from transforms import haar,inverse,decode
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';MODELS=ROOT/'models';REF=ROOT/'temporary_references'
P=json.loads((ROOT/'protocol.json').read_text())['pythia']
def dump(name,obj):
    path=OUT/name;t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(path)
def hashfile(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
@torch.inference_mode()
def lm_eval(m,ids):
    out=[]
    for i in range(0,len(ids),2):
        x=torch.tensor(ids[i:i+2]);logits=m(x,use_cache=False).logits[:,:-1]
        losses=F.cross_entropy(logits.transpose(1,2),x[:,1:],reduction='none').mean(1)
        out.extend(losses.tolist())
    return dict(nll=float(np.mean(out)),block_nll=out,tokens=len(ids)*95)
def native_values(master,curve):return [(decode(c) if curve else c) for c in master]
def replacements(names,shapes,master,curve):return {n:w.reshape(shape).float() for n,shape,w in zip(names,shapes,native_values(master,curve))}
@torch.inference_mode()
def install(m,names,shapes,master,curve):
    params=dict(m.named_parameters())
    for name,shape,c in zip(names,shapes,master):params[name].copy_((inverse(c) if curve else c).reshape(shape).float())
@torch.inference_mode()
def compare(m,names,arm,step,h,values):
    refarm={'curve_sgd':'native_sgd','curve_transport_adamw':'native_adamw'}.get(arm)
    if arm in ['native_sgd','native_adamw']:
        torch.save(dict(parameters=[dict(m.named_parameters())[n].clone() for n in names],hidden=h),REF/f'{arm}_{step}.pt')
        return None
    if not refarm:return None
    ref=torch.load(REF/f'{refarm}_{step}.pt',weights_only=True);diffs=[dict(m.named_parameters())[n].double()-v.double() for n,v in zip(names,ref['parameters'])]
    mx=max(float(d.abs().max()) for d in diffs);sq=sum(float((d*d).sum()) for d in diffs);count=sum(d.numel() for d in diffs)
    logits=m.embed_out(h);old=m.embed_out(ref['hidden']);err=logits-old
    return dict(reference=refarm,step=step,max_abs_parameter_difference=mx,parameter_rms_difference=float(np.sqrt(sq/count)),
        max_abs_hidden_difference=float((h-ref['hidden']).abs().max()),max_abs_logit_difference=float(err.abs().max()),logit_rms_difference=float(err.double().square().mean().sqrt()))
def main(checkpoints):
    begin=time.time();torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    REF.mkdir(exist_ok=True);S.CHECKPOINTS=checkpoints;protocol_hash=hashfile(ROOT/'protocol.json')
    manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text());actual=hashfile(checkpoints/'step143000/model.safetensors');assert actual==manifest['sha256']
    z=np.load(ROOT/'data/lm_token_blocks.npz');data={k:z[k].copy() for k in z.files};m=S.model(143000)
    for p in m.parameters():p.requires_grad_(False)
    entries=[(n,p) for n,p in m.named_parameters() if n.startswith('gpt_neox.layers.')]
    names=[n for n,p in entries];shapes=[tuple(p.shape) for n,p in entries];initial=[p.detach().double().flatten().clone() for n,p in entries]
    assert sum(v.numel() for v in initial)==18914304 and len(names)==72
    dump('pythia_tensor_mapping.json',[dict(name=n,shape=s,count=v.numel()) for n,s,v in zip(names,shapes,initial)])
    initial_checks=[]
    for n,v in zip(names,initial):
        rec=inverse(haar(v));err=float((rec-v).abs().max());assert err<1e-12
        initial_checks.append(dict(name=n,max_abs_roundtrip=err,float32_exact=torch.equal(rec.float(),v.float()),float32_differing_entries=int((rec.float()!=v.float()).sum())))
    dump('pythia_roundtrip_checks.json',initial_checks)
    baseline_ret,_,teachers=S.evaluate(m,'retention');baseline_lm=lm_eval(m,data['validation']);dump('pythia_baseline.json',dict(validation=baseline_lm,retention=baseline_ret))
    allrows=[];equiv=[];first_checks=[];times=[];finals=[]
    for arm in P['arms']:
        t=time.time();curve=arm.startswith('curve');transport='transport' in arm;adam='adamw' in arm
        master=[torch.nn.Parameter(haar(v) if curve else v.clone()) for v in initial]
        optparams=[torch.nn.Parameter(v.clone()) for v in initial] if transport else master
        optimizer=torch.optim.AdamW(optparams,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False) if adam else torch.optim.SGD(optparams,lr=.005,weight_decay=0.,foreach=False)
        def evaluate(step):
            install(m,names,shapes,master,curve)
            with torch.inference_mode():
                lm=lm_eval(m,data['validation']);ret,hidden,_=S.evaluate(m,'retention',teachers)
                h=m.gpt_neox(torch.tensor(data['validation'][:1]),use_cache=False).last_hidden_state
                check=compare(m,names,arm,step,h,None)
                if check:equiv.append(dict(arm=arm,**check));dump('pythia_equivalence.json',equiv)
                np.savez_compressed(ROOT/'responses'/f'{arm}_{step}.npz',**hidden)
            row=dict(arm=arm,step=step,validation=lm,retention=ret)
            allrows.append(row);dump('pythia_evaluations.json',allrows)
            print('EVAL',arm,step,'validation',round(lm['nll'],6),'repeat',round(ret['repeat']['accuracy'],4),flush=True)
        (ROOT/'responses').mkdir(exist_ok=True);evaluate(0);losses=[]
        for step in range(1,25):
            for c in master:c.grad=None
            optimizer.zero_grad(set_to_none=True);x=torch.tensor(data['train'][2*(step-1):2*step])
            overrides=replacements(names,shapes,master,curve)
            output=functional_call(m,overrides,(),dict(input_ids=x,labels=x,use_cache=False));loss=output.loss;loss.backward()
            if step==1:
                # Independent native-coordinate gradients at the same decoded weights.
                probe=[v.clone().requires_grad_() for v in initial]
                native_out=functional_call(m,{n:v.reshape(s).float() for n,s,v in zip(names,shapes,probe)},(),dict(input_ids=x,labels=x,use_cache=False))
                ng=torch.autograd.grad(native_out.loss,probe)
                errors=[float(((inverse(c.grad) if curve else c.grad)-g).abs().max()) for c,g in zip(master,ng)]
                assert max(errors)<1e-10
                first_checks.append(dict(arm=arm,gradient_max_abs_error=max(errors),loss_difference=float(loss.detach())-float(native_out.loss.detach())))
                del probe,native_out,ng
            if transport:
                for p,c in zip(optparams,master):p.grad=inverse(c.grad)
            # Save the first update for equal-function optimizer direction comparison.
            before=[c.detach().clone() for c in master] if step==1 else None
            optimizer.step()
            if transport:
                with torch.no_grad():
                    for c,p in zip(master,optparams):c.copy_(haar(p))
            if step==1:
                delta=[(inverse(c.detach()-b) if curve else c.detach()-b) for c,b in zip(master,before)]
                torch.save(delta,REF/f'{arm}_first_update.pt');del before,delta
            losses.append(dict(step=step,loss=float(loss.detach())));dump(f'pythia_training_{arm}.json',losses)
            if step in P['evaluations']:evaluate(step)
            del overrides,output,loss
        with torch.no_grad():
            # Save exact final float32 decoded weights for the two genuinely different Adam runs.
            if arm in ['native_adamw','curve_adamw']:
                arrays={n:dict(m.named_parameters())[n].numpy().copy() for n in names}
                np.savez_compressed(MODELS/f'{arm}_final_decoder.npz',**arrays)
                finals.append(dict(arm=arm,file=f'models/{arm}_final_decoder.npz',sha256=hashfile(MODELS/f'{arm}_final_decoder.npz')))
        times.append(dict(arm=arm,seconds=time.time()-t));print('ARM COMPLETE',arm,times[-1]['seconds'],flush=True)
        del optimizer,optparams,master;gc.collect()
    updates={a:torch.load(REF/f'{a}_first_update.pt',weights_only=True) for a in P['arms']}
    compare_updates=[]
    for a,b in [('native_sgd','curve_sgd'),('native_adamw','curve_transport_adamw'),('native_adamw','curve_adamw')]:
        aa,bb=updates[a],updates[b];n1=np.sqrt(sum(float((v*v).sum()) for v in aa));n2=np.sqrt(sum(float((v*v).sum()) for v in bb));dot=sum(float((v*w).sum()) for v,w in zip(aa,bb))
        compare_updates.append(dict(a=a,b=b,norm_a=n1,norm_b=n2,cosine=dot/(n1*n2),max_abs_difference=max(float((v-w).abs().max()) for v,w in zip(aa,bb))))
    dump('pythia_first_step_checks.json',dict(gradient_checks=first_checks,update_comparisons=compare_updates))
    assert hashfile(ROOT/'protocol.json')==protocol_hash
    dump('pythia_environment.json',dict(seconds=time.time()-begin,arms=times,training_steps=120,checkpoint_sha256=actual,protocol_sha256=protocol_hash,threads=3,torch=torch.__version__,final_checkpoints=finals))
    print('PYTHIA COMPLETE',time.time()-begin,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');main(ap.parse_args().checkpoints)
