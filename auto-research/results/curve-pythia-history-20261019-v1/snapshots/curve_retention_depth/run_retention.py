"""Paired native/full-Haar continuation with disjoint teacher-preservation replay."""
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
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def dump(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
@torch.inference_mode()
def lm_eval(m,ids):
    values=[]
    for i in range(0,len(ids),2):
        x=torch.tensor(ids[i:i+2]);logits=m(x,use_cache=False).logits[:,:-1]
        values.extend(F.cross_entropy(logits.transpose(1,2),x[:,1:],reduction='none').mean(1).tolist())
    return dict(nll=float(np.mean(values)),block_nll=values,tokens=len(ids)*95)
def replacements(names,shapes,master,curve,backbone=False):
    return {(n.removeprefix('gpt_neox.') if backbone else n):(decode(v) if curve else v).reshape(s).float() for n,s,v in zip(names,shapes,master)}
@torch.inference_mode()
def install(m,names,shapes,master,curve):
    ps=dict(m.named_parameters())
    for n,s,c in zip(names,shapes,master):ps[n].copy_((inverse(c) if curve else c).reshape(s).float())
def copy_batch(rows):
    n=rows[0]['n'];ids=[]
    for j,r in enumerate(rows):
        a=np.array(r['prefix']);order=np.arange(n-1) if j==0 else np.array(r['order']);ids.append(np.r_[0,a,a[order]])
    return torch.tensor(np.stack(ids)),torch.arange(n+1,2*n)
def kl(logits,teacher):
    q=F.log_softmax(logits.double(),dim=-1)
    return (teacher.exp()*(teacher-q)).sum(-1).mean()
@torch.inference_mode()
def bands(m,names,initial,arm,step):
    ps=dict(m.named_parameters());out=[]
    for name,w in zip(names,initial):
        delta=ps[name].double().flatten()-w;c=haar(delta);n=c.numel();small=n;levels=0
        while small%2==0:small//=2;levels+=1
        segments=[('approximation',levels,0,small)];off=small;length=small
        for level in range(levels,0,-1):segments.append(('detail',level,off,off+length));off+=length;length*=2
        assert off==n
        total=float(delta.square().sum());assert abs(float(c.square().sum())-total)<1e-10+1e-10*total
        for kind,lev,start,end in segments:
            a=c[start:end];out.append(dict(arm=arm,step=step,tensor=name,layer=int(name.split('.')[2]),kind=kind,level=lev,support=2**lev,count=end-start,update_energy=float(a.square().sum()),mean_squared_coefficient=float(a.square().mean()),tensor_update_energy=total))
    return out
def main(checkpoints,reuse):
    start=time.time();torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);S.CHECKPOINTS=checkpoints
    protocol_sha=sha(ROOT/'protocol.json');manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text());assert sha(checkpoints/'step143000/model.safetensors')==manifest['sha256']
    z=np.load(ROOT/'data/phase_blocks.npz');data={k:z[k].copy() for k in z.files};inputs=S.read('probe_inputs.json')
    m=S.model(143000)
    for p in m.parameters():p.requires_grad_(False)
    entries=[(n,p) for n,p in m.named_parameters() if n.startswith('gpt_neox.layers.')];names=[n for n,p in entries];shapes=[tuple(p.shape) for n,p in entries];initial=[p.detach().double().flatten().clone() for n,p in entries]
    assert len(names)==72 and sum(v.numel() for v in initial)==18914304
    dump('tensor_mapping.json',[dict(name=n,shape=s,count=v.numel()) for n,s,v in zip(names,shapes,initial)])
    # Normal no-grad tensors are needed because their indices are reused in autograd.
    cache=[];textpos=torch.linspace(0,94,8).round().long()
    with torch.no_grad():
        for i in range(12):
            ids,pos=copy_batch(inputs['sets']['calibration'][2*i:2*i+2]);h=m.gpt_neox(ids,use_cache=False).last_hidden_state[:,pos]
            copyteacher=F.log_softmax(m.embed_out(h).double(),dim=-1)
            textids=torch.tensor(data['replay'][2*i:2*i+2]);h=m.gpt_neox(textids,use_cache=False).last_hidden_state[:,textpos]
            textteacher=F.log_softmax(m.embed_out(h).double(),dim=-1)
            cache.append((ids,pos,copyteacher,textids,textteacher))
        base_ret,base_hidden,prose_teacher=S.evaluate(m,'retention')
        last_teacher={k:F.log_softmax(m.embed_out(torch.from_numpy(v)).double(),dim=-1) for k,v in base_hidden.items()}
    base=dict(validation=lm_eval(m,data['validation']),retention=base_ret)
    dump('baseline.json',base);allrows=[];bandrows=[];training=[];runtimes=[];checks=[];reuse_log=[]
    def evaluate(arm,step):
        with torch.inference_mode():
            lm=lm_eval(m,data['validation']);ret,hidden,_=S.evaluate(m,'retention',prose_teacher)
            for kind in ['repeat','scrambled']:
                lp=F.log_softmax(m.embed_out(torch.from_numpy(hidden[kind+'_hidden'])).double(),dim=-1);old=last_teacher[kind+'_hidden'];vals=(old.exp()*(old-lp)).sum(-1).tolist()
                for row,value in zip(ret[kind]['sequences'],vals):row['last_query_teacher_kl']=value
            row=dict(arm=arm,step=step,validation=lm,retention=ret);allrows.append(row);dump('evaluations.json',allrows)
            short=[r['accuracy'] for r in ret['repeat']['sequences'] if r['n']<=48]
            print('EVAL',arm,step,'NLL',round(lm['nll'],6),'copy',round(float(np.mean(short)),4),flush=True)
    evaluate('baseline',0)
    for coordinate,lam in [('native',0),('curve',0),('native',1),('curve',1),('native',4),('curve',4)]:
        arm=f'{coordinate}_lambda{lam}';curve=coordinate=='curve';t=time.time()
        if lam==0 and reuse is not None:
            name=f'{coordinate}_adamw_final_decoder.npz';path=reuse/'models'/name
            prior=json.loads((reuse/'results/pythia_environment.json').read_text());expected=next(r['sha256'] for r in prior['final_checkpoints'] if r['arm']==coordinate+'_adamw')
            assert sha(path)==expected
            assert np.array_equal(data['train'],np.load(reuse/'data/lm_token_blocks.npz')['train'])
            with np.load(path) as zz,torch.no_grad():
                ps=dict(m.named_parameters())
                for n in names:ps[n].copy_(torch.from_numpy(zz[n]))
            evaluate(arm,24);bandrows.extend(bands(m,names,initial,arm,24));reuse_log.append(dict(arm=arm,sha256=expected,origin='preceding compatibility phase; decoded final weights'))
            runtimes.append(dict(arm=arm,seconds=time.time()-t,new_training_steps=0));continue
        master=[torch.nn.Parameter(haar(v) if curve else v.clone()) for v in initial]
        opt=torch.optim.AdamW(master,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        install(m,names,shapes,master,curve)
        for step in range(1,25):
            opt.zero_grad(set_to_none=True);x=torch.tensor(data['train'][2*(step-1):2*step])
            output=functional_call(m,replacements(names,shapes,master,curve),(),dict(input_ids=x,labels=x,use_cache=False));adapt=output.loss;adapt.backward();av=float(adapt.detach());del output,adapt
            first=[v.grad.clone() for v in master] if step==1 else None
            copyvalue=textvalue=0.
            if lam:
                ids,pos,teacher,txt,txtteacher=cache[(step-1)%12]
                h=functional_call(m.gpt_neox,replacements(names,shapes,master,curve,True),(),dict(input_ids=ids,use_cache=False)).last_hidden_state[:,pos]
                loss=kl(m.embed_out(h),teacher);copyvalue=float(loss.detach());(lam*.5*loss).backward();del h,loss
                h=functional_call(m.gpt_neox,replacements(names,shapes,master,curve,True),(),dict(input_ids=txt,use_cache=False)).last_hidden_state[:,textpos]
                loss=kl(m.embed_out(h),txtteacher);textvalue=float(loss.detach());(lam*.5*loss).backward();del h,loss
            if step==1:
                checks.append(dict(arm=arm,initial_copy_kl=copyvalue,initial_text_kl=textvalue,initial_replay_gradient_max=max(float((v.grad-g).abs().max()) for v,g in zip(master,first))));del first
            opt.step();training.append(dict(arm=arm,step=step,adaptation_nll=av,copy_kl=copyvalue,text_kl=textvalue,total=av+lam*.5*(copyvalue+textvalue)));dump('training.json',training)
            if step in [12,24]:
                install(m,names,shapes,master,curve);evaluate(arm,step);bandrows.extend(bands(m,names,initial,arm,step));dump('haar_update_bands.json',bandrows)
        if lam==4:
            arrays={n:dict(m.named_parameters())[n].detach().numpy().copy() for n in names};path=ROOT/'models'/f'{arm}_final_decoder.npz';np.savez_compressed(path,**arrays)
        runtimes.append(dict(arm=arm,seconds=time.time()-t,new_training_steps=24));print('COMPLETE',arm,round(time.time()-t,2),flush=True)
        del opt,master;gc.collect()
    dump('haar_update_bands.json',bandrows);dump('initial_checks.json',checks);dump('reused_controls.json',reuse_log)
    assert sha(ROOT/'protocol.json')==protocol_sha
    dump('environment.json',dict(seconds=time.time()-start,runs=runtimes,new_training_steps=sum(r['new_training_steps'] for r in runtimes),threads=3,torch=torch.__version__,protocol_sha256=protocol_sha,checkpoint_sha256=manifest['sha256'],teacher_cache_bytes=sum(c[2].numel()*c[2].element_size()+c[4].numel()*c[4].element_size() for c in cache)))
    print('RETENTION COMPLETE',round(time.time()-start,2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');p.add_argument('--reuse-controls',type=Path);a=p.parse_args();main(a.checkpoints,a.reuse_controls)
