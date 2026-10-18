import argparse, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import ARMS, CurveCell
from tasks import episode, check_oracle

ROOT = Path(__file__).resolve().parent

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_hash(h, *tensors):
    for t in tensors: h.update(t.numpy().tobytes())

@torch.no_grad()
def evaluate(model, seeds, delays=(32, 256, 1024), batch=64, intervention='none'):
    model.eval(); results=[]
    for delay in delays:
        for mi, mode in enumerate(('retention', 'conflict', 'reopen')):
            correct=[]; ce=[]; locks=[]; reopened=[]; ph=[]
            digest=hashlib.sha256()
            for seed in seeds:
                x,y,meta=episode(seed+10000*mi+delay, batch, delay, mode)
                tensor_hash(digest,x,y)
                logits, state=model(x, intervention=intervention)
                pred=logits[:,-4:].argmax(-1)
                good=pred.eq(y[:,-4:])
                correct.append(good.numpy()); locks.append(meta['locked'].numpy())
                reopened.append(good.gather(1,meta['reopen_key'][:,None]).numpy()[:,0])
                ce.append(F.cross_entropy(logits[:,-4:].reshape(-1,8), y[:,-4:].reshape(-1), reduction='none').numpy().reshape(-1,4))
                ph.append(float(state[1].square().mean().sqrt()) if state[1].numel() else 0.)
            c=np.concatenate(correct); l=np.concatenate(locks); loss=np.concatenate(ce)
            results.append(dict(mode=mode, delay=delay, accuracy=float(c.mean()),
                locked_accuracy=float(c[l].mean()), unlocked_accuracy=float(c[~l].mean()),
                reopened_accuracy=float(np.concatenate(reopened).mean()) if mode=='reopen' else None,
                exact_episode_accuracy=float(c.all(1).mean()), ce=float(loss.mean()),
                episodes=len(c), targets=int(c.size), p_rms=float(np.mean(ph)),
                input_target_sha256=digest.hexdigest(), per_episode_correct=c.sum(1).tolist(),
                per_episode_ce=loss.mean(1).tolist()))
    return results

def verify():
    torch.set_num_threads(1)
    out={}
    for mode in ('train','retention','conflict','reopen'):
        x,y,_=episode(17,8,32,mode); check_oracle(x,y)
    out['independent_oracle']=True
    x,y,_=episode(92,4,17)
    fixed=CurveCell('fixed48',77).double()
    for arm in ('context64','adaptive64'):
        m=CurveCell(arm,77).double()
        a,_=fixed(x); b,_=m(x)
        err=float((a-b).abs().max()); assert err==0
        out[arm+'_zero_extension_max_error']=err
    m=CurveCell('adaptive64',77).double()
    with torch.no_grad():
        m.updater.weight.normal_(std=.03)
        m.updater.bias[:16].normal_(std=.04)
    full,state=m(x)
    first,s=m(x[:,:11]); second,s=m(x[:,11:],s)
    err=float((full-torch.cat((first,second),1)).abs().max())
    assert err==0
    out['nonzero_chunk_carry_max_error']=err
    altered=x.clone(); altered[:,16:]=torch.randint(0,52,altered[:,16:].shape)
    other,_=m(altered); err=float((full[:,:16]-other[:,:16]).abs().max())
    assert err==0
    out['causal_prefix_max_error']=err
    loss=F.cross_entropy(full.flatten(0,1),y.flatten(),ignore_index=-100)
    loss.backward()
    ix=(3,18); analytic=float(m.updater.weight.grad[ix]); original=float(m.updater.weight[ix])
    def value(delta):
        with torch.no_grad(): m.updater.weight[ix]=original+delta
        z,_=m(x)
        return float(F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100))
    finite=(value(1e-5)-value(-1e-5))/2e-5
    with torch.no_grad():m.updater.weight[ix]=original
    error=abs(analytic-finite)
    assert error < 1e-7
    out['updater_gradient']={'analytic':analytic,'finite_difference':finite,'absolute_error':error}
    out['counts']={arm:CurveCell(arm,0).counts() for arm in ARMS}
    assert out['counts']['adaptive64']==out['counts']['context64']
    assert all(v['state_scalars']==64 for k,v in out['counts'].items() if k!='fixed48')
    out['curve_orthogonality_max_error']=float((m.basis.T@m.basis/(m.width//2)*8-torch.eye(8)).abs().max())
    out['status']='PASS'
    save(ROOT/'VERIFICATION.json',out)
    print(json.dumps(out),flush=True)

def train(arm, seed, steps, split, limit):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder=ROOT/split/f'{arm}_s{seed}'
    assert not (folder/'TERMINAL.json').exists(), 'Refuse overwrite completed run'
    folder.mkdir(parents=True,exist_ok=True)
    model=CurveCell(arm,seed)
    optim=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
    start=time.perf_counter(); cpu0=time.process_time(); digest=hashlib.sha256()
    trace=[]; targets=0; input_tokens=0
    settings=dict(arm=arm,seed=seed,steps=steps,split=split,counts=model.counts(),
                  torch=torch.__version__,platform=platform.platform(),threads=1,
                  source={f:sha(ROOT/f) for f in ('model.py','tasks.py','experiment.py')})
    save(folder/'START.json',settings)
    for i in range(steps):
        # Paired streams across arms; independent streams across training seeds.
        data_seed=7000000+seed*10000+i
        delay=(8,16,32)[np.random.default_rng(data_seed+1).integers(3)]
        mode='retention' if i%4==0 else 'train'
        x,y,_=episode(data_seed,32,int(delay),mode)
        tensor_hash(digest,x,y)
        lr=.003*(.1+.9*.5*(1+math.cos(math.pi*i/max(steps-1,1))))
        for g in optim.param_groups:g['lr']=lr
        model.train(); optim.zero_grad(set_to_none=True)
        logits,_=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        targets+=int((y!=-100).sum()); input_tokens+=x.numel()
        row=dict(step=i+1,loss=float(loss.detach()),lr=lr,grad_norm=float(norm),
                 seconds=time.perf_counter()-start,targets=targets,input_tokens=input_tokens)
        trace.append(row)
        if (i+1)%100==0:
            save(folder/'progress.json',row)
            print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
        if time.perf_counter()-start>limit:
            torch.save(model.state_dict(),folder/'interrupted.pt')
            save(folder/'TERMINAL.json',dict(status='TIME_LIMIT',completed_steps=i+1))
            raise RuntimeError('Trial reached fixed wall limit')
    training_wall=time.perf_counter()-start; training_cpu=time.process_time()-cpu0
    torch.save(model.state_dict(),folder/'final.pt')
    save(folder/'training.json',trace)
    # Pilot selection data never reused for frozen main confirmation.
    eval_seeds=[880001] if split=='pilot' else [990001,990002]
    delays=(32,128) if split=='pilot' else (32,256,1024)
    scores=evaluate(model,eval_seeds,delays)
    save(folder/'scores.json',scores)
    result=dict(**settings,status='COMPLETE',training_wall_seconds=training_wall,
                training_cpu_seconds=training_cpu,run_wall_seconds=time.perf_counter()-start,
                completed_steps=steps,targets=targets,input_tokens=input_tokens,
                training_stream_sha256=digest.hexdigest(),checkpoint_sha256=sha(folder/'final.pt'))
    save(folder/'TERMINAL.json',result)
    print(json.dumps(dict(arm=arm,seed=seed,status='COMPLETE',training_wall=training_wall,
                         scores=[{k:r[k] for k in ('mode','delay','accuracy')} for r in scores])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true')
    p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=991)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--split',default='pilot')
    p.add_argument('--limit',type=float,default=360.)
    a=p.parse_args()
    if a.verify:verify()
    else:train(a.arm,a.seed,a.steps,a.split,a.limit)
