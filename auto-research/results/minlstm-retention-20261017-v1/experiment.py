import argparse, hashlib, json, math, os, platform, resource, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Model, ARMS
from tasks import episode

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def fingerprint(x,y):return hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()

@torch.no_grad()
def evaluate(model, split='main', primary_only=False):
    model.eval();out=[]
    seeds=[991101,991102] if split=='main' else [881101,881102]
    delays=[0,32,64,256,1024] if split=='main' else [32,64]
    for mode in ('retention','conflict','reopen'):
        for delay in delays:
            if primary_only and (mode!='retention' or delay!=256):continue
            accuracies=[];losses=[];datahash=hashlib.sha256();locked=[];opened=[]
            for base in seeds:
                for k in range(4):
                    x,y,meta=episode(base+100*k,batch=32,delay=delay,mode=mode)
                    datahash.update(x.numpy().tobytes());datahash.update(y.numpy().tobytes())
                    logits,_=model(x);logits=logits[:,-4:];targets=y[:,-4:]
                    correct=logits.argmax(-1)==targets
                    accuracies.extend(correct.float().mean(1).tolist())
                    losses.extend(F.cross_entropy(logits.reshape(-1,8),targets.reshape(-1),reduction='none').reshape(-1,4).mean(1).tolist())
                    if meta['locked'].any():locked.extend(correct[meta['locked']].float().tolist())
                    if mode=='reopen':opened.extend(correct[torch.arange(32),meta['reopen_key']].float().tolist())
            out.append(dict(mode=mode,delay=delay,episodes=len(accuracies),final_targets=4*len(accuracies),
                       accuracy=float(np.mean(accuracies)),nats_per_query=float(np.mean(losses)),
                       locked_accuracy=float(np.mean(locked)),reopened_accuracy=float(np.mean(opened)) if opened else None,
                       episode_accuracy=accuracies,data_sha256=datahash.hexdigest()))
    return out

def run(arm,seed,steps,split):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert steps==(protocol['training']['updates'] if split=='main' else protocol['pilot']['updates'])
    registration=json.loads((ROOT/'REGISTRATION.json').read_text())
    assert sha(ROOT/'PROTOCOL.json')==registration['protocol_sha256']
    for name,expected in registration['source_sha256'].items():assert sha(ROOT/name)==expected,name
    output=ROOT/split/f'{arm}_s{seed}';output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cpu=time.process_time();status='FAILED'
    save(output/'START.json',dict(arm=arm,seed=seed,split=split,steps=steps,unix=time.time(),protocol_sha256=sha(ROOT/'PROTOCOL.json')))
    try:
        model=Model(arm);opt=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
        rng=np.random.default_rng(seed+123456);stream=hashlib.sha256();trace=[]
        token_count=target_count=0;model_seconds=0;mode_counts={m:0 for m in ('retention','conflict','reopen')}
        start_train=time.perf_counter();cpu_train=time.process_time()
        for step in range(1,steps+1):
            delay=int(rng.choice([8,16,32,64]));mode=str(rng.choice(list(mode_counts),p=[.75,.125,.125]))
            x,y,_=episode(700000000+seed*10000+step,batch=32,delay=delay,mode=mode)
            stream.update(x.numpy().tobytes());stream.update(y.numpy().tobytes())
            token_count+=x.numel();target_count+=int((y!=-100).sum());mode_counts[mode]+=1
            rate=.0003+.5*(.003-.0003)*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))
            for g in opt.param_groups:g['lr']=rate
            t=time.process_time();model.train();opt.zero_grad(set_to_none=True)
            logits,_=model(x);loss=F.cross_entropy(logits.reshape(-1,8),y.reshape(-1),ignore_index=-100)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();model_seconds+=time.process_time()-t
            trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(gn),delay=delay,mode=mode,
                              train_cpu_seconds=time.process_time()-cpu_train,model_cpu_seconds=model_seconds,
                              train_wall_seconds=time.perf_counter()-start_train,tokens=token_count,targets=target_count,lr=rate))
            if step%100==0:
                save(output/'progress.json',trace[-1]);print(json.dumps(trace[-1]),flush=True)
            if time.perf_counter()-start>480:raise TimeoutError('480-second per-trial budget')
        training=dict(updates=steps,episodes=steps*32,input_tokens=token_count,target_queries=target_count,
                      stream_sha256=stream.hexdigest(),mode_batches=mode_counts,train_cpu_seconds=time.process_time()-cpu_train,
                      model_cpu_seconds=model_seconds,train_wall_seconds=time.perf_counter()-start_train,
                      accounting=model.accounting(),last_loss=trace[-1]['loss'])
        # Freeze final checkpoint before confirmation scoring; no best-seed or checkpoint selection.
        torch.save(dict(model=model.state_dict(),arm=arm,seed=seed,step=steps,protocol_sha256=sha(ROOT/'PROTOCOL.json')),output/'final.pt')
        save(output/'CHECKPOINT_FREEZE.json',dict(sha256=sha(output/'final.pt'),unix=time.time()))
        save(output/'training.json',trace);save(output/'training_summary.json',dict(**training,trace=[r for r in trace if r['step']%100==0]))
        save(output/'scores.json',evaluate(model,split));status='COMPLETE'
    finally:
        save(output/'TERMINAL.json',dict(status=status,arm=arm,seed=seed,steps_requested=steps,
             wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
             peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,unix=time.time()))

def main():
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--seed',type=int,required=True)
    a.add_argument('--split',choices=['pilot','main'],required=True);a.add_argument('--steps',type=int,required=True)
    args=a.parse_args();run(args.arm,args.seed,args.steps,args.split)
if __name__=='__main__':main()
