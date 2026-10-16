"""Fresh recovery of the interrupted context/capacity screen, not a resume."""
import argparse,hashlib,importlib.util,json,math,os,resource,sys,time
from pathlib import Path
PROCESS_BEGAN,CPU_BEGAN=time.perf_counter(),time.process_time()
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('context_parent',ROOT.parent/'gru-batch-wall-20261016-v1/experiment.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
import numpy as np
import torch
from torch.nn import functional as F
save,sha,optimizer,runtime=parent.save,parent.sha,parent.optimizer,parent.runtime

def build(q,p):
    torch.manual_seed(p['seed'])
    return parent.nano.GPT(parent.nano.GPTConfig(**q['config'])) if q['arm']=='nanogpt' else parent.build_gru(q['arm'],q['config'],p['seed'])

def batch(data,starts,context):
    xy=torch.from_numpy(np.stack([np.array(data[s+j:s+j+context+1],dtype=np.int64) for s in starts for j in range(0,2048,context)]))
    return xy[:,:-1].contiguous(),xy[:,1:].contiguous()

def eval_batch(data,ends,context):
    xy=torch.from_numpy(np.stack([np.array(data[e-context:e+1],dtype=np.int64) for e in ends]))
    return xy[:,:-1].contiguous(),xy[:,1:].contiguous()

@torch.no_grad()
def evaluate(m,q,data,ends,swap=False):
    m.eval();losses=[];correct=0;context=q['context']
    for i,e in enumerate(ends):
        x,y=eval_batch(data,[e],context)
        if swap and context>512:
            donor,_=eval_batch(data,[ends[(i+1)%len(ends)]],context)
            x[:,:context-512]=donor[:,:context-512]
        z,_=parent.forward(m,q['arm'],x,y);z=z[:,-512:];y=y[:,-512:]
        losses.append(float(F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))))
        correct+=int((z.argmax(-1)==y).sum())
    ce=float(np.mean(losses))
    return dict(ce=ce,perplexity=math.exp(ce),top1=correct/(512*len(ends)),targets=512*len(ends),window_ce=losses)

def update(m,opt,q,data,starts,lr):
    tick,cpu=time.perf_counter(),time.process_time();m.train()
    x,y=batch(data,starts,q['context'])
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);_,loss=parent.forward(m,q['arm'],x,y);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite training')
    opt.step()
    return dict(wall=time.perf_counter()-tick,cpu=time.process_time()-cpu,ce=float(loss.detach()),gradient_norm=float(norm),lr=lr)

def inputs(args):
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    for name,digest in p['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['data_hashes'].items():assert sha(args.data_dir/name)==digest,name
    assert sha(args.data_dir/'tokenizer.json')==p['tokenizer_sha256']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0,{0,1,2,3});resource.setrlimit(resource.RLIMIT_CPU,(700,710))
    train=np.memmap(args.data_dir/'train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(args.data_dir/'validation.bin',dtype=np.uint16,mode='r')
    assert len(train)==p['data']['train_tokens'] and len(val)==p['data']['validation_tokens']
    starts=np.random.default_rng(p['data_seed']).integers(0,len(train)-2048,size=p['max_blocks'],dtype=np.int64)
    assert hashlib.sha256(starts.tobytes()).hexdigest()==p['train_stream_sha256']
    return p,train,val,starts

def preflight(args,p,train,val,starts):
    targets=[batch(train,starts[:4],c)[1].flatten() for c in (512,1024,2048)]
    assert all(torch.equal(targets[0],x) for x in targets)
    tails=[eval_batch(val,p['final_ends'],c)[1][:,-512:] for c in (512,1024,2048)]
    assert all(torch.equal(tails[0],x) for x in tails)
    checks={}
    for q in p['runs']:
        m=build(q,p);assert sum(v.numel() for v in m.parameters())==q['parameters']
        x,y=batch(train,starts[:1],q['context']);assert x.is_contiguous() and y.is_contiguous()
        opt=optimizer(m,q['peak_lr']);_,loss=parent.forward(m,q['arm'],x,y);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());opt.step()
        row=dict(parameters=q['parameters'],finite_forward_backward_update=True)
        if q['arm']=='nanogpt':
            no_decay={id(v) for g in opt.param_groups if g['weight_decay']==0 for v in g['params']}
            assert id(m.transformer.wte.weight) in no_decay and id(m.transformer.wpe.weight) in no_decay
            row['token_and_position_tables_have_zero_decay']=True
        elif q['context']==2048:
            with torch.no_grad():
                a=m(x);b=m(x[:,:777]);c=m(x[:,777:],b.final_state)
                err=float((a.logits-torch.cat((b.logits,c.logits),1)).abs().max())
                se=max(float((a.final_state[k]-c.final_state[k]).abs().max()) for k in ('fast','segment','document','segment_sum'))
                assert err<1e-4 and se<1e-4,(q['run_id'],err,se)
                row.update(split_logits_max_error=err,split_state_max_error=se)
        checks[q['run_id']]=row;print('PREFLIGHT',q['run_id'],row,flush=True)
    save(ROOT/'PREFLIGHT.json',dict(checks=checks,identical_training_targets=True,identical_final_targets=True,canary_weights_discarded=True,runtime=runtime()))

def train_run(args,p,train,val,starts):
    q=next(x for x in p['runs'] if x['run_id']==args.run_id)
    out=ROOT/'execution'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists() and not (out/'FAILURE.json').exists(),'Terminal trial is immutable'
    m=build(q,p);opt=optimizer(m,q['peak_lr']);cp=out/'checkpoint.pt'
    state=dict(step=0,blocks=0,training_targets=0,train_wall_seconds=0.,train_cpu_seconds=0.,train=[],monitor=[])
    if cp.exists():
        old=torch.load(cp,map_location='cpu',weights_only=False)
        assert old['protocol_sha256']==sha(ROOT/'PROTOCOL.json') and old['run_id']==args.run_id
        m.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);torch.set_rng_state(old['torch_rng']);state=old['state']
    def checkpoint():
        tmp=cp.with_suffix('.tmp');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),state=state,
            recipe=q,run_id=args.run_id,protocol_sha256=sha(ROOT/'PROTOCOL.json')),tmp);tmp.replace(cp)
    def monitor():
        e=evaluate(m,q,val,p['monitor_ends']);r={k:state[k] for k in ('step','training_targets','train_wall_seconds','train_cpu_seconds')}
        r['validation_ce']=e['ce'];state['monitor'].append(r);save(out/'curves.json',state);print(args.run_id,r,flush=True)
    began,cpu_began=time.perf_counter(),time.process_time();setup_wall=began-PROCESS_BEGAN
    try:
        if not state['monitor']:
            save(out/'START.json',dict(runtime=runtime(),protocol_sha256=sha(ROOT/'PROTOCOL.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            monitor();checkpoint()
        recent=[r['wall'] for r in state['train'][-10:]] or [.5]
        budget=p['train_wall_seconds'];interval=p['monitor_interval_seconds'];next_monitor=(int(state['train_wall_seconds']//interval)+1)*interval
        while state['blocks']+q['blocks_per_update']<=len(starts):
            if state['train_wall_seconds']+max(.2,2*max(recent[-10:]))>=budget:break
            if time.perf_counter()-began>p['hard_limits']['worker_wall_seconds']:raise TimeoutError('Worker wall cap')
            progress=min(1.,state['train_wall_seconds']/budget)
            factor=max(.04,progress/.05) if progress<.05 else .1+.45*(1+math.cos(math.pi*(progress-.05)/.95))
            w=state['blocks'];n=q['blocks_per_update'];r=update(m,opt,q,train,starts[w:w+n],q['peak_lr']*factor)
            recent.append(r['wall']);state['step']+=1;state['blocks']+=n;state['training_targets']+=2048*n
            state['train_wall_seconds']+=r['wall'];state['train_cpu_seconds']+=r['cpu']
            state['train'].append(dict(step=state['step'],training_targets=state['training_targets'],train_wall_seconds=state['train_wall_seconds'],train_cpu_seconds=state['train_cpu_seconds'],**r))
            if state['train_wall_seconds']>=next_monitor:monitor();checkpoint();next_monitor+=interval
        if state['monitor'][-1]['step']!=state['step']:monitor()
        checkpoint();final=evaluate(m,q,val,p['final_ends'])
        swapped=evaluate(m,q,val,p['final_ends'],True) if q['context']>512 else None
        result=dict(**q,seed=p['seed'],updates=state['step'],training_targets=state['training_targets'],train_wall_seconds=state['train_wall_seconds'],
            train_cpu_seconds=state['train_cpu_seconds'],final=final,prefix_swapped=swapped,prefix_benefit_ce=swapped['ce']-final['ce'] if swapped else None,
            monitor=state['monitor'],run_wall_seconds=time.perf_counter()-began,run_cpu_seconds=time.process_time()-cpu_began,
            setup_wall_seconds=setup_wall,checkpoint_sha256=sha(cp),protocol_sha256=sha(ROOT/'PROTOCOL.json'),runtime=runtime(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(out/'RESULT.json',result);save(out/'curves.json',state);print('COMPLETE',args.run_id,state['step'],final['ce'],result['prefix_benefit_ce'],flush=True)
    except Exception as error:
        checkpoint();save(out/'FAILURE.json',dict(run_id=args.run_id,error_type=type(error).__name__,error=str(error),completed_updates=state['step'],train_wall_seconds=state['train_wall_seconds']))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','train']);ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--run-id')
    args=ap.parse_args();p,train,val,starts=inputs(args)
    (preflight if args.mode=='preflight' else train_run)(args,p,train,val,starts)
