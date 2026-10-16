"""Small CPU batching/thread screen, then frozen matched-training-wall-time runs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,gzip,hashlib,importlib.util,json,math,platform,resource,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'gru-extended-20261016-v1'
sys.path.insert(0,str(PARENT))
from models import build as build_gru,optimizer
spec=importlib.util.spec_from_file_location('batchwall_nanogpt',ROOT/'vendor/nanogpt/model.py')
nano=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nano;spec.loader.exec_module(nano)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def build(arm,p):
    torch.manual_seed(p['seed'])
    return nano.GPT(nano.GPTConfig(**p['configs'][arm])) if arm=='nanogpt' else build_gru(arm,p['configs'][arm],p['seed'])
def forward(m,arm,x,y):
    # Reuse nanoGPT's own CE. The older wrapper calculated and discarded it,
    # then calculated another identical CE. That extra forward work is absent here.
    if arm=='nanogpt':return m(x,y)
    z=m(x).logits
    return z,F.cross_entropy(z.flatten(0,1),y.flatten())
def batch(data,starts):
    xy=torch.from_numpy(np.stack([np.array(data[s:s+513],dtype=np.int64) for s in starts]))
    return xy[:,:-1].contiguous(),xy[:,1:].contiguous()
def update(m,opt,arm,data,starts,lr):
    tick,cpu=time.perf_counter(),time.process_time();m.train()
    x,y=batch(data,starts)
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);z,loss=forward(m,arm,x,y);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite training')
    opt.step()
    if hasattr(m,'constrain'):m.constrain()
    return dict(wall=time.perf_counter()-tick,cpu=time.process_time()-cpu,ce=float(loss.detach()),gradient_norm=float(norm),lr=lr)
@torch.no_grad()
def evaluate(m,arm,data,starts):
    m.eval();losses=[];correct=0
    for s in starts:
        x,y=batch(data,[s]);z,loss=forward(m,arm,x,y)
        losses.append(float(loss));correct+=int((z.argmax(-1)==y).sum())
    ce=float(np.mean(losses))
    return dict(ce=ce,perplexity=math.exp(ce),top1=correct/(512*len(starts)),targets=512*len(starts),window_ce=losses)
def runtime():
    return dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,platform=platform.platform(),
        threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),affinity=sorted(os.sched_getaffinity(0)),
        device='cpu',dtype='float32',cpu_model=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip())
def inputs(args):
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    for name,digest in p['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['data_hashes'].items():assert sha(args.data_dir/name)==digest,name
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0,set(range(args.threads)))
    resource.setrlimit(resource.RLIMIT_CPU,(600,610))
    train=np.memmap(args.data_dir/'train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(args.data_dir/'validation.bin',dtype=np.uint16,mode='r')
    starts=np.random.default_rng(p['data_seed']).integers(0,len(train)-512,size=p['max_windows'],dtype=np.int64)
    assert hashlib.sha256(starts.tobytes()).hexdigest()==p['train_stream_sha256']
    return p,train,val,starts
def preflight(args,p,train,val,starts):
    checks={}
    for arm in p['arms']:
        m=build(arm,p);assert sum(v.numel() for v in m.parameters())==p['parameters'][arm]
        x,y=batch(train,starts[:2]);grads=[];outputs=[]
        for batched in (True,False):
            m.zero_grad(set_to_none=True)
            if batched:
                z,loss=forward(m,arm,x,y);outputs.append(z.detach());loss.backward()
            else:
                zs=[]
                for i in range(2):
                    z,loss=forward(m,arm,x[i:i+1],y[i:i+1]);zs.append(z.detach());(loss/2).backward()
                outputs.append(torch.cat(zs))
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
            grads.append(torch.cat([v.grad.flatten() for v in m.parameters()]))
        le=float((outputs[0]-outputs[1]).abs().max());ge=float((grads[0]-grads[1]).norm()/grads[1].norm())
        assert le<5e-5 and ge<2e-4,(arm,le,ge)
        checks[arm]=dict(batch_vs_individual_logits_error=le,batch_vs_individual_gradient_relative_l2=ge,parameters=p['parameters'][arm])
        if arm=='nanogpt':
            z,loss=forward(m,arm,x,y);err=float((loss-F.cross_entropy(z.flatten(0,1),y.flatten())).detach())
            assert err==0;checks[arm]['reused_vs_recomputed_ce_error']=err
        print('PREFLIGHT',arm,checks[arm],flush=True)
    save(ROOT/'PREFLIGHT.json',dict(checks=checks,runtime=runtime(),weights_discarded=True))
def benchmark(args,p,train,val,starts):
    out=ROOT/'benchmark'/f'threads{args.threads}.json'
    assert not out.exists(),'Completed benchmark is immutable'
    cells=[(a,b) for a in p['arms'] for b in p['batches']]
    np.random.default_rng(p['benchmark_order_seed']+args.threads).shuffle(cells)
    rows=[]
    for arm,b in cells:
        m=build(arm,p);opt=optimizer(m,.002);samples=[]
        for i in range(p['benchmark_warmups']+p['benchmark_repeats']):
            r=update(m,opt,arm,train,starts[:b],.002)
            if i>=p['benchmark_warmups']:samples.append(r)
        median=statistics.median(x['wall'] for x in samples)
        row=dict(arm=arm,batch=b,threads=args.threads,median_step_wall_seconds=median,
                 tokens_per_wall_second=b*512/median,median_step_cpu_seconds=statistics.median(x['cpu'] for x in samples),samples=samples)
        rows.append(row);print('BENCH',arm,b,args.threads,round(row['tokens_per_wall_second']),flush=True)
        save(out.with_name(out.stem+'.progress.json'),dict(rows=rows,runtime=runtime()))
        del m,opt
    save(out,dict(rows=rows,runtime=runtime(),protocol_sha256=sha(ROOT/'PROTOCOL.json')))
def train_arm(args,p,train,val,starts):
    selection=json.loads((ROOT/'SELECTION.json').read_text());q=next(x for x in selection['runs'] if x['run_id']==args.run_id)
    assert args.threads==q['threads']
    arm,b=q['arm'],q['batch'];out=ROOT/'execution'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists(),'Completed run is immutable'
    cp=out/'checkpoint.pt';m=build(arm,p);opt=optimizer(m,.002)
    state=dict(step=0,windows=0,training_targets=0,train_wall_seconds=0.,train_cpu_seconds=0.,train=[],monitor=[])
    if cp.exists():
        old=torch.load(cp,map_location='cpu',weights_only=False)
        assert old['selection_sha256']==sha(ROOT/'SELECTION.json') and old['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
        m.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);torch.set_rng_state(old['torch_rng']);state=old['state']
    def checkpoint():
        tmp=cp.with_suffix('.tmp');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            state=state,arm=arm,config=p['configs'][arm],protocol_sha256=sha(ROOT/'PROTOCOL.json'),selection_sha256=sha(ROOT/'SELECTION.json')),tmp);tmp.replace(cp)
    def monitor():
        e=evaluate(m,arm,val,p['monitor_starts']);row={k:state[k] for k in ('step','windows','training_targets','train_wall_seconds','train_cpu_seconds')}
        row.update(validation_ce=e['ce'],validation_top1=e['top1']);state['monitor'].append(row)
        save(out/'curves.json',state);print(args.run_id,row,flush=True)
    began=time.perf_counter();cpu_began=time.process_time()
    if not state['monitor']:
        save(out/'START.json',dict(runtime=runtime(),selection_sha256=sha(ROOT/'SELECTION.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        monitor();checkpoint()
    recent=[q['benchmark_step_wall_seconds']];budget=p['train_wall_seconds']
    next_monitor=(int(state['train_wall_seconds']//15)+1)*15
    while state['windows']+b<=len(starts):
        reserve=max(.2,2*max(recent[-10:]))
        if state['train_wall_seconds']+reserve>=budget:break
        if time.perf_counter()-began>180:checkpoint();raise TimeoutError('Per-run wall cap reached')
        progress=min(1.,max(0.,state['train_wall_seconds']/budget))
        factor=max(.04,progress/.05) if progress<.05 else .1+.9*.5*(1+math.cos(math.pi*(progress-.05)/.95))
        lr=.002*factor;w=state['windows'];r=update(m,opt,arm,train,starts[w:w+b],lr)
        recent.append(r['wall']);state['step']+=1;state['windows']+=b;state['training_targets']+=512*b
        state['train_wall_seconds']+=r['wall'];state['train_cpu_seconds']+=r['cpu']
        state['train'].append(dict(step=state['step'],training_targets=state['training_targets'],train_wall_seconds=state['train_wall_seconds'],train_cpu_seconds=state['train_cpu_seconds'],**r))
        if state['train_wall_seconds']>=next_monitor:
            monitor();checkpoint();next_monitor+=15
    if state['monitor'][-1]['step']!=state['step']:monitor()
    checkpoint();final=evaluate(m,arm,val,p['final_starts'])
    result=dict(run_id=args.run_id,arm=arm,batch=b,threads=args.threads,parameters=p['parameters'][arm],seed=p['seed'],
        updates=state['step'],training_targets=state['training_targets'],train_wall_seconds=state['train_wall_seconds'],train_cpu_seconds=state['train_cpu_seconds'],
        final=final,monitor=state['monitor'],budget_unused_seconds=budget-state['train_wall_seconds'],run_wall_seconds=time.perf_counter()-began,
        run_cpu_seconds=time.process_time()-cpu_began,checkpoint_sha256=sha(cp),protocol_sha256=sha(ROOT/'PROTOCOL.json'),selection_sha256=sha(ROOT/'SELECTION.json'),runtime=runtime(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(out/'RESULT.json',result);save(out/'curves.json',state)
    print('COMPLETE',args.run_id,result['updates'],result['train_wall_seconds'],final['ce'],flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','benchmark','train'])
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--threads',type=int,default=1);parser.add_argument('--run-id')
    args=parser.parse_args();p,train,val,starts=inputs(args)
    {'preflight':preflight,'benchmark':benchmark,'train':train_arm}[args.mode](args,p,train,val,starts)
if __name__=='__main__':main()
