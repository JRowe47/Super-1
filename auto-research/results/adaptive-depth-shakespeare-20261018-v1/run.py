"""Preregistered language pilot; all outputs are exclusive-created on the SSD."""
import argparse, hashlib, json, math, os, platform, random, resource, signal
import statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build, logits, SampledCurve
H=Path(__file__).resolve().parent; ROOT=H.parents[2]
C=json.loads((H/'CONFIG.json').read_text())
torch.set_num_threads(C['threads']); torch.set_num_interop_threads(C['threads'])
torch.use_deterministic_algorithms(True)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def event(out, value):
    with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value),flush=True)
def data():
    raw=(H/'corpus.txt').read_bytes(); n=len(raw)
    a=int(n*C['train_fraction']); b=int(n*C['selection_end_fraction'])
    return {k:torch.tensor(list(v),dtype=torch.long) for k,v in
            {'train':raw[:a],'selection':raw[a:b],'confirmation':raw[b:]}.items()}
def windows(values,count):
    starts=torch.arange(count)*(C['context']+1)
    assert int(starts[-1])+C['context']<len(values)
    offsets=starts[:,None]+torch.arange(C['context'])[None,:]
    return values[offsets].contiguous(),values[offsets+1].contiguous()
@torch.no_grad()
def evaluate(m,arm,xy,uniform=False):
    m.eval(); total=0.; digest=hashlib.sha256()
    x,y=xy
    for start in range(0,len(x),C['batch']):
        xx,yy=x[start:start+C['batch']],y[start:start+C['batch']]
        a=m(xx,uniform=True) if uniform else logits(m,arm,xx,yy)
        total+=float(F.cross_entropy(a.reshape(-1,C['vocab']),yy.reshape(-1),reduction='sum'))
        digest.update(a.numpy().tobytes())
    return {'ce':total/y.numel(),'targets':y.numel(),'logits_sha256':digest.hexdigest()}
def optimizer(m):
    groups=[{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':C['weight_decay']},
            {'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0.}]
    return torch.optim.AdamW(groups,lr=C['lr'],betas=tuple(C['betas']),eps=C['eps'])
def check_bounds(start,cpu,out):
    if time.monotonic()-start>C['wall_seconds'] or time.process_time()-cpu>C['cpu_seconds']:
        raise TimeoutError('Preregistered resource envelope exhausted')
    assert sum(p.stat().st_size for p in H.rglob('*') if p.is_file())<C['max_artifact_bytes']
def percentile(values,q):
    s=sorted(values); return s[math.ceil(q*len(s))-1]

def benchmark(out,arm,seed):
    start_cpu=time.process_time()
    initial=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    m=build(C,arm,seed); state=torch.load(out/f'{arm}-{seed}.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(state); del state; m.eval()
    x,y=windows(data()['selection'],1)
    times=[]; decode=[]
    with torch.inference_mode():
        for _ in range(C['benchmark_warmup']): logits(m,arm,x,y)
        for _ in range(C['benchmark_repeats']):
            t=time.perf_counter(); logits(m,arm,x,y); times.append((time.perf_counter()-t)*1000)
            prefix=x[:,:C['context']//2].clone()
            t=time.perf_counter()
            for _ in range(C['benchmark_decode_steps']):
                a=logits(m,arm,prefix,prefix); prefix=torch.cat((prefix,a[:,-1].argmax(-1,keepdim=True)),dim=1)
            decode.append((time.perf_counter()-t)*1000/C['benchmark_decode_steps'])
    return {'arm':arm,'seed':seed,'prefill_ms_p50':statistics.median(times),
            'prefill_ms_p95':percentile(times,.95),'decode_ms_per_token_p50':statistics.median(decode),
            'decode_ms_per_token_p95':percentile(decode,.95),'decode_tokens_per_second':1000/statistics.median(decode),
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'process_peak_before_model_bytes':initial,'weight_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),
            'persistent_kv_bytes':0,'cpu_seconds':time.process_time()-start_cpu,
            'scope':'fresh macOS CPU process peak including runtime/data/workspace; uncached full-prefix decode; batch 1; no GPU or cache throughput claim'}

def main(out):
    assert os.path.ismount('/Volumes/SP AI 01_16')
    assert str(out.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    out.mkdir(exist_ok=False)
    start,cpu=time.monotonic(),time.process_time()
    terminal={'status':'RUNNING','completed_training_rows':0}
    try:
        source=json.loads((H/'SOURCE_MANIFEST.json').read_text())
        assert all(sha(ROOT/k)==v for k,v in source.items()),'Source/data/protocol drift'
        save(out/'START.json',{'protocol_sha256':sha(H/'PROTOCOL.json'),'sources':source,
             'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),
             'runtime':{'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),
                        'device':'cpu','precision':'FP32','threads':C['threads']}})
        d=data(); selection=windows(d['selection'],C['selection_windows']); rows=[]; frozen={}
        for seed in C['seeds']:
            generator=torch.Generator().manual_seed(seed)
            starts=torch.randint(len(d['train'])-C['context'],(C['updates'],C['batch']),generator=generator)
            sampler_sha=hashlib.sha256(starts.numpy().tobytes()).hexdigest()
            order=list(C['arms']); random.Random(seed).shuffle(order)
            for arm in order:
                check_bounds(start,cpu,out)
                m=build(C,arm,seed); opt=optimizer(m)
                initial=evaluate(m,arm,selection); event(out,{'phase':'start','arm':arm,'seed':seed,'selection_ce':initial['ce']})
                wall0,cpu0=time.monotonic(),time.process_time()
                for step in range(C['updates']):
                    m.train(); offsets=starts[step,:,None]+torch.arange(C['context'])[None,:]
                    x=d['train'][offsets].contiguous(); y=d['train'][offsets+1].contiguous()
                    fraction=min((step+1)/C['warmup_updates'],1.)
                    if step>=C['warmup_updates']:
                        phase=(step-C['warmup_updates'])/(C['updates']-C['warmup_updates'])
                        fraction=C['minimum_lr_fraction']+(1-C['minimum_lr_fraction'])*(1+math.cos(math.pi*phase))/2
                    for group in opt.param_groups: group['lr']=C['lr']*fraction
                    opt.zero_grad(set_to_none=True); a=logits(m,arm,x,y)
                    loss=F.cross_entropy(a.reshape(-1,C['vocab']),y.reshape(-1))
                    assert torch.isfinite(loss),'Nonfinite loss'
                    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(m.parameters(),C['clip'])
                    assert torch.isfinite(norm),'Nonfinite gradient'
                    opt.step()
                    if (step+1)%C['selection_every']==0:
                        sel=evaluate(m,arm,selection)
                        event(out,{'phase':'train','arm':arm,'seed':seed,'update':step+1,
                                   'targets':(step+1)*C['batch']*C['context'],'train_ce':loss.item(),'selection_ce':sel['ce']})
                        check_bounds(start,cpu,out)
                path=out/f'{arm}-{seed}.pt'
                with path.open('xb') as f: torch.save(m.state_dict(),f)
                frozen[path.name]=sha(path)
                row={'arm':arm,'seed':seed,'parameters':sum(p.numel() for p in m.parameters()),
                     'updates':C['updates'],'target_exposures':C['updates']*C['batch']*C['context'],
                     'sampler_sha256':sampler_sha,'initial_selection':initial,'final_selection':sel,
                     'training_cpu_seconds_including_selection':time.process_time()-cpu0,
                     'training_wall_seconds_including_selection':time.monotonic()-wall0,
                     'checkpoint':path.name,'checkpoint_sha256':frozen[path.name]}
                if isinstance(m,SampledCurve):
                    with torch.no_grad():
                        _,t=m(selection[0],positions=True)
                    row['selection_positions_mean']=t.mean((0,1)).tolist()
                    row['selection_positions_std']=t.std((0,1)).tolist()
                    row['router_output_weight_norm']=m.router[-1].weight.norm().item()
                rows.append(row); terminal['completed_training_rows']=len(rows)
                del m,opt
        # No confirmation evaluation has occurred before this complete manifest.
        save(out/'FREEZE.json',{'checkpoints':frozen,'rows':rows,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        confirmation=windows(d['confirmation'],C['confirmation_windows'])
        for row in rows:
            arm,seed=row['arm'],row['seed']; path=out/row['checkpoint']
            assert sha(path)==row['checkpoint_sha256']
            m=build(C,arm,seed); m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            first=evaluate(m,arm,confirmation); second=evaluate(m,arm,confirmation)
            assert first==second,'FP32 duplicate drift'
            row['confirmation']=first; row['duplicate']=second
            if arm=='token': row['uniform_intervention_diagnostic']=evaluate(m,arm,confirmation,uniform=True)
            result=subprocess.run([sys.executable,'-B',__file__,'--output',str(out),'--benchmark',arm,'--seed',str(seed)],capture_output=True,text=True,check=True)
            row['benchmark']=json.loads(result.stdout.strip().splitlines()[-1])
            event(out,{'phase':'confirmed','arm':arm,'seed':seed,'ce':first['ce']})
            check_bounds(start,cpu,out)
        lookup={(r['arm'],r['seed']):r['confirmation']['ce'] for r in rows}
        contrasts={}
        for control in ['fixed','global','dense','nanogpt']:
            delta=[lookup['token',s]-lookup[control,s] for s in C['seeds']]
            mean=statistics.mean(delta); sem=statistics.stdev(delta)/math.sqrt(len(delta))
            contrasts[control]={'token_minus_control_nats':delta,'mean':mean,
                'descriptive_paired_t95':[mean-C['t_critical_df2']*sem,mean+C['t_critical_df2']*sem]}
        def passes(name):
            z=contrasts[name]
            return max(z['token_minus_control_nats'])<0 and z['mean']<=-C['minimum_gain_nats'] and z['descriptive_paired_t95'][1]<0
        learning=all(r['final_selection']['ce']<r['initial_selection']['ce'] for r in rows)
        primary=passes('fixed') and learning
        conditional=primary and passes('global')
        outcome='PILOT_SUPPORT_TOKEN_CONDITIONING' if conditional else 'NO_PROMOTION_INCONCLUSIVE_OR_NEGATIVE'
        save(out/'RESULT.json',{'rows':rows,'contrasts':contrasts,'learning_check':learning,
             'primary_fixed_comparison_passed':primary,'token_specific_control_passed':conditional,
             'nanoGPT_comparison_passed':conditional and passes('nanogpt') and passes('dense'),
             'outcome':outcome,'scope':'Tiny Shakespeare byte-level short-budget CPU pilot, not SOTA or general language superiority'})
        terminal.update(status='COMPLETED',outcome=outcome,checkpoint_count=len(frozen),duplicate_exact=True)
    except BaseException as exc:
        terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
                        artifact_bytes=sum(p.stat().st_size for p in H.rglob('*') if p.is_file()))
        save(out/'TERMINAL.json',terminal)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',choices=C['arms']); parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.benchmark: print(json.dumps(benchmark(args.output,args.benchmark,args.seed)))
    else: main(args.output)
