"""Text-only continuation, matched minibatches, validation selection, sealed confirmation."""
import argparse, hashlib, importlib.util, json, math, os, statistics, subprocess, time, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent
ROOT=H.parents[2]
PRIOR=H.parent/'usecase-sampling-descent-20261018-v1'
C=json.loads((H/'CONFIG.json').read_text())
spec=importlib.util.spec_from_file_location('policy', PRIOR/'policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def checkpoint(p,v):
    with p.open('xb') as f: torch.save(v,f)
def setup():
    torch.set_num_threads(C['threads']);torch.set_num_interop_threads(C['threads']);torch.use_deterministic_algorithms(True)
def data(split):
    raw=(H.parent/'adaptive-depth-shakespeare-20261018-v1/corpus.txt').read_bytes()
    a,b=C['text_regions'][split]
    return torch.tensor(list(raw[a:b]),dtype=torch.long)
def windows(d,indices):
    offsets=indices[:,None]+torch.arange(C['context'])[None,:]
    return d[offsets].contiguous(),d[offsets+1].contiguous()
def evaluation(split):
    return windows(data(split),torch.arange(C[split+'_examples'])*(C['context']+1))
@torch.no_grad()
def score(m,xy):
    m.eval();x,y=xy;total=0.;correct=0;digest=hashlib.sha256()
    for i in range(0,len(x),C['batch']):
        logits=m(x[i:i+C['batch']]);target=y[i:i+C['batch']]
        total+=F.cross_entropy(logits.reshape(-1,C['vocab']),target.reshape(-1),reduction='sum').item()
        correct+=(logits.argmax(-1)==target).sum().item();digest.update(logits.numpy().tobytes())
    return {'ce':total/y.numel(),'accuracy':correct/y.numel(),'targets':y.numel(),'logits_sha256':digest.hexdigest()}
def references():
    return [r for r in json.loads((PRIOR/'execution/FREEZE.json').read_text())['selected'] if r['candidate']=='shared_token']
def model(ref,router=None):
    state=torch.load(PRIOR/'execution'/ref['base_checkpoint'],weights_only=True)['base']
    m=policy.build(C,ref['seed'],'token',state);policy.freeze_base(m)
    path=router or PRIOR/'execution'/ref['checkpoint']
    m.router.load_state_dict(torch.load(path,weights_only=True)['router'])
    assert policy.fingerprint(m.base)==ref['base_fingerprint']
    return m

def run(out):
    assert os.path.ismount('/Volumes/SP AI 01_16') and str(out.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    setup();out.mkdir(exist_ok=False);wall,cpu=time.monotonic(),time.process_time();terminal={'status':'RUNNING'}
    try:
        source=json.loads((H/'SOURCE_MANIFEST.json').read_text())
        assert all(sha(ROOT/k)==v for k,v in source.items())
        save(out/'START.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'protocol_sha256':sha(H/'PROTOCOL.json'),'source_manifest':source,'torch':torch.__version__,'pid':os.getpid()})
        selected=evaluation('selection');tune=data('tune');history=[];choices=[]
        for ref in references():
            seed=ref['seed'];baseline=score(model(ref),selected)
            choices.append({'seed':seed,'arm':'incumbent','round':0,'checkpoint':str((PRIOR/'execution'/ref['checkpoint']).relative_to(ROOT)),'selection':baseline,'sha256':ref['sha256']})
            for arm,lr in C['arms'].items():
                m=model(ref);opt=torch.optim.Adam(m.router.parameters(),lr=lr,betas=tuple(C['betas']),eps=C['eps'])
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=C['lr_reduction'],patience=0,threshold=C['validation_min_delta'],threshold_mode='abs')
                scheduler.step(baseline['ce']);best=dict(choices[-1]) if choices[-1]['arm']=='incumbent' else dict(next(r for r in choices if r['seed']==seed and r['arm']=='incumbent'))
                best['arm']=arm
                for rnd in range(1,C['rounds']+1):
                    generator=torch.Generator().manual_seed(seed+rnd*10000);digest=hashlib.sha256();grads=[]
                    used_lr=opt.param_groups[0]['lr'];m.train()
                    for step in range(C['updates_per_round']):
                        x,y=windows(tune,torch.randint(len(tune)-C['context'],(C['batch'],),generator=generator));digest.update(x.numpy().tobytes())
                        opt.zero_grad(set_to_none=True);logits=m(x);loss=F.cross_entropy(logits.reshape(-1,C['vocab']),y.reshape(-1));assert torch.isfinite(loss);loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(m.router.parameters(),C['clip']);assert torch.isfinite(norm);grads.append(float(norm));opt.step()
                        if time.process_time()-cpu>C['cpu_seconds'] or time.monotonic()-wall>C['wall_seconds']:raise TimeoutError('Compute budget exhausted')
                    assert all(p.grad is None for p in m.base.parameters()) and policy.fingerprint(m.base)==ref['base_fingerprint']
                    metric=score(m,selected);scheduler.step(metric['ce'])
                    path=out/f'{arm}-{seed}-round{rnd}.pt'
                    checkpoint(path,{'router':m.router.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'base_sha256':ref['base_sha256'],'round':rnd})
                    row={'seed':seed,'arm':arm,'round':rnd,'checkpoint':str(path.relative_to(ROOT)),'sha256':sha(path),'selection':metric,'lr':used_lr,'next_lr':opt.param_groups[0]['lr'],'batch_sha256':digest.hexdigest(),'gradient_norm_mean':statistics.mean(grads),'base_fingerprint':policy.fingerprint(m.base)}
                    if metric['ce']<best['selection']['ce']:best=row.copy()
                    history.append(row);save(out/f'{arm}-{seed}-round{rnd}.json',row);print(json.dumps(row),flush=True)
                    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < C['max_artifact_bytes']
                choices.append(best)
        # One arm chosen using aggregate validation, before any confirmation reads.
        arm_means={a:statistics.mean(r['selection']['ce'] for r in choices if r['arm']==a) for a in ['incumbent']+list(C['arms'])}
        winner=min(arm_means,key=arm_means.get)
        save(out/'FREEZE.json',{'choices':choices,'winner':winner,'selection_means':arm_means,'history':history})
        confirmation=evaluation('confirmation');rows=[]
        for ref in references():
            for choice in [r for r in choices if r['seed']==ref['seed']]:
                m=model(ref,ROOT/choice['checkpoint']);metrics=score(m,confirmation)
                assert metrics==score(m,confirmation)
                rows.append(dict(choice,confirmation=metrics))
        gaps=[next(r for r in rows if r['seed']==s and r['arm']==winner)['confirmation']['ce']-next(r for r in rows if r['seed']==s and r['arm']=='incumbent')['confirmation']['ce'] for s in C['seeds']]
        mean=statistics.mean(gaps);sem=statistics.stdev(gaps)/math.sqrt(len(gaps));interval=[mean-C['t95_df2']*sem,mean+C['t95_df2']*sem]
        promoted=winner!='incumbent' and max(gaps)<0 and mean<=-C['minimum_gain_nats'] and interval[1]<0
        save(out/'RESULT.json',{'rows':rows,'winner_on_validation':winner,'paired_gaps':gaps,'mean_gap':mean,'paired_t95':interval,'promoted':promoted,'scope':'One validation-selected contrast on new held-out Shakespeare bytes; no broad text or inference speed claim.'})
        terminal.update(status='COMPLETED',round_checkpoints=len(history),confirmation_rows=len(rows),backbone_unchanged=True,promoted=promoted)
    except BaseException as e:
        terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        save(out/'TERMINAL.json',terminal)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
