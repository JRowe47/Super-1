"""Preregistered paired text pilot; immutable rounds and freeze before confirmation."""
import argparse, contextlib, hashlib, importlib.util, io, json, math, os, random
import resource, statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
H = Path(__file__).resolve().parent
ROOT = H.parents[2]
sys.path.insert(0, str(ROOT))
from sampling_library.inference import CachedSampledCurve, cache_bytes
from sampling_library.hierarchical_kv import HierarchicalKV
C = json.loads((H / 'CONFIG.json').read_text())
SPEC = importlib.util.spec_from_file_location('frozen_curve', ROOT / 'auto-research/results/text-depth-library-20261018-v1/model.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    with p.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
def ckpt(p, value):
    with p.open('xb') as f: torch.save(value, f)
def setup():
    assert os.path.ismount('/Volumes/SP AI 01_16')
    assert str(H.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    torch.set_num_threads(C['threads']); torch.set_num_interop_threads(C['threads'])
    torch.use_deterministic_algorithms(True)
def build(arm, seed):
    torch.manual_seed(seed)
    if arm == 'nano':
        with contextlib.redirect_stdout(io.StringIO()):
            return MOD.nano.GPT(MOD.nano.GPTConfig(block_size=C['context'], vocab_size=C['vocab'], n_layer=C['layers'], n_head=C['heads'], n_embd=C['width'], dropout=0., bias=True))
    return MOD.SampledCurve(C, 'token', C['layers'])
def runtime(model, arm):
    return CachedSampledCurve(model, None if arm == 'exact' else lambda i,h,d: HierarchicalKV(h,d,**C['candidates'][arm]))
def logits(model, x, y=None):
    if isinstance(model, MOD.nano.GPT):
        # Targets enable the unmodified upstream full-logit training path.
        return model(x, x.contiguous() if y is None else y)[0]
    result = model(x)
    return result[0] if isinstance(result, tuple) else result
def corpus(split):
    a,b=C['text_regions'][split]
    return torch.tensor(list((H/'corpus.txt').read_bytes()[a:b]),dtype=torch.long)
def windows(d, starts):
    ix=starts[:,None]+torch.arange(C['context'])[None,:]
    return d[ix].contiguous(), d[ix+1].contiguous()
def examples(split):
    return windows(corpus(split), torch.arange(C[split+'_examples'])*(C['context']+1))
def fingerprint(m):
    h=hashlib.sha256()
    for k,p in m.state_dict().items(): h.update(k.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
@torch.no_grad()
def score(m, xy):
    m.eval(); x,y=xy; loss=tail=0.; digest=hashlib.sha256()
    for i in range(0,len(x),C['batch']):
        z=logits(m,x[i:i+C['batch']],y[i:i+C['batch']]); target=y[i:i+C['batch']]
        assert torch.isfinite(z).all()
        loss+=F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='sum').item()
        tail+=F.cross_entropy(z[:,C['context']//2:].reshape(-1,C['vocab']),target[:,C['context']//2:].reshape(-1),reduction='sum').item()
        digest.update(z.numpy().tobytes())
    return {'ce':loss/y.numel(),'tail_ce':tail/(y.numel()/2),'targets':y.numel(),'logits_sha256':digest.hexdigest()}
def bounds(out,cpu,wall):
    if time.process_time()-cpu>C['cpu_seconds'] or time.monotonic()-wall>C['wall_seconds']: raise TimeoutError('Preregistered budget exhausted')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>C['max_artifact_bytes']: raise RuntimeError('Artifact budget exceeded')
def record(out, name, m, opt, meta):
    p=out/(name+'.pt'); ckpt(p,{'model':m.state_dict(),'optimizer':None if opt is None else opt.state_dict()})
    row=dict(meta, checkpoint=p.name, sha256=sha(p)); save(out/(name+'.json'),row)
    print(json.dumps(row),flush=True); return row

def train(out):
    out.mkdir(exist_ok=False); cpu=time.process_time(); wall=time.monotonic(); terminal={'status':'RUNNING'}
    try:
        manifest=json.loads((H/'SOURCE_MANIFEST.json').read_text())
        assert all(sha(ROOT/p)==s for p,s in manifest.items())
        save(out/'START.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_manifest':manifest,'torch':torch.__version__,'device':'cpu','pid':os.getpid()})
        train_data=corpus('train'); selection=examples('selection'); bases=[]; hierarchy=[]; history=[]
        for seed in C['seeds']:
            arms=['nano','curve']; random.Random(seed).shuffle(arms)
            for arm in arms:
                start=time.process_time(); m=build(arm,seed); g=torch.Generator().manual_seed(seed); exposure=hashlib.sha256()
                opt=torch.optim.AdamW(m.parameters(),lr=C['base_lr'],betas=tuple(C['betas']),weight_decay=C['weight_decay']); best=None
                for step in range(C['base_updates']+1):
                    if step:
                        m.train(); x,y=windows(train_data,torch.randint(len(train_data)-C['context'],(C['batch'],),generator=g)); exposure.update(x.numpy().tobytes())
                        scale=min(step/25,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/C['base_updates'])))
                        for group in opt.param_groups: group['lr']=C['base_lr']*scale
                        opt.zero_grad(set_to_none=True); loss=F.cross_entropy(logits(m,x,y).flatten(0,1),y.flatten()); assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),C['clip']); opt.step()
                    if step%C['selection_every']==0:
                        row=record(out,f'{arm}-{seed}-step{step}',m,opt,{'arm':arm,'seed':seed,'step':step,'selection':score(m,selection),'parameters':sum(p.numel() for p in m.parameters()),'targets_seen':step*C['batch']*C['context'],'batch_sha256':exposure.hexdigest(),'cpu_seconds':time.process_time()-start})
                        history.append(row)
                        if best is None or row['selection']['ce']<best['selection']['ce']: best=row
                        bounds(out,cpu,wall)
                bases.append(best); del m,opt
            base=next(r for r in bases if r['arm']=='curve' and r['seed']==seed)
            for arm in C['candidates']:
                model=build('curve',seed); model.load_state_dict(torch.load(out/base['checkpoint'],weights_only=True)['model']); model.requires_grad_(False); model.eval(); before=fingerprint(model)
                m=runtime(model,arm).eval(); opt=torch.optim.Adam(m.memory.parameters(),lr=C['pool_lr']); g=torch.Generator().manual_seed(seed+10000); best=None; exposure=hashlib.sha256(); start=time.process_time()
                for round_id in range(C['pool_rounds']+1):
                    if round_id:
                        for _ in range(C['pool_updates_per_round']):
                            x,y=windows(train_data,torch.randint(len(train_data)-C['context'],(C['pool_batch'],),generator=g)); exposure.update(x.numpy().tobytes())
                            with torch.no_grad(): teacher=model(x).softmax(-1)
                            opt.zero_grad(set_to_none=True); z=logits(m,x); ce=F.cross_entropy(z.flatten(0,1),y.flatten()); kd=F.kl_div(z.log_softmax(-1),teacher,reduction='sum')/y.numel(); loss=ce+C['distillation_weight']*kd
                            assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(m.memory.parameters(),C['clip']); opt.step()
                            bounds(out,cpu,wall)
                    assert fingerprint(model)==before
                    row=record(out,f'{arm}-{seed}-round{round_id}',m.memory,opt,{'arm':arm,'seed':seed,'round':round_id,'base_checkpoint':base['checkpoint'],'base_sha256':base['sha256'],'backbone_fingerprint':before,'selection':score(m,selection),'pool_parameters':sum(p.numel() for p in m.memory.parameters()),'extra_targets_seen':round_id*C['pool_updates_per_round']*C['pool_batch']*C['context'],'batch_sha256':exposure.hexdigest(),'lr':opt.param_groups[0]['lr'],'cpu_seconds':time.process_time()-start})
                    history.append(row)
                    if best is None or row['selection']['ce']<best['selection']['ce']: best=row
                    else:
                        for group in opt.param_groups: group['lr']*=C['lr_reduction']
                hierarchy.append(best); del m,model,opt
        # Paired reference training batches must match, regardless of initialization.
        for seed in C['seeds']:
            digests=[r['batch_sha256'] for r in history if r['seed']==seed and r.get('step')==C['base_updates']]
            assert len(digests)==2 and len(set(digests))==1
        feasible=[]
        for arm in C['candidates']:
            good=all(next(r['selection']['ce'] for r in hierarchy if r['arm']==arm and r['seed']==s)<=next(r['selection']['ce'] for r in bases if r['arm']==ref and r['seed']==s)+math.log1p(C['quality_ppl_tolerance']) for s in C['seeds'] for ref in ('nano','curve'))
            if good: feasible.append(arm)
        capacity=lambda a:C['candidates'][a]['recent_capacity']+sum(C['candidates'][a]['level_capacities'])
        selected=min(feasible,key=capacity) if feasible else min(C['candidates'],key=lambda a:statistics.mean(r['selection']['ce'] for r in hierarchy if r['arm']==a))
        save(out/'FREEZE.json',{'bases':bases,'hierarchy':hierarchy,'history':history,'selected_capacity':selected,'selection_feasible':feasible,'confirmation_opened':False})
        confirmation=examples('confirmation'); rows=[]
        for row in bases:
            m=build(row['arm'],row['seed']); m.load_state_dict(torch.load(out/row['checkpoint'],weights_only=True)['model']); metric=score(m,confirmation); assert metric==score(m,confirmation); rows.append(dict(row,confirmation=metric))
        for best in hierarchy:
            for variant,choice in [('zero',next(r for r in history if r['arm']==best['arm'] and r['seed']==best['seed'] and r.get('round')==0)),('selected',best)]:
                model=build('curve',choice['seed']); model.load_state_dict(torch.load(out/choice['base_checkpoint'],weights_only=True)['model']); m=runtime(model,choice['arm']); m.memory.load_state_dict(torch.load(out/choice['checkpoint'],weights_only=True)['model']); metric=score(m,confirmation); assert metric==score(m,confirmation)
                rows.append(dict(choice,confirmation=metric,variant=variant))
        save(out/'RESULT.json',{'rows':rows,'selected_capacity':selected,'selection_feasible':feasible,'scope':'Fresh from-scratch Alice byte pilot. Quality at512 bytes only; not SOTA or long-context evidence.'})
        terminal.update(status='COMPLETED',confirmation_rows=len(rows),round_checkpoints=len(history))
    except BaseException as e:
        terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc()); raise
    finally:
        terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall); save(out/'TERMINAL.json',terminal)

def canary():
    torch.manual_seed(777); x=torch.randint(256,(2,C['context'])); m=build('curve',73); exact=runtime(m,'exact').eval()
    with torch.no_grad(): torch.testing.assert_close(logits(exact,x),m(x),atol=2e-5,rtol=2e-4)
    m.requires_grad_(False); before=fingerprint(m); h=runtime(m,'h16'); opt=torch.optim.Adam(h.memory.parameters(),lr=C['pool_lr']); initial=[p.clone() for p in h.memory.parameters()]; start=time.process_time()
    opt.zero_grad(); F.cross_entropy(logits(h,x).flatten(0,1),x.flatten()).backward(); opt.step()
    assert fingerprint(m)==before and any(not torch.equal(a,b) for a,b in zip(initial,h.memory.parameters()))
    with torch.no_grad():
        full,cache=h(x); a,state=h(x[:,:37]); b,state=h(x[:,37:],state); torch.testing.assert_close(full,torch.cat((a,b),1),atol=2e-5,rtol=2e-4)
        assert cache_bytes(cache)<=2*C['layers']*x.shape[0]*C['width']*16*4
    n=build('nano',73); assert logits(n,x).shape==(2,C['context'],C['vocab'])
    return {'status':'PASS','pool_step_cpu_seconds':time.process_time()-start,'exact_curve_parity':True,'frozen_backbone':True,'pool_gradient_update':True,'chunk_parity':True,'bounded_state':True,'upstream_nano_full_logits':True}

def memory(out, arm, seed, length, destination):
    freeze=json.loads((out/'FREEZE.json').read_text()); base=next(r for r in freeze['bases'] if r['arm']=='curve' and r['seed']==seed)
    model=build('curve',seed); model.load_state_dict(torch.load(out/base['checkpoint'],weights_only=True)['model']); m=runtime(model,arm).eval()
    if arm!='exact':
        choice=next(r for r in freeze['hierarchy'] if r['arm']==arm and r['seed']==seed); m.memory.load_state_dict(torch.load(out/choice['checkpoint'],weights_only=True)['model'])
    # Only training bytes, independent process for every seed/backend/length.
    data=corpus('train')[:length+C['decode_tokens']].unsqueeze(0); state=None
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,length,C['memory_chunk']):
            z,state=m(data[:,i:min(i+C['memory_chunk'],length)],state); assert torch.isfinite(z).all()
        prefill=time.perf_counter()-start; lat=[]
        for i in range(length,length+C['decode_tokens']):
            t=time.perf_counter(); z,state=m(data[:,i:i+1],state); lat.append(time.perf_counter()-t); assert torch.isfinite(z).all()
    factor=1 if sys.platform=='darwin' else 1024
    save(destination,{'arm':arm,'seed':seed,'length':length,'decode_tokens':C['decode_tokens'],'dtype':'float32','logical_cache_bytes':cache_bytes(state),'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor,'loaded_peak_rss_bytes':baseline*factor,'prefill_seconds':prefill,'decode_ms_median':statistics.median(lat)*1000,'scope':'Isolated process, actual retained KV plus total peak RSS. Above512 bytes is memory mechanics only; no long-context quality claim.'})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['canary','train','memory']); p.add_argument('--output',type=Path,default=H/'execution'); p.add_argument('--arm'); p.add_argument('--seed',type=int); p.add_argument('--length',type=int); p.add_argument('--destination',type=Path); a=p.parse_args(); setup()
    if a.mode=='canary': save(H/'CANARY.json',canary()); print((H/'CANARY.json').read_text())
    elif a.mode=='train': train(a.output)
    else: memory(a.output,a.arm,a.seed,a.length,a.destination)
