"""Bounded quantization-aware recovery with immutable selection/confirmation receipts."""
import argparse, contextlib, copy, hashlib, importlib.util, io, json, math
import os, random, resource, statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent
ROOT=H.parents[2]
sys.path.insert(0,str(ROOT))
from sampling_library.packed_weights import PackedModel, export, fake_parameters
C=json.loads((H/'CONFIG.json').read_text())
P=ROOT/C['parent_run']; OUT=H/'execution'
spec=importlib.util.spec_from_file_location('frozen_curve',ROOT/C['model_source'])
MOD=importlib.util.module_from_spec(spec); spec.loader.exec_module(MOD)


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def checkpoint(p,value):
    with p.open('xb') as f: torch.save(value,f)
def setup():
    assert os.path.ismount('/Volumes/SP AI 01_16')
    assert str(H.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    torch.set_num_threads(C['threads']); torch.set_num_interop_threads(C['threads'])
    torch.use_deterministic_algorithms(True)
def build(arm,seed):
    torch.manual_seed(seed)
    if arm.startswith('nano'):
        with contextlib.redirect_stdout(io.StringIO()):
            m=MOD.nano.GPT(MOD.nano.GPTConfig(block_size=C['context'],vocab_size=C['vocab'],n_layer=C['layers'],n_head=C['heads'],n_embd=C['width'],dropout=0.,bias=True))
        kind='nano'
    else:
        m=MOD.SampledCurve(C,'token',C['layers']); kind='curve'
    record=next(r for r in json.loads((P/'execution/FREEZE.json').read_text())['bases'] if r['arm']==kind and r['seed']==seed)
    path=P/'execution'/record['checkpoint']; assert sha(path)==record['sha256']
    m.load_state_dict(torch.load(path,weights_only=True)['model']); return m.eval()
def forward(m,x,arm=None,positions=False):
    nano=isinstance(m,MOD.nano.GPT)
    args=(x,x.contiguous()) if nano else (x,)
    kwargs={} if nano else {'positions':positions}
    if arm is None: z=m(*args,**kwargs)
    else: z=torch.func.functional_call(m,fake_parameters(m,arm,C['group']),args,kwargs,tie_weights=True)
    return z[0] if nano else z

def data(split):
    a,b=C['regions'][split]
    return torch.tensor(list((P/'corpus.txt').read_bytes()[a:b]),dtype=torch.long)
def windows(d,starts):
    ix=starts[:,None]+torch.arange(C['context'])[None,:]
    return d[ix].contiguous(),d[ix+1].contiguous()
def examples(split): return windows(data(split),torch.arange(C[split+'_examples'])*(C['context']+1))
@torch.no_grad()
def score(fn,xy):
    x,y=xy; loss=0.; digest=hashlib.sha256()
    for i in range(0,len(x),C['batch']):
        z=fn(x[i:i+C['batch']]); yy=y[i:i+C['batch']]
        assert torch.isfinite(z).all()
        loss+=F.cross_entropy(z.flatten(0,1),yy.flatten(),reduction='sum').item()
        digest.update(z.numpy().tobytes())
    return dict(ce=loss/y.numel(),targets=y.numel(),logits_sha256=digest.hexdigest())

def canary():
    torch.manual_seed(901); x=torch.randint(256,(2,37)); rows=[]
    for arm in C['arms']:
        m=build(arm,C['seeds'][0]); packed=PackedModel(export(m,C,arm,C['group'],C['tile']))
        depths=[2,3,4] if arm.startswith('curve') else [None]
        for d in depths:
            if d: m.depth=m.execution_depth=d
            with torch.no_grad():
                a=forward(m,x,arm,positions=bool(d)); b=packed(x,positions=bool(d),depth=d)
                if d:
                    torch.testing.assert_close(a[1],b[1],atol=2e-6,rtol=2e-5); a,b=a[0],b[0]
                torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)
                # Causal prefix equivalence across different sequence lengths.
                z=packed(x[:,:19],depth=d) if d else packed(x[:,:19])
                torch.testing.assert_close(b[:,:19],z,atol=3e-5,rtol=3e-4)
            rows.append(dict(arm=arm,depth=d,max_logit_error=(a-b).abs().max().item(),payload=packed.parameter_bytes,max_decoded_values=packed.max_decoded_values))
        m.zero_grad(); loss=F.cross_entropy(forward(m,x,arm).flatten(0,1),x.flatten()); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    # Tail padding, zero groups, signed nibbles, aliases and round trip.
    w=torch.nn.Linear(7,3,bias=False); w.weight.data.zero_(); w.weight.data[0,0]=-1; w.weight.data[1,1]=1
    b=export(w,C,'nano_q4',C['group'],C['tile']); p=PackedModel(b)
    expected=fake_parameters(w,'nano_q4',C['group'])['weight']
    actual=p.get('weight',torch.arange(21).reshape(3,7))
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    try: p.get('weight')
    except ValueError: pass
    else: raise AssertionError('Dense materialization allowed')
    return dict(status='PASS',rows=rows,gradient_flow=True,causality=True,padding_and_signed_nibbles=True,scope='Mechanics only; no confirmation bytes read')

def bounds(cpu,wall):
    if time.process_time()-cpu>C['cpu_seconds'] or time.monotonic()-wall>C['wall_seconds']: raise TimeoutError('Preregistered budget exhausted')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>C['max_artifact_bytes']: raise RuntimeError('Artifact budget exceeded')

def train():
    OUT.mkdir(exist_ok=False); cpu=time.process_time(); wall=time.monotonic(); terminal={}
    try:
        manifest=json.loads((H/'SOURCE_MANIFEST.json').read_text())
        assert all(sha(ROOT/p)==s for p,s in manifest.items())
        save(OUT/'START.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,pid=os.getpid(),source_manifest_sha256=sha(H/'SOURCE_MANIFEST.json')))
        history=[]; selected=[]; selection=examples('selection'); train_data=data('train')
        for seed in C['seeds']:
            arms=list(C['arms']); random.Random(seed).shuffle(arms)
            for arm in arms:
                student=build(arm,seed); teacher=build(arm,seed).requires_grad_(False)
                opt=torch.optim.AdamW(student.parameters(),lr=C['lr'],betas=tuple(C['betas']),weight_decay=C['weight_decay'])
                gen=torch.Generator().manual_seed(seed+C['batch_seed_offset']); digest=hashlib.sha256(); best=None
                for rnd in range(C['rounds']+1):
                    if rnd:
                        for _ in range(C['updates_per_round']):
                            x,y=windows(train_data,torch.randint(len(train_data)-C['context'],(C['batch'],),generator=gen)); digest.update(x.numpy().tobytes())
                            with torch.no_grad(): target=forward(teacher,x).softmax(-1)
                            opt.zero_grad(set_to_none=True); z=forward(student,x,arm)
                            ce=F.cross_entropy(z.flatten(0,1),y.flatten())
                            kd=F.kl_div(z.log_softmax(-1),target,reduction='sum')/y.numel()
                            loss=ce+C['distillation_weight']*kd
                            assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),C['clip']); opt.step(); bounds(cpu,wall)
                    metric=score(lambda x:forward(student,x,arm),selection)
                    name=f'{arm}-{seed}-round{rnd}'; ck=OUT/(name+'.pt')
                    checkpoint(ck,dict(model=student.state_dict(),optimizer=opt.state_dict()))
                    row=dict(arm=arm,seed=seed,round=rnd,selection=metric,checkpoint=ck.name,sha256=sha(ck),batch_sha256=digest.hexdigest(),lr=opt.param_groups[0]['lr'],extra_targets=rnd*C['updates_per_round']*C['batch']*C['context'])
                    history.append(row); save(OUT/(name+'.json'),row); print(json.dumps(row),flush=True)
                    if best is None or metric['ce']<best['selection']['ce']: best=row
                    else:
                        for g in opt.param_groups: g['lr']*=C['lr_reduction']
                selected.append(best); del student,teacher,opt
        for seed in C['seeds']:
            assert len({r['batch_sha256'] for r in history if r['seed']==seed and r['round']==C['rounds']})==1
        save(OUT/'FREEZE.json',dict(selected=selected,history=history,confirmation_opened=False))
        confirmation=examples('confirmation'); rows=[]
        for best in selected:
            for variant,r in [('zero',next(r for r in history if r['seed']==best['seed'] and r['arm']==best['arm'] and r['round']==0)),('selected',best)]:
                m=build(r['arm'],r['seed']); m.load_state_dict(torch.load(OUT/r['checkpoint'],weights_only=True)['model'])
                path=OUT/f"{r['arm']}-{r['seed']}-{variant}.packed.pt"
                bundle=export(m,C,r['arm'],C['group'],C['tile']); checkpoint(path,bundle)
                packed=PackedModel(torch.load(path,weights_only=True))
                metric=score(packed,confirmation); assert metric==score(packed,confirmation)
                fake=score(lambda x:forward(m,x,r['arm']),confirmation)
                assert abs(metric['ce']-fake['ce'])<C['ce_parity_tolerance']
                rows.append(dict(r,variant=variant,confirmation=metric,fake_confirmation=fake,packed_file=path.name,packed_sha256=sha(path),file_bytes=path.stat().st_size,payload_bytes=packed.parameter_bytes,max_decoded_weight_bytes=packed.max_decoded_values*4))
            # Frozen original teacher on the same new confirmation for attribution.
            teacher=build(best['arm'],best['seed'])
            rows.append(dict(arm=best['arm'],seed=best['seed'],variant='teacher_fp32',confirmation=score(lambda x:forward(teacher,x),confirmation)))
        save(OUT/'RESULT.json',dict(rows=rows,scope='Tiny Alice byte language model,512 context; no long-context/SOTA/generalization or speed claim'))
        terminal=dict(status='COMPLETED',checkpoints=len(history),rows=len(rows))
    except BaseException as e:
        terminal=dict(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc()); raise
    finally:
        terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall); save(OUT/'TERMINAL.json',terminal)

def memory(path,destination):
    # This process instantiates no dense model, optimizer or checkpoint masters.
    start=time.perf_counter(); packed=PackedModel(torch.load(path,weights_only=True))
    x=data('train')[:C['context']].unsqueeze(0)
    with torch.inference_mode(): z=packed(x)
    assert torch.isfinite(z).all()
    factor=1 if sys.platform=='darwin' else 1024
    save(destination,dict(file=path.name,sha256=sha(path),payload_bytes=packed.parameter_bytes,file_bytes=path.stat().st_size,max_decoded_weight_bytes=packed.max_decoded_values*4,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor,load_and_forward_seconds=time.perf_counter()-start,scope='CPU full-sequence512 reference implementation; includes interpreter/activations and index scratch, no KV cache'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['canary','train','memory']); p.add_argument('--path',type=Path); p.add_argument('--destination',type=Path); a=p.parse_args(); setup()
    if a.mode=='canary': save(H/'CANARY.json',canary())
    elif a.mode=='train': train()
    else: memory(a.path,a.destination)
