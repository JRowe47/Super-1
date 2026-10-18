"""Freeze choices before fresh quality confirmation; preserve every seed and round."""
import argparse,hashlib,json,math,os,random,statistics,subprocess,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build,tensor_bytes
H=Path(__file__).resolve().parent;ROOT=H.parents[2];C=json.loads((H/'CONFIG.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def checkpoint(p,value):
    with p.open('xb') as f:torch.save(value,f)
def setup():
    torch.set_num_threads(C['threads']);torch.set_num_interop_threads(C['threads']);torch.use_deterministic_algorithms(True)
def data(split):
    a,b=C['text_regions'][split];return torch.tensor(list((H/'corpus.txt').read_bytes()[a:b]),dtype=torch.long)
def windows(d,starts):
    ix=starts[:,None]+torch.arange(C['context'])[None,:];return d[ix].contiguous(),d[ix+1].contiguous()
def evaluation(split):return windows(data(split),torch.arange(C[split+'_examples'])*(C['context']+1))
@torch.no_grad()
def score(m,xy):
    m.eval();x,y=xy;loss=0.;correct=0;digest=hashlib.sha256()
    for i in range(0,len(x),C['batch']):
        logits,_=m(x[i:i+C['batch']]);targets=y[i:i+C['batch']]
        loss+=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='sum').item();correct+=(logits.argmax(-1)==targets).sum().item();digest.update(logits.numpy().tobytes())
    return {'ce':loss/y.numel(),'accuracy':correct/y.numel(),'targets':y.numel(),'logits_sha256':digest.hexdigest()}
def bounds(out,cpu,start):
    if time.process_time()-cpu>C['cpu_seconds'] or time.monotonic()-start>C['wall_seconds']:raise TimeoutError('Budget exhausted')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>C['max_artifact_bytes']:raise RuntimeError('Artifact budget exceeded')
def contrast(rows,arm,reference):
    gaps=[next(r['confirmation']['ce'] for r in rows if r['seed']==s and r['arm']==arm)-next(r['confirmation']['ce'] for r in rows if r['seed']==s and r['arm']==reference) for s in C['seeds']]
    mean=statistics.mean(gaps);sem=statistics.stdev(gaps)/math.sqrt(len(gaps));return {'gaps':gaps,'mean':mean,'paired_t95':[mean-C['t95_df2']*sem,mean+C['t95_df2']*sem],'mean_ce_ppl_ratio':math.exp(mean)}
def main(out):
    assert os.path.ismount('/Volumes/SP AI 01_16') and str(out.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    setup();out.mkdir(exist_ok=False);cpu,start=time.process_time(),time.monotonic();terminal={'status':'RUNNING'}
    try:
        manifest=json.loads((H/'SOURCE_MANIFEST.json').read_text());assert all(sha(ROOT/p)==s for p,s in manifest.items())
        save(out/'START.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'protocol_sha256':sha(H/'PROTOCOL.json'),'source_manifest':manifest,'torch':torch.__version__,'pid':os.getpid(),'device':'cpu'})
        train=data('train');selection=evaluation('selection');history=[];chosen=[]
        for seed in C['seeds']:
            arms=list(C['patterns']);random.Random(seed).shuffle(arms)
            for arm in arms:
                arm_cpu=time.process_time();m=build(C,arm,seed);opt=torch.optim.AdamW(m.parameters(),lr=C['lr'],betas=tuple(C['betas']),weight_decay=C['weight_decay']);g=torch.Generator().manual_seed(seed);digest=hashlib.sha256();best=None;initial=None
                for step in range(C['updates']+1):
                    if step:
                        m.train();x,y=windows(train,torch.randint(len(train)-C['context'],(C['batch'],),generator=g));digest.update(x.numpy().tobytes())
                        ratio=step/C['warmup'] if step<=C['warmup'] else C['min_lr_fraction']+(1-C['min_lr_fraction'])*.5*(1+math.cos(math.pi*(step-C['warmup'])/(C['updates']-C['warmup'])))
                        for group in opt.param_groups:group['lr']=C['lr']*ratio
                        opt.zero_grad(set_to_none=True);logits,_=m(x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten());assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),C['clip']);assert torch.isfinite(norm);opt.step()
                    if step%C['selection_every']==0:
                        metric=score(m,selection);path=out/f'{arm}-{seed}-step{step}.pt';checkpoint(path,{'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'arm':arm,'seed':seed,'rng_state':g.get_state()})
                        row={'seed':seed,'arm':arm,'step':step,'checkpoint':path.name,'sha256':sha(path),'selection':metric,'batch_sha256':digest.hexdigest(),'parameter_count':sum(p.numel() for p in m.parameters()),'targets_seen':step*C['batch']*C['context'],'cpu_seconds_so_far':time.process_time()-arm_cpu}
                        if step==0:initial=metric
                        if best is None or metric['ce']<best['selection']['ce']:best=row.copy()
                        history.append(row);save(out/f'{arm}-{seed}-step{step}.json',row);print(json.dumps(row),flush=True);bounds(out,cpu,start)
                state=torch.load(out/best['checkpoint'],weights_only=True)['model'];path=out/f'{arm}-{seed}-inference.pt';checkpoint(path,{'model':state,'arm':arm,'seed':seed,'selected_step':best['step']});best.update(inference_checkpoint=path.name,inference_sha256=sha(path),initial_selection=initial,completed_updates=C['updates'],completed_targets=C['updates']*C['batch']*C['context'],completed_cpu_seconds=time.process_time()-arm_cpu);chosen.append(best)
                del m,opt,state
        save(out/'FREEZE.json',{'selected':chosen,'history':history})
        confirmation=evaluation('confirmation');rows=[]
        for choice in chosen:
            m=build(C,choice['arm'],choice['seed']);m.load_state_dict(torch.load(out/choice['inference_checkpoint'],weights_only=True)['model']);metric=score(m,confirmation);assert metric==score(m,confirmation);rows.append(dict(choice,confirmation=metric))
        contrasts={name:contrast(rows,arm,ref) for name,arm,ref in [('relaxed_minus_compact','relaxed','compact'),('relaxed_minus_transformer','relaxed','transformer'),('compact_minus_transformer','compact','transformer')]}
        z=contrasts['relaxed_minus_transformer'];quality_pass=z['paired_t95'][1]<=math.log1p(C['quality_max_relative_ppl_increase']) and all(math.exp(g)<=1+C['quality_max_relative_ppl_increase'] for g in z['gaps'])
        z=contrasts['relaxed_minus_compact'];relaxation_pass=z['mean']<=-C['relaxation_minimum_ce_gain'] and z['paired_t95'][1]<0 and max(z['gaps'])<0
        save(out/'RESULT.json',{'rows':rows,'contrasts':contrasts,'relaxed_quality_noninferiority_pass':quality_pass,'relaxation_improvement_pass':relaxation_pass,'memory_gate':'pending separate isolated-process measurement','scope':'Small Shakespeare byte pilot; quality only at trained256 context. Larger contexts are memory mechanics only. No original-hierarchy, nanoGPT speedrun or SOTA replication.'})
        terminal.update(status='COMPLETED',confirmation_rows=len(rows),round_checkpoints=len(history),quality_noninferiority_pass=quality_pass)
    except BaseException as e:terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc());raise
    finally:terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start);save(out/'TERMINAL.json',terminal)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
