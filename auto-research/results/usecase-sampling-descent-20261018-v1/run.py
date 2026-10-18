"""Benchmark, gradient-tune, validate, repeat; confirmation is scored only after freeze."""
import argparse,hashlib,json,math,os,platform,random,statistics,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from policy import build,freeze_base,fingerprint
from tasks import load_data,batch,evaluation
H=Path(__file__).resolve().parent;ROOT=H.parents[2];C=json.loads((H/'CONFIG.json').read_text())
torch.set_num_threads(C['threads']);torch.set_num_interop_threads(C['threads']);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def checkpoint(p,v):
    with p.open('xb') as f:torch.save(v,f)
def log(out,v):
    with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
    print(json.dumps(v),flush=True)
def loss(pred,y,mask):return F.cross_entropy(pred[mask],y[mask])
@torch.no_grad()
def score(model,xy,uniform=False):
    model.eval();x,y,mask=xy;total=0.;correct=0;targets=0;digest=hashlib.sha256()
    for i in range(0,len(x),C['batch']):
        a=model(x[i:i+C['batch']],uniform=uniform);sel=mask[i:i+C['batch']];target=y[i:i+C['batch']][sel]
        pred=a[sel];total+=F.cross_entropy(pred,target,reduction='sum').item();correct+=(pred.argmax(-1)==target).sum().item();targets+=target.numel();digest.update(pred.numpy().tobytes())
    return {'ce':total/targets,'accuracy':correct/targets,'targets':targets,'logits_sha256':digest.hexdigest()}
def bounds(out,start,cpu):
    if time.monotonic()-start>C['wall_seconds'] or time.process_time()-cpu>C['cpu_seconds']:raise TimeoutError('Resource budget exhausted')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<C['max_artifact_bytes'],'Artifact budget'
def schedule():return torch.optim.lr_scheduler.ReduceLROnPlateau
@torch.no_grad()
def bench(m,xy):
    x=xy[0][:1];m.eval()
    for _ in range(C['benchmark_warmup']):m(x)
    times=[]
    for _ in range(C['benchmark_repeats']):
        start=time.perf_counter();m(x);times.append((time.perf_counter()-start)*1000)
    return {'prefill_batch1_ms_p50':statistics.median(times),'prefill_batch1_ms_p95':sorted(times)[math.ceil(.95*len(times))-1],
            'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'sampling_parameters':sum(p.numel() for p in m.router.parameters()),
            'scope':'CPU FP32 full 64-position forward; no KV cache or total-memory claim'}
def main(out):
    assert os.path.ismount('/Volumes/SP AI 01_16') and str(out.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    out.mkdir(exist_ok=False);start,cpu=time.monotonic(),time.process_time();terminal={'status':'RUNNING'}
    try:
        manifest=json.loads((H/'SOURCE_MANIFEST.json').read_text());assert all(sha(ROOT/k)==v for k,v in manifest.items())
        save(out/'START.json',{'protocol_sha256':sha(H/'PROTOCOL.json'),'source_manifest':manifest,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'runtime':{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','threads':C['threads'],'precision':'FP32'}})
        d=load_data(C);selection={t:evaluation(d,C,t,'selection') for t in C['tasks']};base_records=[]
        for seed in C['seeds']:
            m=build(C,seed);opt=torch.optim.AdamW(m.parameters(),lr=C['base_lr'],betas=tuple(C['betas']),eps=C['eps'],weight_decay=C['weight_decay'])
            generators={t:torch.Generator().manual_seed(seed+1000*j) for j,t in enumerate(C['tasks'])}
            initial={t:score(m,selection[t]) for t in C['tasks']};samplers={t:hashlib.sha256() for t in C['tasks']}
            order=C['tasks']*C['base_updates_per_task'];random.Random(seed).shuffle(order)
            for step,task in enumerate(order):
                m.train();x,y,mask=batch(d,C,task,'train',generators[task]);samplers[task].update(x.numpy().tobytes())
                opt.zero_grad(set_to_none=True);l=loss(m(x),y,mask);assert torch.isfinite(l);l.backward()
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),C['clip']);assert torch.isfinite(norm);opt.step()
                if (step+1)%C['base_updates_per_task']==0:
                    scores={t:score(m,selection[t]) for t in C['tasks']};log(out,{'phase':'base','seed':seed,'updates':step+1,'selection':scores});bounds(out,start,cpu)
            path=out/f'base-{seed}.pt';checkpoint(path,{'base':m.base.state_dict()})
            rec={'seed':seed,'checkpoint':path.name,'sha256':sha(path),'base_fingerprint':fingerprint(m.base),'initial_selection':initial,'final_selection':scores,'updates_per_task':C['base_updates_per_task'],'sampler_sha256':{k:v.hexdigest() for k,v in samplers.items()}}
            base_records.append(rec);del m,opt
        save(out/'BASE_FREEZE.json',base_records)
        chosen=[];history=[]
        for base in base_records:
            seed=base['seed'];state=torch.load(out/base['checkpoint'],weights_only=True)['base'];candidates={}
            specifications=[('shared_token','token',list(C['tasks']))]+[(task+'_'+kind,mode,[task]) for task in C['tasks'] for kind,mode in [('global','global'),('token','token')]]
            for name,mode,tasks in specifications:
                m=build(C,seed,mode,state);params=freeze_base(m)
                opt=torch.optim.Adam(params,lr=C['sampling_lr'],betas=tuple(C['betas']),eps=C['eps'])
                scheduler=schedule()(opt,mode='min',factor=C['lr_reduction'],patience=0,threshold=C['validation_min_delta'],threshold_mode='abs')
                scores={t:score(m,selection[t]) for t in tasks};criterion=statistics.mean(v['ce'] for v in scores.values())
                path=out/f'{name}-{seed}-round0.pt';checkpoint(path,{'router':m.router.state_dict(),'optimizer':opt.state_dict(),'base_sha256':base['sha256'],'round':0})
                candidates[name]={'model':m,'optimizer':opt,'scheduler':scheduler,'mode':mode,'tasks':tasks,'best_metric':criterion,'best_path':path.name,'best_round':0}
                scheduler.step(criterion)
                history.append({'seed':seed,'candidate':name,'round':0,'selection':scores,'criterion':criterion,'checkpoint':path.name,'sha256':sha(path),'lr':C['sampling_lr'],'base_fingerprint':fingerprint(m.base)})
            for round_id in range(1,C['rounds']+1):
                names=list(candidates);random.Random(seed+round_id).shuffle(names)
                for name in names:
                    z=candidates[name];m=z['model'];opt=z['optimizer'];tasks=z['tasks'];grads=[];samplers={t:hashlib.sha256() for t in tasks}
                    generators={t:torch.Generator().manual_seed(seed+round_id*10000+C['tasks'].index(t)*1000) for t in tasks}
                    order=tasks*C['updates_per_round_per_task'];random.Random(seed+round_id).shuffle(order)
                    lr=opt.param_groups[0]['lr']
                    for task in order:
                        m.train();x,y,mask=batch(d,C,task,'tune',generators[task]);samplers[task].update(x.numpy().tobytes())
                        opt.zero_grad(set_to_none=True);l=loss(m(x),y,mask);assert torch.isfinite(l);l.backward()
                        norm=torch.nn.utils.clip_grad_norm_(m.router.parameters(),C['clip']);assert torch.isfinite(norm);grads.append(float(norm));opt.step()
                    assert all(p.grad is None for p in m.base.parameters());assert fingerprint(m.base)==base['base_fingerprint']
                    scores={t:score(m,selection[t]) for t in tasks};criterion=statistics.mean(v['ce'] for v in scores.values())
                    path=out/f'{name}-{seed}-round{round_id}.pt';checkpoint(path,{'router':m.router.state_dict(),'optimizer':opt.state_dict(),'base_sha256':base['sha256'],'round':round_id})
                    if criterion<z['best_metric']:
                        z.update(best_metric=criterion,best_path=path.name,best_round=round_id)
                    z['scheduler'].step(criterion)
                    row={'seed':seed,'candidate':name,'round':round_id,'selection':scores,'criterion':criterion,'checkpoint':path.name,'sha256':sha(path),'lr':lr,'next_lr':opt.param_groups[0]['lr'],'gradient_norm_mean':statistics.mean(grads),'gradient_nonzero_steps':sum(g>0 for g in grads),'updates_this_round':len(order),'updates_per_task_this_round':C['updates_per_round_per_task'],'sampler_sha256':{k:v.hexdigest() for k,v in samplers.items()},'base_fingerprint':fingerprint(m.base),'best_round_so_far':z['best_round']}
                    history.append(row);log(out,dict(phase='sampling',**row));bounds(out,start,cpu)
            for name,z in candidates.items():chosen.append({'seed':seed,'candidate':name,'mode':z['mode'],'tasks':z['tasks'],'best_round':z['best_round'],'checkpoint':z['best_path'],'sha256':sha(out/z['best_path']),'selection_criterion':z['best_metric'],'base_checkpoint':base['checkpoint'],'base_sha256':base['sha256'],'base_fingerprint':base['base_fingerprint']})
            del candidates
        save(out/'SAMPLING_HISTORY.json',history);save(out/'FREEZE.json',{'selected':chosen,'base_records':base_records,'all_rounds':[{k:r[k] for k in ['seed','candidate','round','checkpoint','sha256']} for r in history]})
        # All use cases and candidates are fixed before scoring reserved confirmation.
        results=[];confirmation={t:evaluation(d,C,t,'confirmation') for t in C['tasks']}
        for base in base_records:
            seed=base['seed'];state=torch.load(out/base['checkpoint'],weights_only=True)['base']
            variants=[{'candidate':'uniform','mode':'fixed','tasks':C['tasks'],'checkpoint':None,'best_round':0}]+[z for z in chosen if z['seed']==seed]
            for z in variants:
                m=build(C,seed,z['mode'],state)
                if z['checkpoint']:m.router.load_state_dict(torch.load(out/z['checkpoint'],weights_only=True)['router'])
                assert fingerprint(m.base)==base['base_fingerprint']
                for task in z['tasks']:
                    first=score(m,confirmation[task]);second=score(m,confirmation[task]);assert first==second
                    row={'seed':seed,'task':task,'candidate':z['candidate'],'best_round':z['best_round'],'checkpoint':z['checkpoint'],'confirmation':first,'duplicate':second,'benchmark':bench(m,selection[task])}
                    with torch.no_grad():_,positions=m(selection[task][0],positions=True)
                    mask=selection[task][2];selected=positions[mask];row['positions_mean']=selected.mean(0).tolist();row['positions_std']=selected.std(0).tolist()
                    if z['candidate']==task+'_token':row['uniform_intervention']=score(m,confirmation[task],uniform=True)
                    results.append(row);log(out,{'phase':'confirmed','task':task,'candidate':z['candidate'],'seed':seed,'metrics':first})
        contrasts={};lookup={(r['candidate'],r['task'],r['seed']):r for r in results}
        for task in C['tasks']:
            contrasts[task]={}
            for comparator in ['uniform','shared_token',task+'_global']:
                gaps=[lookup[task+'_token',task,s]['confirmation']['ce']-lookup[comparator,task,s]['confirmation']['ce'] for s in C['seeds']]
                mean=statistics.mean(gaps);sem=statistics.stdev(gaps)/math.sqrt(len(gaps))
                contrasts[task][comparator]={'gaps_nats_per_target':gaps,'mean':mean,'paired_t95':[mean-C['t95_df2']*sem,mean+C['t95_df2']*sem],'familywise_interval':[mean-C['bonferroni_t_df2']*sem,mean+C['bonferroni_t_df2']*sem]}
        decisions={}
        for task in C['tasks']:
            def passed(control):
                z=contrasts[task][control];return max(z['gaps_nats_per_target'])<0 and z['mean']<=-C['minimum_gain_nats'] and z['familywise_interval'][1]<0
            if task=='text':learned=all(b['final_selection'][task]['ce']<b['initial_selection'][task]['ce']*(1-C['minimum_text_relative_gain']) for b in base_records)
            else:learned=all(b['final_selection'][task]['accuracy']>=C['minimum_task_accuracy'] for b in base_records)
            decisions[task]={'base_learning_qualified':learned,'improves_frozen_uniform':learned and passed('uniform'),'specialization_supported':learned and passed('uniform') and passed('shared_token'),'token_conditioning_supported':learned and passed('uniform') and passed('shared_token') and passed(task+'_global')}
        save(out/'RESULT.json',{'rows':results,'contrasts':contrasts,'decisions':decisions,'scope':'Frozen-backbone sampling optimization across one text corpus and two synthetic tasks. No overall nanoGPT or SOTA claim.'})
        terminal.update(status='COMPLETED',base_checkpoints=len(base_records),sampler_round_checkpoints=len(history),confirmation_rows=len(results),duplicate_exact=True,backbone_unchanged=True,decisions=decisions)
    except BaseException as e:
        terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,artifact_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()));save(out/'TERMINAL.json',terminal)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
