"""Learn discrete depth preferences and continuous coordinates on frozen text weights."""
import argparse,hashlib,importlib.util,json,math,os,statistics,subprocess,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;ROOT=H.parents[2];C=json.loads((H/'CONFIG.json').read_text());PRIOR=H.parent/'usecase-sampling-descent-20261018-v1'
spec=importlib.util.spec_from_file_location('depth_model',H/'model.py');dm=importlib.util.module_from_spec(spec);spec.loader.exec_module(dm)
spec=importlib.util.spec_from_file_location('four_step',H.parent/'text-sampling-library-20261018-v1/run.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
sha=parent.sha;save=parent.save;checkpoint=parent.checkpoint

def setup():parent.setup()
def references():return parent.references()
def model(ref,depth,path=None):
    torch.manual_seed(ref['seed']);m=dm.SampledCurve(C,'token',depth)
    m.base.load_state_dict(torch.load(PRIOR/'execution'/ref['base_checkpoint'],weights_only=True)['base'])
    m.router.load_state_dict(torch.load(path or PRIOR/'execution'/ref['checkpoint'],weights_only=True)['router'])
    parent.policy.freeze_base(m);assert parent.policy.fingerprint(m.base)==ref['base_fingerprint'];return m

def data(split):
    raw=(H.parent/'adaptive-depth-shakespeare-20261018-v1/corpus.txt').read_bytes();a,b=C['text_regions'][split]
    return torch.tensor(list(raw[a:b]),dtype=torch.long)
def windows(d,indices):
    offsets=indices[:,None]+torch.arange(C['context'])[None,:];return d[offsets].contiguous(),d[offsets+1].contiguous()
def evaluation(split):return windows(data(split),torch.arange(C[split+'_examples'])*(C['context']+1))
score=parent.score
@torch.no_grad()
def benchmark(m,x):
    m.eval();x=x[:1];times=[]
    for _ in range(C['benchmark_warmup']):m(x)
    for _ in range(C['benchmark_repeats']):
        start=time.perf_counter();m(x);times.append((time.perf_counter()-start)*1000)
    return {'prefill_batch1_ms_median':statistics.median(times),'executed_layers':m.execution_depth,'scope':'One CPU thread FP32 uncached64-byte prefill; full frozen backbone remains stored.'}

def run(out):
    assert os.path.ismount('/Volumes/SP AI 01_16') and str(out.resolve()).startswith('/Volumes/SP AI 01_16/CodexWorkspaces/')
    setup();out.mkdir(exist_ok=False);wall,cpu=time.monotonic(),time.process_time();terminal={'status':'RUNNING'}
    try:
        source=json.loads((H/'SOURCE_MANIFEST.json').read_text());assert all(sha(ROOT/k)==v for k,v in source.items())
        save(out/'START.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),'protocol_sha256':sha(H/'PROTOCOL.json'),'source_manifest':source,'pid':os.getpid(),'torch':torch.__version__})
        tune=data('tune');selection=evaluation('selection');profiles=[];policies=[];history=[]
        for ref in references():
            seed=ref['seed'];models={d:model(ref,d) for d in C['depths']};opts={d:torch.optim.Adam(m.router.parameters(),lr=C['router_lr'],betas=tuple(C['betas']),eps=C['eps']) for d,m in models.items() if d>2}
            depth_logits=torch.nn.Parameter(torch.zeros(len(C['depths'])));depth_opt=torch.optim.Adam([depth_logits],lr=C['depth_lr'],betas=tuple(C['betas']),eps=C['eps'])
            baseline=score(models[max(C['depths'])],selection);incumbent={'seed':seed,'depth':max(C['depths']),'round':0,'checkpoint':str((PRIOR/'execution'/ref['checkpoint']).relative_to(ROOT)),'sha256':ref['sha256'],'selection':baseline}
            best_policy=dict(incumbent,origin='incumbent');best_by_depth={}
            for rnd in range(C['rounds']+1):
                if rnd:
                    g=torch.Generator().manual_seed(seed+rnd*10000);batch_digest=hashlib.sha256()
                    for step in range(C['updates_per_round']):
                        x,y=windows(tune,torch.randint(len(tune)-C['context'],(C['batch'],),generator=g));batch_digest.update(x.numpy().tobytes());losses=[]
                        for depth,m in models.items():
                            m.train();opt=opts.get(depth)
                            if opt:opt.zero_grad(set_to_none=True)
                            logits=m(x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten());assert torch.isfinite(loss);losses.append(loss.detach())
                            if opt:
                                loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.router.parameters(),C['clip']);assert torch.isfinite(norm);opt.step()
                        # Discrete execution is represented by a differentiable categorical expectation during training.
                        depth_opt.zero_grad(set_to_none=True);objective=(depth_logits.softmax(0)*torch.stack(losses)).sum();objective.backward();depth_opt.step()
                        if time.process_time()-cpu>C['cpu_seconds'] or time.monotonic()-wall>C['wall_seconds']:raise TimeoutError('Compute budget exhausted')
                probability=depth_logits.detach().softmax(0).tolist();chosen=C['depths'][int(depth_logits.argmax())]
                rows=[]
                for depth,m in models.items():
                    assert parent.policy.fingerprint(m.base)==ref['base_fingerprint'] and all(p.grad is None for p in m.base.parameters())
                    path=out/f'depth{depth}-{seed}-round{rnd}.pt'
                    checkpoint(path,{'router':m.router.state_dict(),'optimizer':opts[depth].state_dict() if depth in opts else None,'base_sha256':ref['base_sha256'],'execution_depth':depth,'round':rnd})
                    row={'seed':seed,'depth':depth,'round':rnd,'checkpoint':str(path.relative_to(ROOT)),'sha256':sha(path),'selection':score(m,selection)}
                    rows.append(row)
                    if depth not in best_by_depth or row['selection']['ce']<best_by_depth[depth]['selection']['ce']:best_by_depth[depth]=row
                checkpoint(out/f'depth-selector-{seed}-round{rnd}.pt',{'logits':depth_logits.detach(),'optimizer':depth_opt.state_dict(),'depths':C['depths'],'round':rnd})
                selected=next(r for r in rows if r['depth']==chosen)
                if selected['selection']['ce']<best_policy['selection']['ce']:best_policy=dict(selected,origin='learned',depth_probabilities=probability)
                record={'seed':seed,'round':rnd,'depth_probabilities':probability,'argmax_depth':chosen,'selection':rows,'batch_sha256':batch_digest.hexdigest() if rnd else None}
                history.append(record);save(out/f'round-{seed}-{rnd}.json',record);print(json.dumps({'seed':seed,'round':rnd,'depth_probabilities':probability,'argmax_depth':chosen,'selection_ce':{r['depth']:r['selection']['ce'] for r in rows}}),flush=True)
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<C['max_artifact_bytes']
            profiles.extend(best_by_depth.values());policies.append({'seed':seed,'incumbent':incumbent,'selected':best_policy,'final_probabilities':probability,'final_argmax_depth':chosen})
        save(out/'FREEZE.json',{'profiles':profiles,'policies':policies,'history':history})
        confirmation=evaluation('confirmation');rows=[]
        for ref in references():
            p=next(p for p in policies if p['seed']==ref['seed'])
            variants=[dict(r,kind='profile') for r in profiles if r['seed']==ref['seed']]+[dict(p['incumbent'],kind='incumbent'),dict(p['selected'],kind='selected')]
            for row in variants:
                m=model(ref,row['depth'],ROOT/row['checkpoint']);metrics=score(m,confirmation);assert metrics==score(m,confirmation)
                rows.append(dict(row,confirmation=metrics,benchmark=benchmark(m,selection[0])))
        gaps=[next(r for r in rows if r['seed']==s and r['kind']=='selected')['confirmation']['ce']-next(r for r in rows if r['seed']==s and r['kind']=='incumbent')['confirmation']['ce'] for s in C['seeds']]
        mean=statistics.mean(gaps);sem=statistics.stdev(gaps)/math.sqrt(len(gaps));interval=[mean-C['t95_df2']*sem,mean+C['t95_df2']*sem];promoted=max(gaps)<0 and mean<=-C['minimum_gain_nats'] and interval[1]<0
        save(out/'RESULT.json',{'rows':rows,'paired_gaps':gaps,'mean_gap':mean,'paired_t95':interval,'promoted':promoted,'scope':'Quality-only workload depth selector trained on text; lower-depth profiles are explicit tradeoffs, not validated quality substitutes.'})
        terminal.update(status='COMPLETED',confirmation_rows=len(rows),router_checkpoints=len(profiles)*(C['rounds']+1),selector_checkpoints=len(policies)*(C['rounds']+1),promoted=promoted)
    except BaseException as e:
        terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        terminal.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall);save(out/'TERMINAL.json',terminal)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
