"""Replay completed runs, reconcile traces and compare throughput with learning."""
import argparse,gzip,json,math,time,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,build,evaluate,batch,forward,inputs,save,sha

parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
args=parser.parse_args();args.threads=1
p,train,val,starts=inputs(args);selection=json.loads((ROOT/'SELECTION.json').read_text())
rows=[];checks={};manifest={};windows={}
for q in selection['runs']:
    out=ROOT/'execution'/q['run_id'];r=json.loads((out/'RESULT.json').read_text());cp=out/'checkpoint.pt'
    assert sha(cp)==r['checkpoint_sha256'];assert r['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    assert r['selection_sha256']==sha(ROOT/'SELECTION.json')
    assert r['batch']==q['batch'] and r['threads']==q['threads']
    old=torch.load(cp,map_location='cpu',weights_only=False)
    assert old['protocol_sha256']==r['protocol_sha256'] and old['selection_sha256']==r['selection_sha256']
    c=old['state'];assert len(c['train'])==r['updates']
    assert r['training_targets']==r['updates']*r['batch']*512
    assert abs(sum(x['wall'] for x in c['train'])-r['train_wall_seconds'])<1e-7
    assert abs(sum(x['cpu'] for x in c['train'])-r['train_cpu_seconds'])<1e-7
    assert all(x['step']==i+1 and math.isfinite(x['ce']) for i,x in enumerate(c['train']))
    max_step=max(x['wall'] for x in c['train'][-10:]);unused=p['train_wall_seconds']-r['train_wall_seconds']
    assert 0<=unused<=max(.2,2*max_step)+1e-6
    arm=q['arm'];m=build(arm,p);m.load_state_dict(old['model'])
    ev=evaluate(m,arm,val,p['final_starts'])
    error=max(abs(a-b) for a,b in zip(ev['window_ce'],r['final']['window_ce']))
    assert error<5e-6,(q['run_id'],error)
    # Final-weight batching parity, with the same mean-over-all-targets objective.
    x,y=batch(val,p['final_starts'][:2]);outputs=[];grads=[]
    for together in (True,False):
        m.zero_grad(set_to_none=True)
        if together:
            z,loss=forward(m,arm,x,y);outputs.append(z.detach());loss.backward()
        else:
            zs=[]
            for i in range(2):
                z,loss=forward(m,arm,x[i:i+1],y[i:i+1]);zs.append(z.detach());(loss/2).backward()
            outputs.append(torch.cat(zs))
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        grads.append(torch.cat([v.grad.flatten() for v in m.parameters()]))
    le=float((outputs[0]-outputs[1]).abs().max());ge=float((grads[0]-grads[1]).norm()/grads[1].norm())
    assert le<5e-5 and ge<2e-4,(q['run_id'],le,ge)
    checks[q['run_id']]=dict(replay_max_ce_error=error,batch_logits_error=le,batch_gradient_relative_l2=ge,
        trace_reconciled=True,budget_unused_seconds=unused)
    row={k:r[k] for k in ('run_id','arm','batch','threads','parameters','updates','training_targets','train_wall_seconds','train_cpu_seconds','run_wall_seconds','peak_rss_kib')}
    row.update(role=q['role'],ce=r['final']['ce'],perplexity=r['final']['perplexity'],tokens_per_wall_second=r['training_targets']/r['train_wall_seconds'],
        mean_cpu_cores=r['train_cpu_seconds']/r['train_wall_seconds'])
    rows.append(row);windows[q['run_id']]=r['final']['window_ce']
    manifest[q['run_id']]=dict(path=str(cp.relative_to(ROOT)),sha256=sha(cp),bytes=cp.stat().st_size)
    raw=(out/'curves.json').read_bytes();assert json.loads(raw)==c
    (out/'curves.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    print('CHECKED',q['run_id'],row['ce'],flush=True)
contrasts=[]
for arm in p['arms']:
    base=next(x for x in rows if x['arm']==arm and x['role']=='baseline')
    selected=next((x for x in rows if x['arm']==arm and x['role']=='throughput'),base)
    contrasts.append(dict(arm=arm,baseline_ce=base['ce'],selected_ce=selected['ce'],ce_change=selected['ce']-base['ce'],
        observed_throughput_ratio=selected['tokens_per_wall_second']/base['tokens_per_wall_second'],
        target_exposure_ratio=selected['training_targets']/base['training_targets'],optimizer_update_ratio=selected['updates']/base['updates']))
summary=dict(run_id=ROOT.name,status='COMPLETE_EXPLORATORY',seed=p['seed'],train_wall_budget_seconds=p['train_wall_seconds'],
    rows=sorted(rows,key=lambda x:(x['role'],x['ce'])),contrasts=contrasts,final_targets=16384,
    total_training_cpu_seconds=sum(r['train_cpu_seconds'] for r in rows),total_training_wall_seconds=sum(r['train_wall_seconds'] for r in rows),
    interpretation='Choose settings by benchmark throughput, then observe quality; a throughput winner is not automatically a learning winner',
    publication=dict(repository='JRowe47/Super-1',base='main',upstream_submission_permitted=False))
save(ROOT/'SUMMARY.json',summary);save(ROOT/'CHECKS.json',checks);save(ROOT/'CHECKPOINT_MANIFEST.json',manifest)
save(ROOT/'TERMINAL.json',dict(status='COMPLETE',runs=len(rows),checkpoint_replay_passed=True,batching_parity_passed=True,training_running=False,
    official_test_accessed=False,new_paid_resources=False,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
assert not args.archive.exists();args.archive.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for v in manifest.values():z.write(ROOT/v['path'],v['path'])
    z.writestr('CHECKPOINTS.json',json.dumps(manifest,indent=2)+'\n')
    z.writestr('README.txt','Extract under auto-research/results/gru-batch-wall-20261016-v1/. Source/results live in JRowe47/Super-1. Exact train/validation arrays are in prior gru_extended_checkpoints_20261016.zip, SHA256 c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6. Full model/AdamW/RNG/trace retained for all six runs.\n')
with zipfile.ZipFile(args.archive) as z:
    assert z.testzip() is None
    for v in manifest.values():assert hashlib.sha256(z.read(v['path'])).hexdigest()==v['sha256']
save(ROOT/'PRESERVATION.json',dict(archive=args.archive.name,bytes=args.archive.stat().st_size,sha256=sha(args.archive),
    contents='All final checkpoints, optimizer/RNG/traces; Git-backed source/results and previously saved data are not duplicated',status='VERIFIED',
    exact_data_archive='gru_extended_checkpoints_20261016.zip',exact_data_sha256='c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6'))
regpath=ROOT.parents[2]/'auto-research/RUN_REGISTRY.json';reg=json.loads(regpath.read_text());entry=next(x for x in reg['runs'] if x['run_id']==ROOT.name)
entry.update(status='COMPLETE_EXPLORATORY',selection_sha256=sha(ROOT/'SELECTION.json'),completed_runs=len(rows),summary={x['run_id']:x['ce'] for x in rows},publication_target='JRowe47/Super-1 main only')
save(regpath,reg);print(json.dumps(summary,indent=2))
