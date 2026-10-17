import json, time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, save, sha, evaluate
from model import Model, ARMS

def main():
    torch.set_num_threads(1)
    p=json.loads((ROOT/'PROTOCOL.json').read_text());seeds=p['design']['seeds']
    summaries={};replay=[];checkpoints=[]
    data_hashes={};train_hashes={}
    for arm in ARMS:
        all_scores=[];training=[]
        for seed in seeds:
            d=ROOT/'main'/f'{arm}_s{seed}'
            assert json.loads((d/'TERMINAL.json').read_text())['status']=='COMPLETE'
            t=json.loads((d/'training_summary.json').read_text());assert t['updates']==p['training']['updates']
            scores=json.loads((d/'scores.json').read_text());training.append({k:v for k,v in t.items() if k!='trace'})
            all_scores.append({(r['mode'],r['delay']):r for r in scores})
            signature=(t['stream_sha256'],t['input_tokens'],t['target_queries'])
            if seed in train_hashes:assert train_hashes[seed]==signature
            train_hashes[seed]=signature
            for r in scores:
                key=(r['mode'],r['delay'])
                if key in data_hashes:assert data_hashes[key]==r['data_sha256']
                data_hashes[key]=r['data_sha256']
            checkpoint=d/'final.pt';digest=sha(checkpoint)
            assert digest==json.loads((d/'CHECKPOINT_FREEZE.json').read_text())['sha256']
            saved=torch.load(checkpoint,weights_only=True);m=Model(arm);m.load_state_dict(saved['model'])
            rerun=evaluate(m,primary_only=True)[0];reference=all_scores[-1][('retention',256)]
            assert rerun==reference,(arm,seed)
            replay.append(dict(arm=arm,seed=seed,primary_exact_replay=True,checkpoint_sha256=digest))
            checkpoints.append(dict(path=str(checkpoint.relative_to(ROOT)),sha256=digest,bytes=checkpoint.stat().st_size))
        rows=[]
        for mode in ('retention','conflict','reopen'):
            for delay in p['evaluation']['delays']:
                r=[s[(mode,delay)] for s in all_scores]
                acc=[v['accuracy'] for v in r]
                rows.append(dict(mode=mode,delay=delay,accuracy_mean=float(np.mean(acc)),
                                 accuracy_seed_min=min(acc),accuracy_seed_max=max(acc),accuracy_per_seed=acc,
                                 nats_per_query_mean=float(np.mean([v['nats_per_query'] for v in r])),
                                 locked_accuracy_mean=float(np.mean([v['locked_accuracy'] for v in r])),
                                 reopened_accuracy_mean=float(np.mean([v['reopened_accuracy'] for v in r])) if mode=='reopen' else None))
        summaries[arm]=dict(accounting=Model(arm).accounting(),scores=rows,training=training,
            mean_train_cpu_seconds=float(np.mean([t['train_cpu_seconds'] for t in training])),
            mean_model_cpu_seconds=float(np.mean([t['model_cpu_seconds'] for t in training])))
    def score(arm,delay):return next(r for r in summaries[arm]['scores'] if r['mode']=='retention' and r['delay']==delay)
    gains=100*(np.array(score('minlstm64',256)['accuracy_per_seed'])-np.array(score('mingru64',256)['accuracy_per_seed']))
    competent=all(score(arm,64)['accuracy_mean']>=.90 for arm in ('mingru64','minlstm64'))
    success=competent and gains.mean()>=5 and np.all(gains>0)
    result=dict(run_id=ROOT.name,status='COMPLETE',seeds=seeds,arms=summaries,
       primary=dict(paired_gain_pp=gains.tolist(),mean_gain_pp=float(gains.mean()),competence_guard_pass=competent,
                    success_rule_met=bool(success),outcome='USEFUL_SYNTHETIC_GAIN' if success else 'NO_RETENTION_PROMOTION'),
       paired_training_streams_verified=True,paired_evaluation_streams_verified=True,
       train_cpu_seconds_total=sum(t['train_cpu_seconds'] for a in summaries.values() for t in a['training']),
       main_launcher=json.loads((ROOT/'main'/'LAUNCH_TERMINAL.json').read_text()),unix=time.time())
    checks=json.loads((ROOT/'VERIFICATION.json').read_text())['checks']
    fields=('output_max_abs','gradient_max_abs','input_gradient_max_abs','chunk_max_abs','optimizer_max_abs')
    def original_pass(row):return all(row[k] < (2e-12 if row['dtype']=='torch.float64' else 2e-6) for k in fields)
    fidelity=dict(equal_state_models_pass=all(original_pass(r) for r in checks if r['arm']!='lstm64'),
                  supplemental_lstm64_original_optimizer_parity='PASS' if all(original_pass(r) for r in checks if r['arm']=='lstm64') else 'FAIL',
                  audit='STRICT_FIDELITY_AUDIT.json')
    result['strict_fidelity']=fidelity
    result['public_publication']='EXPLICITLY_AUTHORIZED_BY_USER_2026-10-17'
    save(ROOT/'SUMMARY.json',result);save(ROOT/'REPLAY_CHECKS.json',dict(status='PASS',rows=replay))
    save(ROOT/'CHECKPOINT_MANIFEST.json',dict(checkpoints=checkpoints))
    save(ROOT/'TERMINAL.json',dict(status='COMPLETE',main_trials=12,updates_per_trial=3000,
         primary=result['primary'],checkpoint_replays=12,protocol_sha256=sha(ROOT/'PROTOCOL.json'),
         strict_fidelity=fidelity,public_publication=result['public_publication'],unix=time.time()))
    print(json.dumps(result['primary'],indent=2))
    for a in summaries:
        print(a, 'retention32/64/256/1024=',[round(100*score(a,d)['accuracy_mean'],2) for d in (32,64,256,1024)],
              'train_cpu_s=',round(summaries[a]['mean_train_cpu_seconds'],2))

if __name__=='__main__':main()
