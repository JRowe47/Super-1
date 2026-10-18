"""Sequential isolated-process memory measurements; no training or confirmation selection."""
import json,os,random,statistics,subprocess,sys,time,traceback
import run
H=run.H;C=run.C;out=H/'execution';folder=out/'memory';folder.mkdir(exist_ok=False);start=time.monotonic();terminal={'status':'RUNNING'}
try:
    jobs=[(arm,seed,context) for arm in C['patterns'] for seed in C['seeds'] for context in C['memory_contexts']];random.Random(C['canary_seed']).shuffle(jobs);records=[]
    for arm,seed,context in jobs:
        path=folder/f'{arm}-{seed}-{context}.json';env=dict(os.environ,TMPDIR=str(H/'tmp'),PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-B',str(H/'measure.py'),'--arm',arm,'--seed',str(seed),'--context',str(context),'--output',str(path)]
        result=subprocess.run(cmd,cwd=run.ROOT,env=env,text=True,capture_output=True,timeout=C['max_prefill_seconds']+60)
        with (folder/f'{arm}-{seed}-{context}.log').open('x') as f:f.write(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(f'Memory worker failed: {arm}/{seed}/{context}: {result.stderr}')
        record=json.loads(path.read_text());records.append(record);print(json.dumps({'arm':arm,'seed':seed,'context':context,'peak_rss':record['process_peak_rss_bytes'],'retained_bytes':record['retained_model_plus_state_bytes']}),flush=True)
    summary={};result=json.loads((out/'RESULT.json').read_text())
    for context in C['memory_contexts']:
        summary[str(context)]={}
        for arm in C['patterns']:
            ratios={key:[] for key in ['process_peak_rss_bytes','retained_model_plus_state_bytes','prefill_seconds','decode_ms_per_token']}
            for seed in C['seeds']:
                candidate=next(r for r in records if (r['arm'],r['seed'],r['context'])==(arm,seed,context));baseline=next(r for r in records if (r['arm'],r['seed'],r['context'])==('transformer',seed,context))
                for key in ratios:ratios[key].append(candidate[key]/baseline[key])
            summary[str(context)][arm]={'paired_ratios':ratios,'mean_ratios':{k:statistics.mean(v) for k,v in ratios.items()}}
    target=summary[str(max(C['memory_contexts']))]['relaxed'];memory_pass=max(target['paired_ratios']['process_peak_rss_bytes'])<=C['memory_target_ratio'];latency_pass=max(target['mean_ratios'][k] for k in ('prefill_seconds','decode_ms_per_token'))<=C['latency_max_ratio']
    run.save(out/'MEMORY.json',{'records':records,'summary':summary,'primary_arm':'relaxed','primary_context':max(C['memory_contexts']),'quality_pass':result['relaxed_quality_noninferiority_pass'],'total_memory_pass':memory_pass,'latency_pass':latency_pass,'combined_gate':memory_pass and latency_pass and result['relaxed_quality_noninferiority_pass'],'scope':'Quality is tested at256 only.8192 is untrained memory/cached-execution stress. Even a joint pilot pass would not establish long-context quality or SOTA.'})
    terminal.update(status='COMPLETED',workers=len(records))
except BaseException as e:terminal.update(status='FAILED_OR_INCOMPLETE',error=repr(e),traceback=traceback.format_exc());raise
finally:terminal['wall_seconds']=time.monotonic()-start;run.save(out/'MEMORY_TERMINAL.json',terminal)
