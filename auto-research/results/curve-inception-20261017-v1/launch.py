import argparse, concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
from experiment import ROOT, save, sha
from model import ARMS

def main():
    a=argparse.ArgumentParser();a.add_argument('--split',choices=['pilot','main'],required=True)
    a.add_argument('--steps',type=int,required=True);a.add_argument('--workers',type=int,default=3)
    args=a.parse_args()
    seeds=[991] if args.split=='pilot' else [101701,101702,101703]
    jobs=[(arm,seed) for seed in seeds for arm in ARMS]
    if args.split=='main':
        p=json.loads((ROOT/'PROTOCOL.json').read_text())
        assert args.steps==p['training']['updates']
        assert p['status']=='FROZEN_BEFORE_MAIN_TRAINING'
    else:
        assert not (ROOT/'PILOT_PROTOCOL.json').exists()
        save(ROOT/'PILOT_PROTOCOL.json',dict(status='FROZEN_BEFORE_PILOT',seeds=seeds,steps=args.steps,
             arms=ARMS,choice='One common recipe. Pilot may adjust common budget or repair bugs; no per-arm tuning.',
             source={f:sha(ROOT/f) for f in ('model.py','tasks.py','experiment.py','launch.py')},
             main_evaluation='Pilot evaluates separate selection seeds at32/128, never main256/1024 confirmation.'))
    start=time.perf_counter()
    def worker(job):
        arm,seed=job;log=ROOT/args.split/f'{arm}_s{seed}.log';log.parent.mkdir(exist_ok=True)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,str(ROOT/'experiment.py'),'--arm',arm,'--seed',str(seed),
             '--steps',str(args.steps),'--split',args.split,'--limit','360']
        with log.open('w') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=420)
        return dict(arm=arm,seed=seed,returncode=result.returncode,log=str(log.relative_to(ROOT)))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,j) for j in jobs]):
            r=f.result();rows.append(r);print(json.dumps(r),flush=True)
            save(ROOT/args.split/'LAUNCH_PROGRESS.json',rows)
    save(ROOT/args.split/'LAUNCH_TERMINAL.json',dict(runs=rows,wall_seconds=time.perf_counter()-start,
                                                   status='COMPLETE' if all(r['returncode']==0 for r in rows) else 'FAILED'))
    if any(r['returncode'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
