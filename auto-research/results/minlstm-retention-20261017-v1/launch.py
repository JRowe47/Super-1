import argparse, concurrent.futures, json, os, subprocess, sys, time
from experiment import ROOT,save
from model import ARMS

def main():
    a=argparse.ArgumentParser();a.add_argument('--split',choices=['pilot','main'],required=True);a.add_argument('--workers',type=int,default=3)
    args=a.parse_args();p=json.loads((ROOT/'PROTOCOL.json').read_text());split=args.split
    seeds=p['design']['seeds'] if split=='main' else p['pilot']['seeds']
    steps=p['training']['updates'] if split=='main' else p['pilot']['updates']
    assert json.loads((ROOT/'VERIFICATION.json').read_text())['status']=='PASS'
    start=time.perf_counter();rows=[]
    def worker(job):
        arm,seed=job;log=ROOT/split/f'{arm}_s{seed}.log';log.parent.mkdir(exist_ok=True)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with log.open('w') as f:
            r=subprocess.run([sys.executable,str(ROOT/'experiment.py'),'--arm',arm,'--seed',str(seed),
                              '--split',split,'--steps',str(steps)],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
        return dict(arm=arm,seed=seed,returncode=r.returncode,log=str(log.relative_to(ROOT)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,(arm,seed)) for seed in seeds for arm in ARMS]):
            row=f.result();rows.append(row);print(json.dumps(row),flush=True);save(ROOT/split/'LAUNCH_PROGRESS.json',rows)
    status='COMPLETE' if all(r['returncode']==0 for r in rows) else 'FAILED'
    save(ROOT/split/'LAUNCH_TERMINAL.json',dict(status=status,runs=rows,wall_seconds=time.perf_counter()-start))
    if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
