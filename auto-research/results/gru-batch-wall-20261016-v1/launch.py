"""Sequential workers prevent benchmark/training contention from this experiment."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);args=parser.parse_args()
p=json.loads((ROOT/'PROTOCOL.json').read_text());data=str(args.data_dir.resolve())
logs=ROOT/'logs';logs.mkdir(exist_ok=True)
def worker(name,mode,threads=1,run=None):
    cmd=[sys.executable,'-u',str(ROOT/'experiment.py'),mode,'--data-dir',data,'--threads',str(threads)]
    if run:cmd+=['--run-id',run]
    print('START',name,flush=True)
    with (logs/f'{name}.log').open('a') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        try:code=proc.wait(timeout=360 if mode!='train' else 210)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10);raise
    print('EXIT',name,code,flush=True)
    assert code==0,f'{name} failed; inspect its log'
if not (ROOT/'PREFLIGHT.json').exists():worker('preflight','preflight')
for t in p['threads']:
    if not (ROOT/'benchmark'/f'threads{t}.json').exists():worker(f'bench_t{t}','benchmark',t)
selection_path=ROOT/'SELECTION.json'
if not selection_path.exists():
    rows=[r for t in p['threads'] for r in json.loads((ROOT/'benchmark'/f'threads{t}.json').read_text())['rows']]
    runs=[];chosen={}
    for arm in p['arms']:
        cells=[r for r in rows if r['arm']==arm];best=max(cells,key=lambda r:r['tokens_per_wall_second'])
        baseline=next(r for r in cells if r['batch']==1 and r['threads']==1);chosen[arm]=best
        for role,r in [('baseline',baseline),('throughput',best)]:
            if role=='throughput' and (r['batch'],r['threads'])==(1,1):continue
            runs.append(dict(run_id=f'{arm}_{role}',arm=arm,role=role,batch=r['batch'],threads=r['threads'],benchmark_step_wall_seconds=r['median_step_wall_seconds']))
    np.random.default_rng(101624).shuffle(runs)
    selection=dict(status='FROZEN_BEFORE_QUALITY_TRAINING',selected=chosen,runs=runs,protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    selection_path.write_text(json.dumps(selection,indent=2)+'\n')
else:selection=json.loads(selection_path.read_text())
print('SELECTED',json.dumps(selection['runs']),flush=True)
for r in selection['runs']:
    if not (ROOT/'execution'/r['run_id']/'RESULT.json').exists():worker(r['run_id'],'train',r['threads'],r['run_id'])
print('ALL COMPLETE',flush=True)
