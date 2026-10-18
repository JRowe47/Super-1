"""Bounded CPU launcher; source/protocol must be frozen before invocation."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

root=Path(__file__).resolve().parent
p=json.loads((root/'PROTOCOL.json').read_text())
assert hashlib.sha256((root/'experiment.py').read_bytes()).hexdigest()==p['source_sha256']
assert json.loads((root/'VERIFICATION.json').read_text())['source_sha256']==p['source_sha256']
logdir=root.parents[1]/'logs'/root.name;logdir.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
started=time.time();workers=[];rows=[]
for seed in p['seeds']:
    log=open(logdir/f's{seed}.log','x')
    code="import resource,runpy,sys;resource.setrlimit(resource.RLIMIT_CPU,(120,125));sys.argv=['experiment.py','main','--seed',sys.argv[1]];runpy.run_path('experiment.py',run_name='__main__')"
    process=subprocess.Popen([sys.executable,'-c',code,str(seed)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    workers.append((seed,process,log));print(json.dumps({'seed':seed,'pid':process.pid,'status':'RUNNING'}),flush=True)
for seed,process,log in workers:
    try:code=process.wait(timeout=max(1,600-(time.time()-started)))
    except subprocess.TimeoutExpired:process.terminate();code=process.wait(timeout=10)
    log.close();row={'seed':seed,'returncode':code};rows.append(row);print(json.dumps(row),flush=True)
result={'status':'TRAINING_COMPLETE' if all(r['returncode']==0 for r in rows) else 'FAILED',
        'rows':rows,'wall_seconds':time.time()-started,'started_unix':started}
(root/'LAUNCH_TERMINAL.json').write_text(json.dumps(result,indent=2)+'\n')
if result['status']!='TRAINING_COMPLETE':sys.exit(1)
checkpoints={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((root/'main').rglob('*.pt'))}
(root/'CHECKPOINT_FREEZE.json').write_text(json.dumps({'checkpoints':checkpoints,'frozen_before_confirmation':True,'time_unix':time.time()},indent=2)+'\n')
