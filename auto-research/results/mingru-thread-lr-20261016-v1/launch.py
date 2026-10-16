"""Run the frozen seven-trial screen sequentially, preserving any failure."""
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, required=True)
args = parser.parse_args()
p = json.loads((ROOT/'PROTOCOL.json').read_text())
(ROOT/'logs').mkdir(exist_ok=True)
codes = {}
for q in p['runs']:
    rid = q['run_id']; out = ROOT/'execution'/rid
    if (out/'RESULT.json').exists():
        codes[rid] = 'completed_previously'; continue
    if (out/'FAILURE.json').exists():
        codes[rid] = 'failed_previously'; continue
    print('START', rid, flush=True)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[name] = str(q['threads'])
    with (ROOT/'logs'/f'{rid}.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(ROOT/'experiment.py'), '--data-dir', str(args.data_dir.resolve()), '--run-id', rid], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=210)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            code = 'launcher_timeout'
        codes[rid] = code
    print('EXIT', rid, code, flush=True)
(ROOT/'LAUNCH_RESULT.json').write_text(json.dumps(codes, indent=2)+'\n')
print('ALL DECLARED TRIALS ATTEMPTED', flush=True)
