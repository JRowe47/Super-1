"""Limits every child, including imports; receipts charge child and launcher CPU."""
import argparse, datetime, json, os, resource, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(stage):
    cfg = json.loads((ROOT/'PROTOCOL.json').read_text())
    if stage == 'verify':
        jobs, cap = [[sys.executable, str(ROOT/'verify.py')]], 30
    elif stage == 'pilot':
        jobs = [[sys.executable, str(ROOT/'experiment.py'), '--seed', str(cfg['pilot']['seed']), '--pilot']]
        cap = 110
        assert json.loads((ROOT/'VERIFY_RECEIPT.json').read_text())['status'] == 'COMPLETE'
    else:
        jobs = [[sys.executable, str(ROOT/'experiment.py'), '--seed', str(seed)] for seed in cfg['main']['seeds']]
        cap = 230
        assert json.loads((ROOT/'PILOT_RECEIPT.json').read_text())['status'] == 'COMPLETE'
        pilot = json.loads((ROOT/'pilot'/str(cfg['pilot']['seed'])/'RESULT.json').read_text())
        assert all(a['status'] == 'COMPLETE' for a in pilot['arms'].values()), 'Pilot numerical failure'
    receipt = ROOT/(stage.upper()+'_RECEIPT.json')
    assert not receipt.exists(), 'Do not overwrite an execution receipt'
    start = time.monotonic()
    date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cap, cap+1))
    children, codes, error = [], [], None
    try:
        for job in jobs:
            children.append(subprocess.Popen(job, env=env, preexec_fn=limits))
        deadline = start + (540 if stage == 'main' else 240)
        for child in children:
            codes.append(child.wait(timeout=max(1., deadline-time.monotonic())))
    except BaseException as exc:
        error = repr(exc)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
    child_r, own = resource.getrusage(resource.RUSAGE_CHILDREN), resource.getrusage(resource.RUSAGE_SELF)
    out = {'stage': stage, 'started_at_utc': date, 'status': 'COMPLETE' if codes == [0]*len(jobs) and error is None else 'FAILED',
           'child_exit_codes': [c.returncode for c in children], 'error': error,
           'wall_seconds': time.monotonic()-start, 'child_cpu_seconds': child_r.ru_utime+child_r.ru_stime,
           'launcher_cpu_seconds': own.ru_utime+own.ru_stime, 'child_count': len(children),
           'per_child_cpu_cap_seconds': cap, 'all_children_exited': all(c.poll() is not None for c in children)}
    receipt.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out), flush=True)
    if out['status'] != 'COMPLETE':
        sys.exit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['verify', 'pilot', 'main'])
    main(p.parse_args().stage)
