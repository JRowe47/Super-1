"""Run the locked final evaluation and reports after both arm workers finish."""
import argparse,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def complete():
    for kind in ['interpolated','retained']:
        path=ROOT/'results'/f'training_{kind}.json'
        if not path.exists() or json.loads(path.read_text()).get('status')!='complete':return False
    return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    deadline=time.monotonic()+3600
    while not complete():
        if not args.wait:raise RuntimeError('Finish both matched training runs first')
        if time.monotonic()>deadline:raise TimeoutError('Training completion was not observed within one hour')
        time.sleep(5)
    for name,log in [('evaluate_phase6.py',ROOT/'evaluation.log'),
                     ('verify_export.py',ROOT/'export_verification.log'),
                     ('build_report.py',ROOT.parent/'phase6_build.log')]:
        print('START',name,flush=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/name)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',name,flush=True)

if __name__=='__main__':main()
