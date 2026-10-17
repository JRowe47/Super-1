"""Bounded sequential workers; archive every terminal checkpoint before next trial.

--upload-helper is optional for third-party reproduction. This execution uses
the supported Library helper with a one-file progress-save at each boundary.
"""
import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--archive-dir',type=Path,required=True)
ap.add_argument('--upload-helper',type=Path);ap.add_argument('--upload-python',default='python');args=ap.parse_args()
p=json.loads((ROOT/'PROTOCOL.json').read_text());assert (ROOT/'PREFLIGHT.json').exists()
(ROOT/'logs').mkdir(exist_ok=True);args.archive_dir.mkdir(parents=True,exist_ok=True);codes={}
for q in p['runs']:
    rid=q['run_id'];out=ROOT/'execution'/rid
    if not (out/'RESULT.json').exists() and not (out/'FAILURE.json').exists():
        print('START',rid,flush=True);env=os.environ.copy()
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='4'
        began=time.perf_counter()
        with (ROOT/'logs'/f'{rid}.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiment.py'),'train','--data-dir',str(args.data_dir.resolve()),'--run-id',rid],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=p['hard_limits']['launcher_wall_seconds'])
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code='launcher_timeout'
        codes[rid]=dict(exit_code=code,end_to_end_worker_wall_seconds=time.perf_counter()-began)
        if code!=0 and not (out/'FAILURE.json').exists():
            out.mkdir(parents=True,exist_ok=True);(out/'FAILURE.json').write_text(json.dumps(dict(run_id=rid,error_type='WorkerExit',error=str(code)),indent=2)+'\n')
        (ROOT/'LAUNCH_RESULT.json').write_text(json.dumps(codes,indent=2)+'\n');print('EXIT',rid,code,flush=True)
    cp=out/'checkpoint.pt'
    if cp.exists() and not (out/'PRESERVATION.json').exists():
        archive=args.archive_dir/f'gru_context_capacity_v2_{rid}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:z.write(cp,'checkpoint.pt')
        receipt=dict(run_id=rid,archive_name=archive.name,archive_bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),status='LOCAL_ONLY')
        if args.upload_helper:
            request=dict(uploads=[dict(local_path=str(archive.resolve()),purpose='create_library_file',library_artifact_type='other')])
            saved=subprocess.run([args.upload_python,str(args.upload_helper)],input=json.dumps(request),text=True,capture_output=True,timeout=120)
            # No retry if the outcome is unknown. Inspect the retained receipt first.
            log=args.archive_dir/f'{rid}_upload_result.json';log.write_text(saved.stdout)
            if saved.returncode!=0:
                (args.archive_dir/f'{rid}_upload_error.txt').write_text(saved.stderr)
                raise RuntimeError('Checkpoint preservation failed or uncertain; inspect upload receipt before resuming')
            response=json.loads(saved.stdout)
            receipt.update(status='UPLOAD_HELPER_COMPLETED',upload_result=response)
        (out/'PRESERVATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('CHECKPOINT_ARCHIVED',rid,receipt['status'],flush=True)
print('ALL DECLARED TRIALS ATTEMPTED',flush=True)
