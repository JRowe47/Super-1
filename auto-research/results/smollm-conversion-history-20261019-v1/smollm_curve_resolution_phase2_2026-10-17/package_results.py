"""Package completed, verified results without the reference model or caches."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    for name in ['summary.json','checkpoint_verification.json']:
        assert json.loads((ROOT/'results'/name).read_text())['status']=='PASS'
    assert (ROOT/'results/phase2_report.html').is_file()
    excluded={'cache','reference_model','__pycache__','recorded_results'}
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or any(part in excluded for part in rel.parts):continue
        if path.suffix in ['.parquet','.pyc'] or path.name=='SHA256SUMS':continue
        files.append(path)
    sums=ROOT/'SHA256SUMS'
    sums.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files))
    files.append(sums)
    output=ROOT.parent/'smollm_curve_resolution_phase2_2026-10-17.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:archive.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'files':len(files),
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
