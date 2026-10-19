"""Archive completed, verified research without regenerable teacher caches."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent


def main():
    for name in ['summary.json','export_verification.json']:
        assert json.loads((ROOT/'results'/name).read_text())['status']=='PASS'
    assert (ROOT/'results/phase3_report.html').exists()
    excluded={'cache','reference_model','__pycache__','recorded_results'};files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or any(p in excluded for p in rel.parts):continue
        if path.name=='SHA256SUMS' or path.suffix in ['.parquet','.pyc']:continue
        # Pilot weights are not main results; their inputs and metadata suffice
        # to regenerate the pilot initialization from the pinned teacher.
        if rel.parts[0]=='pilot' and path.suffix=='.safetensors':continue
        files.append(path)
    sums=ROOT/'SHA256SUMS'
    sums.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files))
    files.append(sums);output=ROOT.parent/'smollm_depth_curve_phase3_2026-10-17.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:archive.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'files':len(files),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
