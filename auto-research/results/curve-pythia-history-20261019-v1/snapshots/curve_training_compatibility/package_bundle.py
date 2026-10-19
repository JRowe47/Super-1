"""Create and verify a reproducible release archive, excluding temporary training files."""
import ast,hashlib,json,zipfile
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class CheckHTML(HTMLParser):
    def __init__(self):super().__init__();self.images=0;self.ids=set();self.anchors=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='a' and a.get('href','').startswith('#'):self.anchors.append(a['href'][1:])
        if tag=='img':
            assert a.get('alt') and a.get('src','').startswith('data:image/png;base64,')
            self.images+=1
def included(path):
    rel=path.relative_to(ROOT)
    if any(p in {'temporary_references','stable_references','__pycache__','qa'} for p in rel.parts):return False
    if path.suffix in {'.log','.parquet','.pt','.partial','.zip'}:return False
    if path.name in {'SHA256SUMS.json','model.safetensors'}:return False
    return path.is_file() and not any(p.startswith('.') for p in rel.parts)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    for path in ROOT.glob('*.py'):ast.parse(path.read_text())
    check=CheckHTML();check.feed((ROOT/'curve_training_compatibility_report.html').read_text())
    assert check.images==3 and set(check.anchors)<=check.ids
    assert sha(ROOT/'protocol.json')==json.loads((ROOT/'results/pythia_environment.json').read_text())['protocol_sha256']
    for item in json.loads((ROOT/'results/pythia_environment.json').read_text())['final_checkpoints']:
        assert sha(ROOT/item['file'])==item['sha256']
    files=sorted(p for p in ROOT.rglob('*') if included(p))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    mf=ROOT/'SHA256SUMS.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
    target=ROOT.parent/(ROOT.name+'.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files+[mf]:
            z.write(path,str(Path(ROOT.name)/path.relative_to(ROOT)),compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,expected in manifest.items():
            assert hashlib.sha256(z.read(str(Path(ROOT.name)/name))).hexdigest()==expected,name
    print(json.dumps(dict(archive=str(target),files=len(files)+1,bytes=target.stat().st_size,sha256=sha(target),html_figures=check.images,checksum_verification='passed'),indent=2))
if __name__=='__main__':main()
