"""Build and check the reproducible archive, excluding downloadable large sources."""
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
        if tag=='img':assert a.get('alt') and a.get('src','').startswith('data:image/png;base64,');self.images+=1
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def include(p):
    rel=p.relative_to(ROOT)
    return p.is_file() and not any(x in ['__pycache__','qa'] or x.startswith('.') for x in rel.parts) and p.suffix not in ['.parquet','.pt','.partial','.tmp','.zip'] and p.name not in ['SHA256SUMS.json','model.safetensors']
def main():
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    h=CheckHTML();h.feed((ROOT/'curve_retention_depth_report.html').read_text());assert h.images==3 and set(h.anchors)<=h.ids
    assert sha(ROOT/'protocol.json')==json.loads((ROOT/'results/environment.json').read_text())['protocol_sha256']
    rows=json.loads((ROOT/'results/transport_trajectories.json').read_text());assert len(rows)==120
    assert len(list((ROOT/'models').glob('transport_*.npz')))==40
    for arm in ['native_lambda4','curve_lambda4']:assert (ROOT/'models'/f'{arm}_final_decoder.npz').is_file()
    files=sorted(p for p in ROOT.rglob('*') if include(p));mf=ROOT/'SHA256SUMS.json';manifest={str(p.relative_to(ROOT)):sha(p) for p in files};mf.write_text(json.dumps(manifest,indent=2)+'\n')
    target=ROOT.parent/(ROOT.name+'.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[mf]:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,expected in manifest.items():assert hashlib.sha256(z.read(str(Path(ROOT.name)/name))).hexdigest()==expected,name
    print(json.dumps(dict(archive=str(target),files=len(files)+1,bytes=target.stat().st_size,sha256=sha(target),status='All release checks passed'),indent=2))
if __name__=='__main__':main()
