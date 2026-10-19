"""Download pinned external inputs. Existing files are verified, never silently replaced."""
import argparse, hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fetch(url,path,expected):
    if path.exists():
        if digest(path)!=expected:raise RuntimeError('Existing file hash mismatch: '+str(path))
        print('Verified',path.relative_to(ROOT));return
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    req=urllib.request.Request(url,headers={'User-Agent':'curve-training-compatibility/1.0'})
    with urllib.request.urlopen(req,timeout=90) as src,tmp.open('wb') as dst:
        while chunk:=src.read(4*1024*1024):dst.write(chunk)
    if digest(tmp)!=expected:raise RuntimeError('Downloaded hash mismatch: '+str(tmp))
    tmp.replace(path);print('Downloaded and verified',path.relative_to(ROOT))
def main(dataset):
    m=json.loads((ROOT/'checkpoint_manifest.json').read_text())
    url='https://huggingface.co/'+m['repo']+'/resolve/'+m['commit']+'/model.safetensors'
    fetch(url,ROOT/'checkpoints/step143000/model.safetensors',m['sha256'])
    for item in json.loads((ROOT/'data/bundled_model_metadata.json').read_text()):
        assert digest(ROOT/item['path'])==item['sha256'],item['path']
    if dataset:
        for item in json.loads((ROOT/'data/source.json').read_text())['files']:
            fetch(item['url'],ROOT/'data'/(item['split']+'.parquet'),item['sha256'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',action='store_true');main(p.parse_args().dataset)
