"""Export any depth-six component at every native position, without inference."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def inverse(c,path):
    a=c.astype(np.float64).copy()
    for p in path[::-1]:
        b=np.empty(a.size*2);b[::2]=a/np.sqrt(2);b[1::2]=(1 if p=='a' else -1)*a/np.sqrt(2);a=b
    return a
def main():
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--direction',choices=['early','late'],default='early');p.add_argument('--matched',action='store_true');p.add_argument('--output',type=Path,default=Path('component.npz'));a=p.parse_args()
    if len(a.path)!=6 or set(a.path)-set('ad'):p.error('path must contain exactly six a/d decisions')
    z=np.load(ROOT/'native_local_curves.npz');paths=list(z['paths']);i=paths.index(a.path);c=z[a.direction+'_coefficients'][i]
    natural=inverse(c,a.path);meta=json.loads((ROOT/'results/curve_metadata.json').read_text());scale=meta['common_norm']/np.linalg.norm(natural) if a.matched else 1.
    np.savez_compressed(a.output,native_flat_indices=z['native_flat_indices'],natural_coefficients=c,
        applied_coefficients=c*scale,natural_component=natural,applied_component=natural*scale,
        step512=z['step512'],step1000=z['step1000'],step8000=z['step8000'])
    print(json.dumps(dict(path=a.path,direction=a.direction,matched=a.matched,scale=scale,norm=float(np.linalg.norm(natural*scale)),coefficients=1024,native_positions=65536,output=str(a.output.resolve()))))
if __name__=='__main__':main()
