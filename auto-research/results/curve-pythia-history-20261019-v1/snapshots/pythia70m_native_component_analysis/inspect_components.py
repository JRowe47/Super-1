"""Extract any archived component or frequency subtree at full sample resolution.

Examples:
  python inspect_components.py --layer 3 --tensor attention.query_key_value.weight --start 512 --end 1000 --basis haar --level 9 --index 1152 --output component.npz
  python inspect_components.py --layer 3 --tensor attention.query_key_value.weight --start 512 --end 1000 --basis fourier --low 100 --high 101 --output mode.npz

Fourier [low, high) may represent one mode or any node of a recursive frequency partition.
Every output includes before, after, and signed difference curves on the original grid.
"""
import argparse,json,pathlib,numpy as np
from scipy import fft
from insitu import ROOT,COEFF,STEPS,ihaar

def extract(layer,name,start,end,basis,low=0,high=None,level=None,index=None):
    p=COEFF/f'pythia70m_native_layer{layer}_coefficients.npz'
    with np.load(p) as z:
        meta=next(m for m in json.loads(str(z['metadata_json'])) if m['tensor']==name)
        key=meta['fourier_array'] if basis=='fourier' else meta['haar_array']
        coeff=z[key][[STEPS.index(start),STEPS.index(end)]].copy()
    results=[]
    if basis=='fourier':
        if high is None:high=coeff.shape[1]
        if not 0<=low<high<=coeff.shape[1]:raise ValueError('Invalid half-open frequency range')
        coeff[:,:low]=0;coeff[:,high:]=0
        results=[fft.irfft(x.astype(np.complex128),n=meta['n'],norm='ortho') for x in coeff]
        descriptor={'basis':basis,'frequency_low':low,'frequency_high_exclusive':high}
    else:
        g=next(g for g in meta['haar_layout'] if g['kind']=='detail' and g['level']==level)
        if index is None or not 0<=index<g['count']:raise ValueError('Index outside this detail scale')
        packed=g['offset']+index
        for c in coeff:
            only=np.zeros_like(c);only[packed]=c[packed];results.append(ihaar(only,meta['haar_layout']))
        descriptor={'basis':basis,'level':level,'index':index,'support_start':index*2**level,'support_end_exclusive':(index+1)*2**level}
    return {'before':results[0],'after':results[1],'delta':results[1]-results[0],
       'metadata_json':np.array(json.dumps({'layer':layer,'tensor':name,'shape':meta['shape'],'start':start,'end':end,**descriptor}))}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True);ap.add_argument('--tensor',required=True)
    ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True)
    ap.add_argument('--basis',choices=['fourier','haar'],required=True);ap.add_argument('--low',type=int,default=0)
    ap.add_argument('--high',type=int);ap.add_argument('--level',type=int);ap.add_argument('--index',type=int)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    args=vars(ap.parse_args());out=args.pop('output');args['name']=args.pop('tensor')
    np.savez_compressed(out,**extract(**args));print(out.resolve())
