"""Save the full-precision decoder-curve baseline at step1000."""
import argparse,hashlib,json,pathlib,numpy as np
from scipy import fft
from insitu import *

def restore_decoder_from_curves(model,path):
    import torch
    with np.load(path) as z, torch.no_grad():
        meta=json.loads(str(z['metadata_json']))
        for row in meta:
            params=dict(model.gpt_neox.layers[row['layer']].named_parameters())
            rec=fft.irfft(z[row['key']],n=row['n'],norm='ortho').astype(np.float32).reshape(row['shape'])
            params[row['tensor']].copy_(torch.from_numpy(rec))
    return model

def main(checkpoints):
    arrays={};meta=[];maxerr=0.
    for layer in range(6):
        for tid,name in enumerate(NAMES):
            x=tensor(checkpoints,1000,layer,name);key=f'l{layer}_t{tid:02d}'
            c=fft.rfft(x.astype(np.float64).ravel(),norm='ortho')
            rec=fft.irfft(c,n=x.size,norm='ortho').astype(np.float32).reshape(x.shape)
            maxerr=max(maxerr,float(np.max(np.abs(rec-x))))
            arrays[key]=c;meta.append({'layer':layer,'tensor':name,'key':key,'shape':list(x.shape),'n':x.size})
    arrays['metadata_json']=np.array(json.dumps(meta));arrays['step']=np.array(1000)
    path=COEFF/'pythia70m_step1000_exact_decoder_curves.npz';np.savez_compressed(path,**arrays)
    dump('exact_baseline_artifact.json',{'file':path.name,'step':1000,'precision':'complex128',
      'max_reconstructed_float32_parameter_error':maxerr,'bytes':path.stat().st_size,
      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'usage':'Load the pinned step1000 model for architecture and non-block parameters, then call restore_decoder_from_curves(model, path). All six decoder blocks are reconstructed from the saved curves.'})
    print(path.name,path.stat().st_size,maxerr,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=pathlib.Path,default=ROOT.parent/'pythia70m_spectral_2026-10-17'/'checkpoints')
    main(ap.parse_args().checkpoints)
