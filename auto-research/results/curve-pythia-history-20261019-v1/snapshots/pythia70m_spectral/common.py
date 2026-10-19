import json, pathlib, numpy as np
from scipy import fft
from safetensors.numpy import load_file

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
STEPS = json.loads((ROOT/'protocol.json').read_text())['steps']
SEED = 20261017
FAMILIES = ['q','k','v','o','mlp_in','mlp_out']

def dump(name, x):
    (OUT/name).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def state(step):
    return load_file(str(ROOT/'checkpoints'/f'step{step}'/'model.safetensors'))

def tensor_map(s):
    result={}
    for layer in range(6):
        p=f'gpt_neox.layers.{layer}.'
        qkv=s[p+'attention.query_key_value.weight'].reshape(8,3,64,512)
        for i,fam in enumerate(['q','k','v']):
            result[(layer,fam)]=qkv[:,i].reshape(512,512).copy()
        result[(layer,'o')]=s[p+'attention.dense.weight']
        result[(layer,'mlp_in')]=s[p+'mlp.dense_h_to_4h.weight']
        result[(layer,'mlp_out')]=s[p+'mlp.dense_4h_to_h.weight']
    return result

def real_fft(x, axis=-1):
    """N real orthonormal cos/sin coordinates for N real values; no redundancy."""
    a=np.moveaxis(np.asarray(x,dtype=np.float32),axis,-1)
    n=a.shape[-1]
    z=fft.rfft(a,axis=-1,norm='ortho',workers=1)
    c=np.empty_like(a)
    c[...,0]=z[...,0].real
    m=(n-1)//2
    c[...,1:2*m+1:2]=np.sqrt(2)*z[...,1:m+1].real
    c[...,2:2*m+1:2]=np.sqrt(2)*z[...,1:m+1].imag
    if n%2==0: c[...,-1]=z[...,-1].real
    return np.moveaxis(c,-1,axis)

def real_ifft(c,axis=-1):
    a=np.moveaxis(np.asarray(c,dtype=np.float32),axis,-1)
    n=a.shape[-1]; m=(n-1)//2
    z=np.empty(a.shape[:-1]+(n//2+1,),np.complex64)
    z[...,0]=a[...,0]
    z[...,1:m+1]=(a[...,1:2*m+1:2]+1j*a[...,2:2*m+1:2])/np.sqrt(2)
    if n%2==0: z[...,-1]=a[...,-1]
    return np.moveaxis(fft.irfft(z,n=n,axis=-1,norm='ortho',workers=1),-1,axis)

def energy(x):
    return float(np.sum(np.asarray(x,dtype=np.float64)**2))

def concentration(c):
    e=np.asarray(c,dtype=np.float64).ravel()**2
    total=float(e.sum()); n=e.size
    if total<=1e-30:
        return {'energy':total,'k95_fraction':0.,'top01':0.,'top05':0.,'top10':0.,'top25':0.,'entropy_fraction':0.}
    sorted_e=np.sort(e)[::-1]
    cs=np.cumsum(sorted_e)
    p=e/total
    out={'energy':total,'k95_fraction':float((np.searchsorted(cs,0.95*total)+1)/n),
         'entropy_fraction':float(-np.sum(p*np.log(p+1e-300))/np.log(n))}
    for frac,label in [(0.01,'top01'),(0.05,'top05'),(0.10,'top10'),(0.25,'top25')]:
        out[label]=float(cs[max(1,int(n*frac))-1]/total)
    return out

def band_tree(x,depth=6):
    flat=np.asarray(x,dtype=np.float32).ravel()
    z=fft.rfft(flat,norm='ortho',workers=1)
    e=np.abs(z.astype(np.complex128))**2
    e[1:-1]*=2
    if flat.size%2: e[-1]*=2
    total=float(e.sum())
    levels={}
    for d in range(1,depth+1):
        edges=np.linspace(0,len(e),2**d+1,dtype=int)
        levels[str(d)]=[float(e[a:b].sum()/total) if total else 0. for a,b in zip(edges[:-1],edges[1:])]
    return levels

def ordered(x,family,p):
    return x[p,:] if family=='mlp_in' else x[:,p]

def unordered(x,family,p):
    return ordered(x,family,np.argsort(p))

def axis_for(family):
    return 0 if family=='mlp_in' else 1

def fit_orderings():
    """Deterministic PCA1 sorting of early joint in/out neuron-update signatures."""
    a=tensor_map(state(2000)); b=tensor_map(state(4000))
    orders={}; details=[]
    for l in range(6):
        x=b[(l,'mlp_in')]-a[(l,'mlp_in')]
        y=(b[(l,'mlp_out')]-a[(l,'mlp_out')]).T
        x=x/(np.sqrt(np.mean(x*x))+1e-20)
        y=y/(np.sqrt(np.mean(y*y))+1e-20)
        sig=np.concatenate([x,y],axis=1)
        sig=sig-sig.mean(axis=0,keepdims=True)
        rng=np.random.default_rng(SEED+l)
        v=rng.normal(size=sig.shape[1]).astype(np.float32)
        for _ in range(40):
            v=sig.T@(sig@v); v/=np.linalg.norm(v)+1e-20
        scores=sig@v
        if scores[np.argmax(np.abs(scores))]<0: scores=-scores
        orders[(l,'pca_frozen')]=np.argsort(scores,kind='stable')
        for seed in range(3):
            orders[(l,f'random{seed}')]=np.random.default_rng(SEED+100*l+seed).permutation(2048)
        orders[(l,'native')]=np.arange(2048)
        details.append({'layer':l,'fit_from':2000,'fit_to':4000,
          'pca1_variance_fraction':energy(scores)/energy(sig),
          'eigen_residual':float(np.linalg.norm(sig.T@(sig@v)-energy(scores)*v)/(energy(scores)+1e-20))})
    np.savez(OUT/'neuron_orderings.npz',**{f'{l}_{n}':p for (l,n),p in orders.items()})
    dump('ordering_fit.json',details)
    return orders

def get_orderings():
    f=OUT/'neuron_orderings.npz'
    if not f.exists(): return fit_orderings()
    with np.load(f) as a:
        return {(int(k.split('_',1)[0]),k.split('_',1)[1]):a[k] for k in a.files}

def mask_coefficients(c,fraction,mode,rng=None,axis=None):
    k=max(1,int(c.size*fraction)); mask=np.zeros(c.shape,bool)
    if mode=='top':
        ix=np.argpartition(np.abs(c).ravel(),-k)[-k:]; mask.ravel()[ix]=True
    elif mode=='random':
        mask.ravel()[rng.choice(c.size,k,replace=False)]=True
    elif mode=='low':
        # Frequency-first traversal gives exactly the requested real-coordinate count.
        shape=np.moveaxis(c,axis,0).shape
        m=np.zeros(shape,bool); m.ravel()[:k]=True
        mask=np.moveaxis(m,0,axis)
    else: raise ValueError(mode)
    return np.where(mask,c,0),int(mask.sum())
