"""Small, exhaustive behavioral calibration of native parameter-curve components.

No network access is required. Python, NumPy, SciPy and CPU PyTorch suffice.
All stored parameters, transform coefficients and counterfactual logits are float64.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import fft
from scipy.stats import spearmanr
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
MODELS = ROOT / 'models'
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
P = json.loads((ROOT / 'protocol.json').read_text())
STEPS = P['steps']
INTERVALS = P['intervals']
X = np.array(list(itertools.product([-1., 1.], repeat=4)))
TARGETS = np.column_stack([X[:, 0], X[:, 0]*X[:, 1], np.where(X[:, 2] < 0, X[:, 0], X[:, 1])])
WALSH = np.stack([np.prod(X[:, [i for i in range(4) if (m >> i) & 1]], axis=1) for m in range(16)], axis=1)
FEATURE_NAMES = ['1' if m == 0 else ''.join('abcd'[i] for i in range(4) if (m >> i) & 1) for m in range(16)]
NAMES = ['l0.weight', 'l0.bias', 'l1.weight', 'l1.bias', 'out.weight', 'out.bias']

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def haar(x):
    a = np.asarray(x, np.float64).ravel(); ds = []; level = 0
    while len(a) % 2 == 0:
        ds.append((a[::2]-a[1::2])/np.sqrt(2))
        a = (a[::2]+a[1::2])/np.sqrt(2); level += 1
    blocks = [a]+ds[::-1]; layout = []; off = 0
    for j, b in enumerate(blocks):
        lev = level if j == 0 else level-j+1
        layout.append(dict(kind='scaling' if j == 0 else 'detail', level=lev, offset=off, count=len(b), support=2**lev))
        off += len(b)
    return np.concatenate(blocks), layout

def ihaar(c, layout):
    a = np.asarray(c[:layout[0]['count']], np.float64).copy()
    for g in layout[1:]:
        d = c[g['offset']:g['offset']+g['count']]; b = np.empty(2*len(a))
        b[::2] = (a+d)/np.sqrt(2); b[1::2] = (a-d)/np.sqrt(2); a = b
    return a

def packet(x, path):
    a = np.asarray(x, np.float64)
    for k in path:
        a = (a[::2] + (1 if k == 'a' else -1)*a[1::2])/np.sqrt(2)
    return a

def ipacket(a, path):
    a = np.asarray(a, np.float64).copy()
    for k in path[::-1]:
        b = np.empty(2*len(a)); b[::2] = a/np.sqrt(2)
        b[1::2] = (1 if k == 'a' else -1)*a/np.sqrt(2); a = b
    return a

def component_catalog(shapes):
    catalog = []
    for tid, shape in enumerate(shapes):
        n = int(np.prod(shape)); _, layout = haar(np.zeros(n))
        base = dict(tensor=NAMES[tid], tid=tid, shape=list(shape), n=n)
        for g in layout:
            for k in range(g['count']):
                catalog.append(dict(**base, basis='haar', atom=g['offset']+k, kind=g['kind'], level=g['level'], index=k,
                                    start=k*g['support'], end=(k+1)*g['support'], leaf=True))
        def add_fourier(lo, hi, depth):
            idx = len(catalog)
            catalog.append(dict(**base, basis='fourier', low=lo, high=hi, depth=depth, leaf=hi-lo == 1))
            if hi-lo > 1:
                mid = (lo+hi)//2
                a = add_fourier(lo, mid, depth+1); b = add_fourier(mid, hi, depth+1)
                catalog[idx]['children'] = [a, b]
            return idx
        add_fourier(0, n//2+1, 0)
        depth = layout[0]['level']
        def add_packet(path):
            idx = len(catalog)
            catalog.append(dict(**base, basis='packet', path=path, depth=len(path), leaf=len(path) == depth))
            if len(path) < depth:
                a = add_packet(path+'a'); b = add_packet(path+'d')
                catalog[idx]['children'] = [a,b]
            return idx
        add_packet('')
    for i, c in enumerate(catalog): c['id'] = i
    return catalog

def projections(params, catalog):
    """All projected interval updates in original tensor coordinates."""
    ans = []; caches = {}
    for c in catalog:
        tid = c['tid']; x = params[tid].ravel(); n = len(x)
        if tid not in caches:
            hc, layout = haar(x); caches[tid] = (hc, layout, fft.rfft(x, norm='ortho'))
        hc, layout, fc = caches[tid]
        if c['basis'] == 'haar':
            v = np.zeros(n); v[c['atom']] = hc[c['atom']]; y = ihaar(v, layout)
        elif c['basis'] == 'fourier':
            z = np.zeros_like(fc); z[c['low']:c['high']] = fc[c['low']:c['high']]
            y = fft.irfft(z, n=n, norm='ortho')
        else:
            y = ipacket(packet(x, c['path']), c['path'])
        ans.append(y.reshape(params[tid].shape))
    return ans

def infer(params, x=X):
    """Supports model batch dimension, including no batch."""
    z = np.asarray(x)
    for i in (0,2,4):
        z = np.matmul(z, np.swapaxes(params[i], -1,-2)) + params[i+1][...,None,:]
        if i < 4: z = np.tanh(z)
    return z

def interventions(base, changes, catalog, sign):
    m = len(catalog)
    ps = [np.broadcast_to(p, (m,)+p.shape).copy() for p in base]
    for i,c in enumerate(catalog): ps[c['tid']][i] += sign*changes[i]
    return infer(ps)

class Toy(nn.Module):
    def __init__(self, outputs):
        super().__init__(); self.l0=nn.Linear(4,8); self.l1=nn.Linear(8,8); self.out=nn.Linear(8,outputs)
    def forward(self,x): return self.out(torch.tanh(self.l1(torch.tanh(self.l0(x)))))

def safe_rho(a,b):
    if np.ptp(a) < 1e-14 or np.ptp(b) < 1e-14: return None
    return float(spearmanr(a,b).statistic)

def rms(x, axis=None): return np.sqrt(np.mean(np.square(x), axis=axis))

def analyze(task, seed, params, logits, y, catalog):
    train = []
    for si,s in enumerate(STEPS):
        l = logits[si]; q = np.tanh(l/2)
        train.append(dict(task=task,seed=seed,step=s,accuracy=np.mean((l>0)==(y>0),axis=0).tolist(),
            bce=float(np.mean(np.logaddexp(0,-y*l))),target_margin=np.mean(q*y,axis=0).tolist(),
            function_coefficients=(WALSH.T@q/len(X)).T.tolist(),
            irrelevant_d_energy=float(np.sum((WALSH.T@q/len(X))[8:]**2))))
    rollback=[]; transplant=[]; norms=[]; summary=[]; interactions=[]; recon=[]
    for pi,(s0,s1) in enumerate(INTERVALS):
        i0=STEPS.index(s0); i1=STEPS.index(s1)
        old=[p[i0] for p in params]; new=[p[i1] for p in params]
        delta=[b-a for a,b in zip(old,new)]
        changes=projections(delta,catalog)
        norms.append([float(np.linalg.norm(v)) for v in changes])
        rb=interventions(new,changes,catalog,-1); fw=interventions(old,changes,catalog,+1)
        rollback.append(rb); transplant.append(fw)
        qb=np.tanh(logits[i1]/2); qa=np.tanh(logits[i0]/2)
        qr=np.tanh(rb/2); qf=np.tanh(fw/2)
        eff=qb[None]-qr; efl=logits[i1][None]-rb
        feff=qf-qa[None]
        drops=np.mean(qb*y,axis=0)[None]-np.mean(qr*y,axis=1)
        for basis in ['haar','fourier','packet']:
            ids=[i for i,c in enumerate(catalog) if c['basis']==basis and c['leaf']]
            n=np.array(norms[-1])[ids]; e=rms(eff[ids],axis=(1,2))
            summary.append(dict(task=task,seed=seed,start=s0,end=s1,basis=basis,leaf_count=len(ids),
              change_effect_spearman=safe_rho(n,e),median_effect=float(np.median(e)),max_effect=float(np.max(e)),
              changed_decisions=int(np.sum((rb[ids]>0)!=(logits[i1]>0)[None])),
              rollback_transplant_effect_correlation=safe_rho(e,rms(feff[ids],axis=(1,2)))))
        for i,c in enumerate(catalog):
            if 'children' not in c: continue
            a,b=c['children']; inter=eff[i]-eff[a]-eff[b]; il=efl[i]-efl[a]-efl[b]
            den=float(rms(efl[a])+rms(efl[b])); pden=float(rms(eff[a])+rms(eff[b]))
            interactions.append(dict(task=task,seed=seed,start=s0,end=s1,component=i,basis=c['basis'],tensor=c['tensor'],
                logit_interaction_rms=float(rms(il)),probability_interaction_rms=float(rms(inter)),
                logit_relative=float(rms(il)/(den+1e-30)),probability_relative=float(rms(inter)/(pden+1e-30)),
                child_logit_effect_sum=den,parent_logit_effect=float(rms(efl[i])),
                parameter_additivity_max_abs=float(np.max(np.abs(changes[i]-changes[a]-changes[b])))))
        # Full update plus before must recover after; and removing it must recover before.
        recon.append(dict(start=s0,end=s1,full_update_logits_max_abs=float(np.max(np.abs(infer([a+d for a,d in zip(old,delta)])-logits[i1])))))
    arrays={'inputs':X,'targets':y,'steps':np.array(STEPS),'intervals':np.array(INTERVALS),
            'original_logits':np.array(logits),'rollback_logits':np.array(rollback),'transplant_logits':np.array(transplant),
            'component_delta_norm':np.array(norms),'walsh_basis':WALSH,'catalog_json':np.array(json.dumps(catalog))}
    for tid,p in enumerate(params):
        arrays[f'p{tid}']=p
        arrays[f'fourier{tid}']=fft.rfft(p.reshape(len(STEPS),-1),axis=1,norm='ortho')
        arrays[f'haar{tid}']=np.stack([haar(v)[0] for v in p])
        # Full precision reconstruction checks, at every checkpoint, every tensor.
        for si in range(len(STEPS)):
            hc,layout=haar(p[si]); h=ihaar(hc,layout).reshape(p[si].shape)
            f=fft.irfft(arrays[f'fourier{tid}'][si],n=p[si].size,norm='ortho').reshape(p[si].shape)
            recon.append(dict(tensor=NAMES[tid],step=STEPS[si],haar_max_abs=float(np.max(abs(h-p[si]))),fourier_max_abs=float(np.max(abs(f-p[si])))))
    np.savez_compressed(MODELS/f'{task}_seed{seed}.npz',**arrays)
    return train,summary,interactions,recon

def main():
    start=time.time(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    tx=torch.tensor(X); logs=[]; summaries=[]; interactions=[]; checks=[]; counts=[]
    dump('input_semantics.json',dict(inputs=X.tolist(),targets=TARGETS.tolist(),output_names=['readout','parity','context'],
        feature_names=FEATURE_NAMES,ground_truth_walsh=(WALSH.T@TARGETS/16).T.tolist()))
    for task in P['tasks']:
        y=TARGETS if task=='joint' else TARGETS[:,[['readout','parity','context'].index(task)]]
        ty=torch.tensor((y+1)/2)
        for seed in P['seeds']:
            runstart=time.time(); torch.manual_seed(seed); model=Toy(y.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=P['optimizer']['lr'])
            states=[]; logits=[]
            for step in range(max(STEPS)+1):
                if step in STEPS:
                    states.append([p.detach().numpy().copy() for p in model.parameters()])
                    logits.append(model(tx).detach().numpy().copy())
                if step == max(STEPS): break
                opt.zero_grad(set_to_none=True); loss=nn.functional.binary_cross_entropy_with_logits(model(tx),ty)
                loss.backward();opt.step()
            params=[np.stack([s[i] for s in states]) for i in range(6)]
            shapes=[p[0].shape for p in params]; catalog=component_catalog(shapes)
            npcheck=float(np.max(np.abs(infer([p[-1] for p in params])-logits[-1])))
            assert npcheck<1e-12, npcheck
            a,b,c,d=analyze(task,seed,params,logits,y,catalog)
            logs.extend(a);summaries.extend(b);interactions.extend(c);checks.append(dict(task=task,seed=seed,numpy_torch_logit_error=npcheck,checks=d))
            acc=a[-1]['accuracy'];counts.append(dict(task=task,seed=seed,parameters=sum(p[0].size for p in params),components=len(catalog),final_accuracy=acc,
                seconds=time.time()-runstart))
            print(json.dumps(counts[-1]),flush=True)
    dump('training.json',logs); dump('leaf_summaries.json',summaries);dump('sibling_interactions.json',interactions);dump('numerical_checks.json',checks)
    dump('runs.json',counts)
    dump('environment.json',dict(seconds=time.time()-start,numpy=np.__version__,torch=torch.__version__,cpu_threads=1,runs=len(counts),training_updates=20*max(STEPS),
        component_intervention_cases=sum(c['components']*len(INTERVALS)*2 for c in counts),
        counterfactual_input_evaluations=sum(c['components']*len(INTERVALS)*2*len(X) for c in counts),seed_correspondence='Only within each model; no across-seed alignment or pooling of addresses'))
    print('CALIBRATION COMPLETE',flush=True)

if __name__=='__main__': main()
