"""Read-only depth Fourier/DCT diagnostics for the recovered phase-6 models.

No model training or language-model evaluation. All transforms run in float64.
Usage: python3 run_probe.py --phase6 PATH
"""
import argparse, hashlib, json, math, os, struct, time
from pathlib import Path
import numpy as np
import scipy
from scipy.fft import rfft, irfft, dct, idct
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

class SafeArrays:
    """Minimal read-only F32/BF16 safetensors reader; never executes pickle."""
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.offset = 8 + n
    def get(self, name):
        h = self.header[name]
        dtype = {'F32': '<f4', 'BF16': '<u2'}[h['dtype']]
        a, b = h['data_offsets']
        assert b-a == math.prod(h['shape'])*np.dtype(dtype).itemsize
        assert self.offset+b <= self.path.stat().st_size
        x = np.memmap(self.path, dtype=dtype, mode='r', offset=self.offset+a, shape=tuple(h['shape']))
        if h['dtype'] == 'BF16':
            return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)
        return np.asarray(x)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def bands(n):
    j = np.arange(n)
    c = dct(np.eye(n), type=2, norm='ortho', axis=0)
    f = np.exp(-2j*np.pi*np.outer(np.arange(n//2+1), j)/n)/np.sqrt(n)
    multiplicity = np.full(n//2+1, 2.0)
    multiplicity[0] = 1
    if n % 2 == 0: multiplicity[-1] = 1
    return c, f, multiplicity

def spectral(gram):
    n = len(gram)
    c, f, multiplicity = bands(n)
    d = np.maximum(np.diag(c @ gram @ c.T), 0)
    e = np.maximum(np.diag(f @ gram @ f.conj().T).real, 0)*multiplicity
    total = float(np.trace(gram))
    ev = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0)
    return dict(total_energy=total, dct_energy=d.tolist(), fft_energy=e.tolist(),
        dct_fraction=(d/total).tolist(), fft_fraction=(e/total).tolist(),
        singular_values=np.sqrt(ev).tolist(), numerical_rank=int((ev > ev[0]*1e-12).sum()),
        rank_tolerance_relative_squared=1e-12,
        dct_low8_fraction=float(d[:8].sum()/total),
        dct_low8_nonconstant_fraction=float(d[1:8].sum()/d[1:].sum()),
        dct_nonconstant_low8_relative_rmse=float(np.sqrt(d[8:].sum()/d[1:].sum())),
        fft_modes_0_to_3_fraction=float(e[:4].sum()/total),
        dct_90pct_prefix=int(np.searchsorted(np.cumsum(d), .9*total)+1),
        dct_99pct_prefix=int(np.searchsorted(np.cumsum(d), .99*total)+1),
        max_parseval_relative_error=float(max(abs(d.sum()-total),abs(e.sum()-total))/total))

def permutations(gram, seed=71701, count=1000):
    rng=np.random.default_rng(seed);n=len(gram);c,_,_=bands(n)
    total=np.trace(gram);values=[]
    for _ in range(count):
        p=rng.permutation(n);g=gram[np.ix_(p,p)]
        e=np.diag(c @ g @ c.T)
        values.append(float(e[1:8].sum()/e[1:].sum()))
    observed=spectral(gram)['dct_low8_nonconstant_fraction']
    return dict(count=count,seed=seed,statistic='DCT modes 1 through 7 / all nonconstant energy',
        observed=observed,mean=float(np.mean(values)),
        percentile_2_5_97_5=np.quantile(values,[.025,.975]).tolist(),
        descriptive_upper_tail_fraction=float((np.sum(np.array(values)>=observed)+1)/(count+1)),
        caution='Exploratory layer-order null, not a language-quality test or confirmatory p-value.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase6',type=Path,required=True);args=ap.parse_args()
    src=args.phase6;start=time.perf_counter()
    manifest=json.loads((src/'model_manifest.json').read_text())
    original_path=src/'reference_model/model.safetensors'
    assert sha(original_path)==manifest['files']['model.safetensors']['sha256']
    orig=SafeArrays(original_path)
    checkpoint={a:SafeArrays(src/'checkpoints'/f'{a}.safetensors') for a in ['retained','interpolated']}
    slopes={s:SafeArrays(src/'reference/phase4_checkpoints'/f'window_{s}.safetensors') for s in [4,12]}
    families=sorted(n.split('.',2)[2] for n in checkpoint['retained'].header if n.startswith('4.0.'))
    assert len(families)==9
    keys=['original30','retained_initial','retained_final','interpolated_initial','interpolated_final','retained_delta','interpolated_delta']
    pooled={k:np.zeros((30 if k=='original30' else 28,)*2) for k in keys}
    norm_pooled={k:np.zeros_like(v) for k,v in pooled.items()}
    all_families={};checks={};roundtrip_seconds=0.;count_per_block=0
    positions=[];i=0
    while i<30:
        if i in [4,12]:
            positions.extend([(i,j) for j in range(3)]);i+=4
        else:positions.append(i);i+=1
    original_map=[p if isinstance(p,int) else p[0]+[0,2,3][p[1]] for p in positions]
    for family in families:
        before=time.perf_counter()
        base=np.stack([orig.get(f'model.layers.{i}.{family}') for i in range(30)])
        count_per_block+=math.prod(base.shape[1:])
        states={'original30':base.reshape(30,-1)}
        for arm in ['retained','interpolated']:
            initial=[];final=[]
            for p in positions:
                if isinstance(p,int):initial.append(base[p]);final.append(base[p]);continue
                s,j=p
                if arm=='retained':v=base[s+[0,2,3][j]]
                else:
                    t=(j+.5)*4/3-.5;lo=math.floor(t);u=t-lo
                    h=(2*u**3-3*u**2+1,u**3-2*u**2+u,-2*u**3+3*u**2,u**3-u**2)
                    m=slopes[s].get(family.replace('.','__'))
                    v=h[0]*base[s+lo]+h[1]*m[lo]+h[2]*base[s+lo+1]+h[3]*m[lo+1]
                initial.append(v);final.append(checkpoint[arm].get(f'{s}.{j}.{family}'))
            states[f'{arm}_initial']=np.stack(initial).reshape(28,-1)
            states[f'{arm}_final']=np.stack(final).reshape(28,-1)
            states[f'{arm}_delta']=states[f'{arm}_final'].astype(np.float64)-states[f'{arm}_initial'].astype(np.float64)
        records={}
        for name,x in states.items():
            x64=np.asarray(x,dtype=np.float64);gram=x64@x64.T
            records[name]=spectral(gram)
            if 'layernorm' in family:norm_pooled[name]+=gram
            else:pooled[name]+=gram
        x=np.asarray(states['retained_final'],dtype=np.float64)
        tick=time.perf_counter();f=rfft(x,axis=0,norm='ortho',workers=4)
        back=irfft(f,n=28,axis=0,norm='ortho',workers=4)
        fft_err=float(np.max(np.abs(back-x)));del f,back
        c=dct(x,type=2,axis=0,norm='ortho',workers=4)
        back=idct(c,type=2,axis=0,norm='ortho',workers=4)
        dct_err=float(np.max(np.abs(back-x)));del c,back
        roundtrip_seconds+=time.perf_counter()-tick
        checks[family]={'fft_float64_max_abs_error':fft_err,'dct_float64_max_abs_error':dct_err}
        for arm in ['retained','interpolated']:
            delta=states[f'{arm}_delta'];a=states[f'{arm}_initial']
            active=[j for j,p in enumerate(positions) if not isinstance(p,int)]
            records[f'{arm}_delta']['update_over_active_initial_rms']=float(np.linalg.norm(delta)/np.linalg.norm(a[active].astype(np.float64)))
            records[f'{arm}_delta']['active_rows']=active
            assert np.count_nonzero(delta[[j for j in range(28) if j not in active]])==0
        all_families[family]=records
        print(json.dumps({'family':family,'seconds':round(time.perf_counter()-before,3),'checks':checks[family]}),flush=True)
        del states,base,x,x64,initial,final,delta,a
    aggregates={k:spectral(g) for k,g in pooled.items()}
    norms={k:spectral(g) for k,g in norm_pooled.items()}
    nulls={k:permutations(pooled[k]) for k in ['retained_final','interpolated_final','retained_delta','interpolated_delta']}
    np.savez_compressed(OUT/'depth_grams.npz',**{k:v for k,v in pooled.items()},**{'norm_'+k:v for k,v in norm_pooled.items()})
    diag=dict(model=manifest['repository'],revision=manifest['revision'],
        analysis='Depth-only transforms over corresponding tensor entries; embeddings/head excluded.',
        implementation_versions={'numpy':np.__version__,'scipy':scipy.__version__},
        parameters_per_block=count_per_block,body28_parameters=28*count_per_block,
        minimum_real_dft_coefficients_per_weight_trajectory=28,nonnegative_fourier_bands=15,
        projection_tensor_count=7,norm_tensor_count=2,original_block_map_for_retained=original_map,
        phase6_changed_rows=[j for j,p in enumerate(positions) if not isinstance(p,int)],
        aggregate_projection_spectra=aggregates,aggregate_norm_spectra=norms,
        family_spectra=all_families,layer_order_nulls=nulls,roundtrip_checks=checks,
        fourier_plus_dct_roundtrip_seconds=roundtrip_seconds,total_analysis_seconds=time.perf_counter()-start,
        sources={str(p.relative_to(src)):sha(p) for p in [original_path,*[src/'checkpoints'/f'{a}.safetensors' for a in ['retained','interpolated']],*[src/'reference/phase4_checkpoints'/f'window_{s}.safetensors' for s in [4,12]]]},
        limits=['Post-hoc weights only: no new language evaluations or training.',
            'Phase 6 trained ordinary native blocks then lifted them into curves; this is not a curve-coefficient-training comparison.',
            'Only six of 28 rows changed; update rank <=6 follows from this design and is not discovered compression.',
            'Global Fourier/DCT interpolation reproduces sampled 28 weights, not the earlier Hermite curve between samples nor its 30 original knots.',
            'Layer permutations are descriptive controls; these permuted weights are not claimed to preserve model function.',
            'Full spectral energy is not functional importance. Band interventions require validation loss/KL measurement.',
            'Coordinates are unaligned across layers; neuron/head symmetries can alter spectra without altering a suitably permuted network function.',
            'Only initial/final phase-6 weights are saved; this is endpoint drift, not a time-resolved training trajectory.'])
    (OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    print(json.dumps({'finished':True,'seconds':diag['total_analysis_seconds'],'fft_dct_roundtrip_seconds':roundtrip_seconds,
        'retained':aggregates['retained_final'],'retained_delta':aggregates['retained_delta']}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
