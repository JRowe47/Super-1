import time, platform, json, numpy as np
from scipy import fft, linalg
from common import *

def main():
    begin=time.time(); orders=fit_orderings()
    initial=tensor_map(state(0)); prev=initial; prevstep=0; prevdelta={}
    rows=[]; alignment=[]; rank=[]; checks=[]
    for step in STEPS:
        current=tensor_map(state(step)); t=time.time()
        for (l,fam),x in current.items():
            kinds={'weights':x,'from_init':x-initial[(l,fam)]}
            if step: kinds['interval']=x-prev[(l,fam)]
            for kind,a in kinds.items():
                c=real_fft(a.ravel()); stats=concentration(c)
                row={'step':step,'start':prevstep if kind=='interval' else 0,'layer':l,'family':fam,'kind':kind,
                     'shape':list(a.shape),'n':a.size,**stats,
                     'weight_coordinate_top10':concentration(a)['top10'],
                     'low10':energy(c[:int(c.size*.1)])/(stats['energy']+1e-30),
                     'relative_to_init_norm':float(np.sqrt(energy(a)/(energy(initial[(l,fam)])+1e-30))),
                     'frequency_tree':band_tree(a)}
                if kind=='interval':
                    row['net_norm_per_step']=float(np.sqrt(stats['energy'])/(step-prevstep))
                    pd=prevdelta.get((l,fam))
                    if pd is not None:
                        row['cosine_previous_interval']=float(np.sum(a.astype(np.float64)*pd)/(np.sqrt(energy(a)*energy(pd))+1e-30))
                rows.append(row)
            if step and fam.startswith('mlp'):
                a=x-prev[(l,fam)]; axis=axis_for(fam)
                for name in ['native','random0','random1','random2','pca_frozen']:
                    p=orders[(l,name)]; y=ordered(a,fam,p); c=real_fft(y,axis=axis)
                    st=concentration(c)
                    low,_=mask_coefficients(c,.1,'low',axis=axis)
                    alignment.append({'step':step,'start':prevstep,'layer':l,'family':fam,'ordering':name,
                       'fit_status':'future' if prevstep>=4000 else ('fit' if step==4000 else 'pre_fit'),
                       **st,'low10':energy(low)/(energy(a)+1e-30)})
                if step in [512,2000,8000,32000,143000]:
                    sv=linalg.svdvals(a,check_finite=False)
                    e=sv.astype(np.float64)**2; cs=np.cumsum(e)/e.sum()
                    rank.append({'step':step,'start':prevstep,'layer':l,'family':fam,
                      'rank95':int(np.searchsorted(cs,.95)+1),'top1':float(cs[0]),'top16':float(cs[15]),
                      'stable_rank':float(e.sum()/e[0])})
            if step in [0,4000,8000,143000] and fam=='mlp_in':
                c=real_fft(x,axis=0); recon=real_ifft(c,axis=0)
                flat=x.ravel(); z=fft.rfft(flat,norm='ortho')
                # Reconstruct disjoint frequency leaves with phases intact.
                n=flat.size; edges=np.linspace(0,z.size,9,dtype=int); total=np.zeros(n,np.float32)
                for a,b in zip(edges[:-1],edges[1:]):
                    part=np.zeros_like(z); part[a:b]=z[a:b]
                    total+=fft.irfft(part,n=n,norm='ortho')
                checks.append({'step':step,'layer':l,
                   'fft_relative_energy_error':abs(energy(c)/energy(x)-1),
                   'axis_reconstruction_relative_rms':float(np.sqrt(energy(recon-x)/energy(x))),
                   'eight_band_reconstruction_relative_rms':float(np.sqrt(energy(total-flat)/energy(flat)))})
        if step:
            prevdelta={key:current[key]-prev[key] for key in current}
        prev=current; prevstep=step
        dump('spectra.json',rows); dump('ordering_spectra.json',alignment)
        dump('singular_values_summary.json',rank); dump('transform_checks.json',checks)
        print(f'step {step}: {time.time()-t:.2f}s; {len(rows)} tensor views',flush=True)
    rng=np.random.default_rng(SEED)
    noise=[]
    for seed in range(5):
        a=rng.normal(size=1048576).astype(np.float32); c=real_fft(a)
        noise.append({'seed_index':seed,**concentration(c),'low10':energy(c[:int(c.size*.1)])/energy(c)})
    dump('gaussian_controls.json',noise)
    dump('spectral_environment.json',{'seconds':time.time()-begin,'python':platform.python_version(),
       'numpy':np.__version__,'optimizer_steps':0,'main_matrix_parameters':sum(x.size for x in initial.values())})
    assert max(x['axis_reconstruction_relative_rms'] for x in checks)<1e-5
    assert max(x['eight_band_reconstruction_relative_rms'] for x in checks)<1e-5
    print('SPECTRA COMPLETE',flush=True)

if __name__=='__main__': main()
