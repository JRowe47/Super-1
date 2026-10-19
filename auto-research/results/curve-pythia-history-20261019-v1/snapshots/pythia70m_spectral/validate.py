"""Numerical and immutable-input audit; no training."""
import hashlib, json, importlib.metadata, numpy as np
from common import *

def main():
    manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text()); hashes=[]
    for r in manifest:
        p=ROOT/'checkpoints'/r['revision']/'model.safetensors'
        h=hashlib.sha256()
        with p.open('rb') as f:
            while c:=f.read(2**20): h.update(c)
        ok=h.hexdigest()==r['sha256']; assert ok
        hashes.append({'step':r['step'],'sha256_matches_download':ok})
    transforms=json.loads((OUT/'transform_checks.json').read_text())
    perms=json.loads((OUT/'permutation_checks.json').read_text())
    spectra=json.loads((OUT/'spectra.json').read_text())
    assert len(spectra)==1368
    assert set(r['step'] for r in spectra)==set(STEPS)
    assert all(r['argmax_agreement']==1 for r in perms)
    assert all(r['relative_rms_logit_difference']<1e-5 for r in perms)
    max_tree_err=max(abs(sum(v)-1) for r in spectra if r['energy']>0 for v in r['frequency_tree'].values())
    assert max_tree_err<1e-10
    checks=[]
    for n in [31,32,511,512]:
        a=np.random.default_rng(SEED+n).normal(size=(3,n)).astype(np.float32)
        c=real_fft(a); b=real_ifft(c)
        err=np.sqrt(energy(a-b)/energy(a)); assert err<1e-6
        checks.append({'n':n,'relative_reconstruction_rms':float(err),'relative_energy_error':abs(energy(c)/energy(a)-1)})
    a=tensor_map(state(64000)); b=tensor_map(state(143000)); late=[]
    for l in range(6):
        d=b[(l,'q')]-a[(l,'q')]
        late.append({'layer':l,'query_delta_energy':energy(d),'head_energy':[energy(z) for z in d.reshape(8,64,512)],
          'max_abs_query_weight':float(np.abs(b[(l,'q')]).max()),'max_abs_query_delta':float(np.abs(d).max()),
          'top1_weight_energy_fraction':float(np.max(d*d)/energy(d))})
    dump('late_query_check.json',late)
    provenance={}
    for name,p in [('original_project_archive',ROOT.parent/'project_sources'/'01-geometric_curve_memory_project_2026-10-15.zip'),
                   ('previous_fft_archive',ROOT.parent/'prior_fft'/'smollm_recursive_spectral_2026-10-17.zip')]:
        if p.exists(): provenance[name]={'filename':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    prior=ROOT.parent/'prior_fft'/'extracted'/'smollm_recursive_spectral'/'results'/'summary.json'
    if prior.exists(): provenance['previous_fft_summary']=json.loads(prior.read_text())
    dump('continuity_provenance.json',provenance)
    dump('validation.json',{'checkpoint_hashes':hashes,'extra_transform_checks':checks,
      'max_frequency_tree_sum_error':max_tree_err,
      'max_axis_reconstruction_relative_rms':max(r['axis_reconstruction_relative_rms'] for r in transforms),
      'max_band_reconstruction_relative_rms':max(r['eight_band_reconstruction_relative_rms'] for r in transforms),
      'tensor_views':len(spectra),'checkpoint_count':len(STEPS),'passed':True})
    versions={m:importlib.metadata.version(m) for m in ['torch','transformers','safetensors','numpy','scipy','matplotlib','tokenizers']}
    dump('package_versions.json',versions)
    print('VALIDATION PASSED',versions,flush=True)

if __name__=='__main__': main()
