"""Extract a native component curve and its recorded behavioral effects."""
import argparse,json
from pathlib import Path
import numpy as np
from causal_probe import ROOT,OUT,read,project
ap=argparse.ArgumentParser();ap.add_argument('component');ap.add_argument('--output',type=Path);a=ap.parse_args()
cats={c['id']:c for c in read('candidate_catalog.json')};c=cats[a.component]
z=np.load(ROOT/'native_qkv_curves.npz');delta=[z[f'step1000_layer{l}'].astype(np.float64)-z[f'step512_layer{l}'] for l in range(6)]
v=project(c,delta);before=z[f"step512_layer{c['layer']}"];after=z[f"step1000_layer{c['layer']}"]
result=dict(component=c,update_norm=float(np.linalg.norm(v)),recorded_effects=[r for r in read('confirmation_effects.json') if r['id']==a.component])
if a.output:
    np.savez_compressed(a.output,before=before,after=after,component_update=v,
      rollback=(after.astype(np.float64)-v).astype(np.float32),transplant=(before.astype(np.float64)+v).astype(np.float32),metadata_json=np.array(json.dumps(result)))
    result['saved_full_resolution_curves']=str(a.output)
print(json.dumps(result,indent=2))
