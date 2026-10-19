"""Linear-algebra/resource planning only; no new model is trained or evaluated."""
import math
import numpy as np
from common import OUT,dump

def basis(t,knots=30):
    a=np.zeros(knots)
    if abs(t-round(t))<1e-10:return a
    j=math.floor(t);u=t-j
    a[j]=u**3-2*u**2+u;a[j+1]=u**3-u**2
    return a

def main():
    grids={}
    for n in [28,26]:
        coordinates=[(i+.5)*30/n-.5 for i in range(n)]
        matrix=np.stack([basis(t) for t in coordinates])
        singular=np.linalg.svd(matrix,compute_uv=False)
        collisions=[{'sample_index':i,'original_knot':round(t)} for i,t in enumerate(coordinates) if abs(t-round(t))<1e-10]
        rank=int(np.linalg.matrix_rank(matrix))
        grids[str(n)]={'coordinates':coordinates,'native_knot_collisions':collisions,
            'derivative_sample_rank':rank,'arbitrary_independent_sample_corrections_supported':rank==n,
            'nonzero_condition_number':float(singular[0]/singular[rank-1]),
            'right_inverse_error':float(np.abs(matrix@np.linalg.pinv(matrix)-np.eye(n)).max())}
    body=28*3540096;original=134515008
    result={'scope':'Feasibility planning only. No global-curve model trained or quality-tested.',
        'grids':grids,'proposed_28_block_trainable_values':body,
        'original_preservation':'Clone all reduced-model decoder blocks before training; keep embeddings, tied head and final norm fixed. Store all original knots separately.',
        'parameter_gradient_and_adam_plus_original_bytes':4*(4*body+original),
        'parameter_gradient_and_adam_plus_original_gib':4*(4*body+original)/2**30,
        'memory_limit':'This is only parameter/gradient/moment/original storage. Excludes activations, teacher cache, temporary buffers, framework overhead and curve construction.',
        'curve_rule':'For N=28 use C0(t)+a(t) A28+ D, with a 30-knot Hermite base computed from original blocks and 28 target-minus-base tensors. The 28 sample locations do not coincide with original knots.',
        'collision_limit':'At the uniform N=26 grid, samples 6 and 19 land on immutable original knots 7 and 22. Those sample weights cannot be changed independently in a curve depending only on depth. They must remain fixed or the sampling grid/parameterization must change.'}
    dump(OUT/'global_curve_feasibility.json',result)
    print({n:{'rank':v['derivative_sample_rank'],'collisions':v['native_knot_collisions'],'condition':v['nonzero_condition_number']} for n,v in grids.items()})
    print('Parameter-state lower bound GiB',result['parameter_gradient_and_adam_plus_original_gib'])

if __name__=='__main__':main()
