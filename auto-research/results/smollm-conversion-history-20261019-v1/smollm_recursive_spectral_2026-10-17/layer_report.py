"""Figures and report section for the layer/permutation extension."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def build_layer_section(load,savefig,img,table):
    layers=load('layer_family_spectra.json');controls=load('permutation_spectral_controls.json');checks=load('permutation_functional_checks.json')
    adapted=[r for r in layers if 'retained_delta' in r['variants']]
    families=['mlp.down_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight']
    density=[]
    for row in layers:
        p=row['variants']['retained_final']['block'];leaf=p['levels']['6']
        density.append(np.array(leaf['fraction'])/(np.array(leaf['real_dof'])/p['n']))
    density=np.array(density)
    changes=np.array([[100*np.sqrt(r['variants']['retained_delta']['families'][f]['energy']/r['variants']['retained_initial']['families'][f]['energy']) for f in families] for r in adapted])
    fine=np.array([r['states']['retained_final']['block']['fine_power_total_variation'] for r in controls])*100
    coarse=np.array([r['states']['retained_final']['block']['coarse_64_power_total_variation'] for r in controls])*100
    fig,axes=plt.subplots(1,3,figsize=(15,8),layout='constrained',gridspec_kw={'width_ratios':[1.25,1.1,1]})
    limit=max(.02,float(np.max(abs(density-1))))
    im=axes[0].imshow(density,aspect='auto',extent=[0,.5,27.5,-.5],cmap='RdBu_r',norm=TwoSlopeNorm(vmin=1-limit,vcenter=1,vmax=1+limit))
    axes[0].set(title='Each layer has its own spectrum',xlabel='Frequency (cycles per serialized weight)',ylabel='28-block position (original source layer)')
    axes[0].set_yticks(range(28),[f"{r['block28']} ({r['original_layer']})" for r in layers],fontsize=8)
    fig.colorbar(im,ax=axes[0],location='bottom',label='Energy density / flat spectrum',shrink=.9)
    im=axes[1].imshow(changes,aspect='auto',cmap='YlGnBu',vmin=0)
    axes[1].set(title='Training changes differ by tensor',xlabel='Projection family',ylabel='Adapted 28-block position')
    axes[1].set_xticks(range(7),['Down','Gate','Up','Q','K','V','O'],rotation=40)
    axes[1].set_yticks(range(6),[r['block28'] for r in adapted])
    for i in range(6):
        for j in range(7):axes[1].text(j,i,f'{changes[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if changes[i,j]>.55*changes.max() else '#17243b')
    fig.colorbar(im,ax=axes[1],location='bottom',label='Update norm / initial weight norm (%)',shrink=.9)
    y=np.arange(6);axes[2].barh(y-.17,fine,height=.32,color='#2563eb',label='Individual FFT bins')
    axes[2].barh(y+.17,coarse,height=.32,color='#e87924',label='Grouped into 64 bands')
    for i in range(6):
        axes[2].text(fine[i]+.7,y[i]-.17,f'{fine[i]:.1f}%',va='center',fontsize=9)
        axes[2].text(coarse[i]+.7,y[i]+.17,f'{coarse[i]:.2f}%',va='center',fontsize=9)
    axes[2].set_yticks(y,[r['block28'] for r in controls]);axes[2].invert_yaxis()
    axes[2].set(title='Same function, different spectrum',xlabel='Total variation of normalized power (%)',ylabel='Permuted 28-block position',xlim=(0,57))
    axes[2].legend(loc='lower right',fontsize=8);axes[2].grid(axis='x',alpha=.15)
    fig.suptitle('Layer identity and neuron coordinates · SmolLM2-135M',fontsize=18,fontweight='bold')
    fig.supxlabel(f"Valid MLP permutations in six blocks: relative RMS logit change {checks['relative_rms_logit_difference']:.2g}; all 1,024 next-token argmax predictions unchanged.",fontsize=10,color='#52627a')
    savefig(fig,'layer_coordinates')
    summary={'functional_checks':checks,'fine_power_total_variation_range':[float(fine.min()/100),float(fine.max()/100)],'coarse_64_power_total_variation_range':[float(coarse.min()/100),float(coarse.max()/100)],'fixed_mask_inverse_mapped_relative_difference_range':[min(r['fixed_down_mask_relative_difference_after_inverse_mapping'] for r in controls),max(r['fixed_down_mask_relative_difference_after_inverse_mapping'] for r in controls)],'all_28_layers_profiled':True,'families_per_layer':9,'cross_layer_alignment_performed':False}
    rows=[]
    for r in controls:
        x=r['states']['retained_final']['block']
        rows.append([r['block28'],r['original_layer'],f"{100*x['fine_power_total_variation']:.2f}%",f"{100*x['coarse_64_power_total_variation']:.3f}%",f"{r['fixed_down_mask_relative_difference_after_inverse_mapping']:.3f}"])
    section=f'''<h2>Layer identity and permutation coordinates</h2>
<p>This extension was added in response to the layer/permutation question after the primary outcomes. Each of the 28 retained-final blocks now has its own recursive FFT, together with separate transforms of all nine tensor families. Both initial/final checkpoints and their updates are additionally resolved for the six adapted blocks. All indices are zero-based. The underlying data retain all six recursion levels; the figure shows 64 leaves for the block spectra.</p>
{img('layer_coordinates.png')}
<p>A layer has many parameterizations that implement the same function. For a gated MLP, f(x) = W<sub>down</sub>[SiLU(W<sub>gate</sub>x) ⊙ W<sub>up</sub>x]. With permutation matrix P, replace the gate/up weights by PW<sub>gate/up</sub> and the down weights by W<sub>down</sub>Pᵀ. The internal neuron labels change, while f(x) is unchanged. We applied one independently drawn permutation to each of the six trained MLPs, preserving every attention matrix, norm, residual channel, and layer position.</p>
<p>On 1,024 training-context targets, the 28-block model's relative RMS logit change was {checks['relative_rms_logit_difference']:.3g}; its maximum absolute logit change was {checks['maximum_absolute_logit_difference']:.3g}, consistent with FP32 accumulation order. All next-token argmax predictions agreed. The NLL difference was {checks['nll_difference']:.3g} nats. Inverting the permutations restored the candidate tensors exactly, and all original model parameter hashes remained unchanged.</p>
<p>Despite the preserved function, the normalized whole-block power spectra changed by {fine.min():.1f}–{fine.max():.1f}% in total variation at individual FFT bins. After aggregation into 64 bands, their difference was only {coarse.min():.2f}–{coarse.max():.2f}%. Total variation is half the L1 distance between normalized power distributions; this is not a fraction of altered model behavior. Fine spectral changes can be large even when coarse energy plots remain nearly flat.</p>
{table(['28-block position','Original layer','Fine-bin power TV','64-band power TV','Fixed-mask relative difference'],rows)}
<p>The final column tests the existing selected frequency mask on the down-projection update, whose column axis labels intermediate MLP neurons. We apply the same frequency mask in permuted coordinates, inverse-map the resulting update, and compare with filtering in native coordinates. Differences of roughly 1.22 times the native filtered-update norm show that the fixed spectral support does not commute with neuron permutation. This is a coordinate-sensitivity diagnostic, not a new loss/perplexity intervention.</p>
<p>Layer separation is necessary but does not itself solve alignment. Different layers are not assumed to be permutations of one common function. Their internal MLP labels may vary independently, whereas arbitrary permutations of the residual stream cannot be made independently inside one block without compatible changes to its connections. Attention has additional coupled constraints, including rotary positions and grouped-query sharing; no attention permutation was attempted here.</p>
<p>The next comparison should preserve original layer identities and tensor families, compare raw versus explicitly aligned internal coordinates, and record inverse mappings so native checkpoints remain exactly recoverable. Matching must use training activations/weights and be tested for functional preservation. For checkpoints descended from the same initialization, keep the existing correspondence as the primary comparison; independently realigning each endpoint could conceal real training changes. Across depth, distinguish unchanged retained blocks from newly interpolated blocks. This study demonstrates the need for the control, not that cross-layer alignment will improve depth interpolation.</p>'''
    return section,summary
