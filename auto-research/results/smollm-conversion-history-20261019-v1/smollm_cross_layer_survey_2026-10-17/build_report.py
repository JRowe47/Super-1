"""Render the completed cross-layer diagnostic and its next-experiment proposal."""
from pathlib import Path
import base64,html,json,hashlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text())
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    s=read('summary.json');f=read('follow_up_summary.json');assert s['status']==f['status']=='PASS'
    a=read('activations.json');w=read('weights.json');skip=read('single_block_ablation.json');pairs=read('pair_results.json')
    assert len(skip)==30 and len(pairs)==12
    normalized=np.load(OUT/'normalized_similarity.npz');robust=read('normalized_similarity.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'text.color':'#193246','axes.labelcolor':'#334155','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(15,12),gridspec_kw={'height_ratios':[1.15,1]})
    fig.subplots_adjust(left=.085,right=.95,bottom=.11,top=.85,wspace=.3,hspace=.47)
    fig.suptitle('SmolLM2-135M · relationships across all 30 blocks',x=.085,y=.96,ha='left',fontsize=23,fontweight='bold')
    fig.text(.085,.918,'Exploratory probe · 1,024 tokens from 7 reused validation articles · original weights throughout',fontsize=12,color='#64748b')
    for ax,key,title,xlabel in [(axes[0,0],'state_unit_token_linear_cka','A  Residual states remain similar','Residual depth (0 = embeddings)'),
                              (axes[0,1],'update_unit_token_linear_cka','B  Block updates differ more','Block index')]:
        m=normalized[key];im=ax.imshow(m,vmin=0,vmax=1,cmap='viridis',origin='lower',interpolation='nearest')
        ax.set(title=title,xlabel=xlabel,ylabel=xlabel);fig.colorbar(im,ax=ax,fraction=.045,pad=.035,label='Unit-token linear CKA')
    ax=axes[1,0];protected={0,1,11,28,29}
    ax.bar([r['removed_block'] for r in skip],[r['delta_nll'] for r in skip],color=['#c56b48' if r['removed_block'] in protected else '#007d88' for r in skip])
    ax.set_yscale('symlog',linthresh=.05);ax.set_yticks([0,.05,.2,1,5,10],['0','.05','.2','1','5','10'])
    ax.set(title='C  Removing one block changes model quality',xlabel='Removed block (zero based)',ylabel='Increase in NLL, nats/token')
    ax.grid(axis='y',alpha=.2);ax.text(0,-.26,'Nonlinear vertical scale. Smaller increases are better.\nOrange: first two blocks, block 11, and last two blocks.',transform=ax.transAxes,fontsize=8,color='#64748b')
    ax=axes[1,1];groups=[r['pair'] for r in pairs if r['operation']=='swap_order'];x=np.arange(len(groups))
    for j,(op,label,color) in enumerate([('swap_order','Swap order','#6850b5'),('remove_both','Remove both','#007d88')]):
        vals=[next(r['delta_nll'] for r in pairs if r['pair']==g and r['operation']==op) for g in groups]
        ax.bar(x+(j-.5)*.36,vals,width=.34,label=label,color=color)
    ax.set_xticks(x,[f'{g[0]} / {g[1]}' for g in groups]);ax.set(title='D  Adjacent-pair checks',xlabel='Block pair',ylabel='Increase in NLL, nats/token')
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=9)
    ax.text(0,-.26,'Exploratory pairs chosen after the initial survey.\nEach change is evaluated separately in the full model.',transform=ax.transAxes,fontsize=8,color='#64748b')
    fig.savefig(OUT/'cross_layer_survey.png',dpi=180,facecolor='white');fig.savefig(OUT/'cross_layer_survey.svg',facecolor='white');plt.close(fig)
    least=s['lowest_single_skip_delta_nll']; most=s['highest_single_skip_delta_nll']
    source_rows=[['ShortGPT','https://arxiv.org/abs/2403.03853','Layer influence and removal; motivates a causal check rather than relying on similarity alone.'],
      ['CKA','https://arxiv.org/abs/1905.00414','Comparison of representational geometry. This probe also normalizes token magnitudes as an exploratory robustness check.'],
      ['Git Re-Basin','https://arxiv.org/abs/2209.04836','Permutation symmetries matter in weight comparison; does not establish that different transformer depths can be merged.'],
      ['LayerSkip','https://arxiv.org/abs/2404.16710','Layer dropout and early-exit training improve early predictions. Its decoding result is not a guarantee for untrained depth subsampling.']]
    blockrows=[[r['block'],f"{r['mean_input_output_cosine']:.4f}",f"{r['residual_update_rms_over_input_rms']:.4f}",
      f"{skip[r['block']]['delta_nll']:.5f}",f"{skip[r['block']]['perplexity_ratio']:.3f}×"] for r in a['blocks']]
    pairrows=[]
    for g in groups:
        swap=next(r for r in pairs if r['pair']==g and r['operation']=='swap_order');drop=next(r for r in pairs if r['pair']==g and r['operation']=='remove_both')
        pairrows.append([f'{g[0]} / {g[1]}',f"{swap['delta_nll']:+.5f}",f"{swap['perplexity_ratio']:.3f}×",f"{drop['delta_nll']:+.5f}",f"{drop['interaction_delta_nll']:+.5f}"])
    p14=next(r for r in pairs if r['pair']==[14,15] and r['operation']=='swap_order')
    p1011=next(r for r in pairs if r['pair']==[10,11] and r['operation']=='remove_both')
    encoded=base64.b64encode((OUT/'cross_layer_survey.png').read_bytes()).decode()
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmolLM2-135M: cross-layer survey</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#edf2f5;color:#193246;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1200px;margin:28px auto;background:white;padding:44px 50px;border-radius:16px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:24px;margin-top:36px}}.lead{{font-size:21px}}.meta{{color:#64748b}}.box{{background:#edf7f7;border-left:4px solid #007d88;padding:20px}}img{{width:100%}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #dce4ea;text-align:left;white-space:nowrap}}th{{background:#edf2f5}}pre{{background:#edf2f5;padding:18px;white-space:pre-wrap}}a{{color:#006f81}}li{{margin:9px 0}}@media(max-width:700px){{main{{margin:0;padding:22px 18px}}h1{{font-size:30px}}}}</style></head><body><main>
<p class="meta">RESEARCH DIRECTION CHECK · ALL 30 ORIGINAL BLOCKS · NO TRAINING</p>
<h1>SmolLM2-135M: relationships across layers</h1>
<p class="lead">Move to a whole-model view and train small groups of layers jointly. Block 0 is unusually sensitive and cannot represent the rest of the depth stack.</p>
<div class="box">The initial 30-block survey took {s['elapsed_seconds']:.1f} seconds and peaked at {s['peak_rss_mib']/1024:.2f} GiB RSS. The normalization and pair follow-up took {f['elapsed_seconds']:.1f} seconds and peaked at {f['peak_rss_mib']/1024:.2f} GiB. These measurements establish that whole-model forward diagnostics fit this CPU environment. They do not establish the cost of training the entire model.</div>
<p>The environment has no CUDA device and an 8 GiB cgroup memory limit. The original model has {s['model_parameters']:,} unique parameters and uses four CPU threads here. Small group distillation is a reasonable next task; substantial full-model pretraining is outside the practical scope of this environment.</p>
<img src="data:image/png;base64,{encoded}" alt="State and update similarity across depth, individual block removal sensitivity, and adjacent-pair interventions">
<h2>What the measurements say</h2>
<p>Raw residual-state CKA has a median adjacent value of {s['median_adjacent_state_cka']:.5f}. Large activation magnitudes can dominate that measure. After normalizing each sampled token vector to unit length before CKA, the median is {f['normalized_state_cka_median_adjacent']:.4f}; for the updates added by successive blocks it is {f['normalized_update_cka_median_adjacent']:.4f}. Similar residual states are therefore a weak basis for claiming interchangeable computations. CKA compares geometry, not exact functional equivalence.</p>
<p>The median adjacent raw-weight cosine is {s['median_adjacent_raw_weight_cosine']:.5f}. Replacing an interior weight tensor by the average of its two neighbors gives a median relative weight error of {s['median_neighbor_average_relative_weight_error']:.3f} when all block parameters are concatenated. This is a weight reconstruction diagnostic, not a language-model interpolation experiment. Neuron and head symmetries are not aligned in these measurements.</p>
{table(['Tensor family','Mean adjacent raw cosine','Centered depth components for 90% energy'],[[r['tensor'],f"{r['adjacent_cosine_mean']:.5f}",r['centered_depth_rank_for_90_percent_energy']] for r in w['tensors']])}
<p>Normalization weights vary across depth in a much lower-dimensional way than the projection matrices in their current coordinates. That is evidence for treating tensor families separately, not forcing the entire block into one shared scalar curve.</p>
<h2>Causal checks: individual blocks and pairs</h2>
<p>Every individual removal increased NLL on this probe. The least damaging was block {least['removed_block']}: perplexity rose {100*(least['perplexity_ratio']-1):.1f}%. Removing block {most['removed_block']} produced a {most['perplexity_ratio']:.0f}× ratio. Block 11 also has a large activation transition and removal cost; the last blocks are sensitive. This supports a piecewise construction with retained transition blocks, subject to confirmation on broader data.</p>
{table(['Pair','Swap Δ NLL','Swap PPL / original','Remove both Δ NLL','Removal interaction'],pairrows)}
<p>The interaction column is joint-removal Δ NLL minus the sum of the two individual-removal Δ NLL values. Positive means superadditive damage on this metric; negative means subadditive damage. Neither sign alone proves compensation, redundancy or a usable depth curve. Swapping blocks 14/15 changes perplexity by {100*(p14['perplexity_ratio']-1):+.1f}%; this tests local order sensitivity without removing either function. These six pairs were chosen after the initial survey to cover depth and the block-11 transition.</p>
<p>A particularly useful interaction is blocks 10/11: removing block 11 alone gives {skip[11]['perplexity_ratio']:.2f}× perplexity, while removing both gives {p1011['perplexity_ratio']:.2f}×. On this probe, removing an additional block reduces the damage. Both changes are severe degradations, but the result shows why individual-block importance cannot fully predict the effect of a joint change.</p>
<details><summary>All 30 individual-block measurements</summary>{table(['Block','Mean input/output cosine','Update RMS / input RMS','Removal Δ NLL','Removal PPL / original'],blockrows)}</details>
<h2>The next experiment</h2>
<ol><li><strong>Represent all 30 blocks at their original depth knots.</strong> Build separate matrix-valued curves for homologous Q, K, V, O, gate, up and down projections, plus separate normalization curves. Keep embeddings and the output mapping as boundary operations. Retain every original weight initially. At the 30 original knots, the native model must reproduce its outputs within a strict numerical tolerance. This construction is proposed, not implemented by this diagnostic.</li>
<li><strong>Compare small joint windows at different depths.</strong> Blocks 12–15 and 4–7 are provisional candidates from this probe: the first has low individual-removal costs, and the second includes the relatively order-tolerant 6/7 pair. Leave the sensitive entrance, block-11 transition and late blocks intact. Fit each window jointly with both intermediate-state and whole-model output supervision. Compare a faithful four-step form with three and two steps, while holding the surrounding model fixed. This is a two-region comparison, not a claim that either window is optimal.</li>
<li><strong>Make alignment and resolution separate experiments.</strong> Test function-preserving MLP neuron permutations before weight interpolation. Attention alignment must respect grouped-query and RoPE structure. First retain the full parameter budget; add a shared low-rank basis only after conversion fidelity is established. Compare with simple layer removal and a parameter-matched direct replacement.</li>
<li><strong>Expand only after the local joint experiment works.</strong> Use multiple depth segments and train several evaluation resolutions, then test 30 → 24 → 20 → 15 full-stack steps. Choose nonuniform sampling using observed functional error and sensitivity, rather than assuming spline curvature equals information. Check language quality, actual runtime and KV-cache behavior separately.</li></ol>
<p>This changes the optimization target from approximating an isolated early-block MLP to preserving a trajectory through several actual transformer blocks. The earlier one-block negative result does not settle whether this broader construction can work. A full-depth curve with all original knots is a coordinate change; useful coarse sampling remains an empirical question.</p>
<h2>Limits and verification</h2>
<p>This is a direction-finding probe: eight 129-token windows supply 1,024 shifted prediction targets from seven phase-3 validation articles. They were already used in phase 3. No untouched test set, task benchmark, long-context evaluation or training-seed sweep is claimed. All candidates use the same probe. CKA uses 256 evenly indexed tokens; the normalized variant and six-pair checks were added after inspecting the initial survey.</p>
<p>Token positions are dependent; the small sample does not justify narrow confidence intervals or broad claims of layer redundancy. Residual updates are compared on their natural inputs along the same token trajectories, rather than matched hidden-state inputs. Removing two blocks changes the input distribution seen by later blocks. These are interventions in the full network, not isolated function-equivalence tests.</p>
<p>All {s['parameter_tensors_checked']} original learned tensors are unchanged by hash after each script. The full model after restoration has a maximum logit difference of {s['restored_max_logit_difference']:.1f} in the initial restoration check; the pair follow-up has exactly matching restored per-window losses. No new model checkpoint is trained or promoted.</p>
<h2>Reproduce</h2>
<pre>python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python download_model.py
python run_survey.py
python follow_up.py
python build_report.py</pre>
<p>Python 3.12; pinned model revision {html.escape(read('protocol.json')['revision'])}. The package includes the exact probe tokens, source manifests, result arrays and plotting code. It excludes the downloadable original checkpoint. The two diagnostic scripts overwrite their own result files and do not train or save modified model weights.</p>
<h2>Research context</h2>
<ul>{''.join('<li><a href="'+url+'">'+name+'</a>: '+html.escape(note)+'</li>' for name,url,note in source_rows)}</ul>
<p>The supplied Rohan comments motivate functions of model depth, orthogonal basis initialization and input-conditioned coefficients. Those claims are not an implementation specification. This survey tests neither input-conditioned curve coefficients nor the separate hierarchical-state and cache claims.</p>
</main></body></html>'''
    (OUT/'cross_layer_report.html').write_text(document)
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in {'reference_model','__pycache__'} for part in p.relative_to(ROOT).parts) and p.name!='SHA256SUMS']
    sums=ROOT/'SHA256SUMS';sums.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in sorted(files)))
    files.append(sums);archive=ROOT.parent/'smollm_cross_layer_survey_2026-10-17.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'report':str(OUT/'cross_layer_report.html'),'archive':str(archive),'files':len(files),'archive_bytes':archive.stat().st_size},indent=2))


if __name__=='__main__':main()
