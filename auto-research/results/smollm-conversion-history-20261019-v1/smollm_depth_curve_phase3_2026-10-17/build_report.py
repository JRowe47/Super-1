"""Build plots and a self-contained report from the completed phase-3 run."""
from pathlib import Path
import base64, csv, html, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
COLORS={'direct_192':'#8b9aaa','direct_768':'#3f6b50','curve_fixed8':'#6850b5','curve_mixed':'#007d88','qk':'#be841d'}
NAMES={'direct_192':'One pass · width 192','direct_768':'One pass · width 768',
       'curve_fixed8':'Curve · trained at 8','curve_mixed':'Curve · mixed-depth training'}


def read(name):return json.loads((OUT/name).read_text())


def name(config):
    if config['kind']=='qk':return 'Phase-2 shared Q/K'
    suffix=' + shared Q/K' if config['kind']=='combined' else ''
    if config['student'].startswith('direct'):return NAMES[config['student']]+suffix
    return NAMES[config['student']]+f" · N={config['steps']}"+suffix


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def path_projection():
    data=np.load(OUT/'curve_trajectories.npz')
    anchor=data['anchor'][-1];teacher=anchor+data['target'][-1]
    paths={n:data[str(n)][:,-1,:] for n in [1,2,4,8,64]}
    cloud=np.vstack([p-anchor for p in paths.values()]+[(teacher-anchor)[None]])
    _,s,vh=np.linalg.svd(cloud,full_matrices=False);basis=vh[:2].T
    projection={n:(p-anchor)@basis for n,p in paths.items()}
    explained=float((s[:2]**2).sum()/(s**2).sum())
    return data,projection,(teacher-anchor)@basis,explained


def plot_path(ax,projection,teacher,explained,large=False,footer_y=-.2):
    palette={1:'#c56b48',2:'#c59b21',4:'#008791',8:'#6850b5',64:'#7d8d9a'}
    for n in [64,8,4,2,1]:
        points=projection[n]
        ax.plot(points[:,0],points[:,1],linestyle='--' if n==64 else '-',
          marker=None if n==64 else 'o',markersize=3 if n!=1 else 4,
          color=palette[n],alpha=.7 if n==64 else .95,label=f'{n} steps'+(' · diagnostic' if n==64 else ''))
        ax.scatter(points[-1,0],points[-1,1],s=25,color=palette[n],zorder=4)
    ax.scatter([0],[0],s=75,facecolor='white',edgecolor='#213343',zorder=5,label='Start after attention')
    ax.scatter(teacher[0],teacher[1],marker='*',s=150,color='#152a38',zorder=6,label='Original MLP endpoint')
    ax.set_xlabel('Projected state coordinate 1');ax.set_ylabel('Projected state coordinate 2')
    ax.grid(alpha=.15);ax.legend(fontsize=8 if large else 7,loc='best',frameon=False)
    ax.text(0,footer_y,f'Last token of first validation window · 2D projection captures {explained:.1%} of trajectory energy.\nGeometry is a projection; it is not semantic distance or a full-model benchmark.',transform=ax.transAxes,fontsize=8,color='#64748b')


def figures(training,test,timing,selection):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'text.color':'#172b3a','axes.labelcolor':'#334155',
        'xtick.color':'#475569','ytick.color':'#475569'})
    data,projection,teacher,explained=path_projection()
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    fig.subplots_adjust(left=.08,right=.96,bottom=.15,top=.82,wspace=.28,hspace=.55)
    fig.suptitle('SmolLM2-135M',x=.08,y=.965,ha='left',fontsize=27,fontweight='bold')
    fig.text(.08,.921,'Learned depth curves · four trained students · only the first block’s MLP changes',fontsize=13,color='#475569')
    fig.text(.08,.882,'Validation quality gate: '+('at least one curve passed' if selection['passed_gate'] else 'no curve passed'),fontsize=11,color='#007d88' if selection['passed_gate'] else '#a84f3d')
    ax=axes[0,0]
    for row in training:
        key=row['spec']['id'];h=row['history']
        ax.plot([r['update'] for r in h],[math.sqrt(r['selection_score']) for r in h],color=COLORS[key],lw=2,label=NAMES[key])
    ax.set(title='A  Validation endpoint fitting',xlabel='Optimizer updates',ylabel='Relative RMS error of teacher MLP update')
    ax.grid(alpha=.17);ax.legend(fontsize=8,frameon=False)
    ax=axes[0,1]
    for key in ['curve_fixed8','curve_mixed']:
        rows=sorted([r for r in test if r['configuration'].get('student')==key and r['configuration']['kind']=='student'],key=lambda r:r['configuration']['steps'])
        ns=np.array([r['configuration']['steps'] for r in rows]);p=np.array([r['metrics']['perplexity_ratio_to_original'] for r in rows])
        lo=np.exp([r['metrics']['delta_nll_document_bootstrap_95ci']['lower'] for r in rows]);hi=np.exp([r['metrics']['delta_nll_document_bootstrap_95ci']['upper'] for r in rows])
        ax.plot(ns,p,'o-',lw=2,color=COLORS[key],label=NAMES[key]);ax.fill_between(ns,lo,hi,color=COLORS[key],alpha=.12)
    for key in ['direct_192','direct_768']:
        row=next(r for r in test if r['configuration'].get('student')==key)
        ax.axhline(row['metrics']['perplexity_ratio_to_original'],color=COLORS[key],ls='--',lw=1.5,label=NAMES[key])
    ax.axhline(1,color='#25384a',ls=':',lw=1,label='Original model')
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],['1','2','4','8']);ax.set_yscale('log')
    ax.set_yticks([1,2,3,4,6],['1×','2×','3×','4×','6×'])
    ax.set(title='B  Held-out language quality',xlabel='MLP field evaluations',ylabel='Perplexity / original (lower is better)')
    ax.grid(alpha=.17);ax.legend(fontsize=8,frameon=False)
    ax.text(0,-.28,'Shading: pointwise 95% document-bootstrap intervals.\n16,384 predicted tokens from 16 previously unused test articles.',transform=ax.transAxes,fontsize=8,color='#64748b')
    ax=axes[1,0];chosen=selection['selected_curve']['student']
    for cached in [False,True]:
        rows=sorted([r for r in timing if r['configuration'].get('student')==chosen and r['materialized']==cached],key=lambda r:r['configuration']['steps'])
        ax.plot([r['configuration']['steps'] for r in rows],[r['measurements']['block']['median_paired_ratio'] for r in rows],
                'o-',lw=2,color='#007d88' if not cached else '#6850b5',label='CPU latency · sampled on demand' if not cached else 'CPU latency · cached matrices')
    ns=[1,2,4,8];ax.plot(ns,[.25+.09375*n for n in ns],'--',color='#8b9aaa',label='Learned projection MAC fraction')
    ax.axhline(1,color='#25384a',ls=':',lw=1,label='Original block')
    ax.set_xscale('log',base=2);ax.set_xticks(ns,list(map(str,ns)))
    ax.set(title='C  First-block time and compute',xlabel='MLP field evaluations',ylabel='Ratio to original first block')
    ax.grid(alpha=.17);ax.legend(fontsize=8,frameon=False)
    ax.text(0,-.3,'Selected curve family; batch 1, 128 tokens, FP32, four CPU threads.\nCached matrices use extra memory. These are first-block ratios.',transform=ax.transAxes,fontsize=8,color='#64748b')
    ax=axes[1,1];plot_path(ax,projection,teacher,explained,footer_y=-.3)
    ax.set_title('D  Projected hidden-state depth paths')
    fig.savefig(OUT/'phase3_summary.png',dpi=180,facecolor='white');fig.savefig(OUT/'phase3_summary.svg',facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,7),gridspec_kw={'width_ratios':[1.25,1]})
    fig.subplots_adjust(left=.075,right=.96,bottom=.23,top=.79,wspace=.25)
    fig.suptitle('SmolLM2-135M · learned depth curve',x=.075,y=.95,ha='left',fontsize=25,fontweight='bold')
    fig.text(.075,.885,NAMES[chosen]+' · the same four control vectors at every evaluation resolution',fontsize=13,color='#475569')
    if not selection['passed_gate']:fig.text(.075,.843,'This student did not pass the language-quality gate.',fontsize=11,color='#a84f3d')
    plot_path(axes[0],projection,teacher,explained,large=True)
    axes[0].set_title('State trajectories from 1 to 64 Euler steps')
    axes[1].plot(data['depth'],data['weight_derivative_norm'],color=COLORS[chosen],lw=2.2)
    axes[1].scatter(np.linspace(0,1,4),np.interp(np.linspace(0,1,4),data['depth'],data['weight_derivative_norm']),color=COLORS[chosen],s=35)
    axes[1].set(title='How fast the learned weights change',xlabel='Continuous depth t',ylabel='Frobenius norm of dW(t)/dt')
    axes[1].grid(alpha=.17)
    axes[1].text(0,-.2,'Dots mark the four spline control depths.\nWeight-curve speed measures variation, not information content.',transform=axes[1].transAxes,fontsize=9,color='#64748b')
    fig.savefig(OUT/'depth_curve_paths.png',dpi=180,facecolor='white');fig.savefig(OUT/'depth_curve_paths.svg',facecolor='white');plt.close(fig)
    return explained


def main():
    summary=read('summary.json');training=read('training_summary.json');test=read('test.json');pilot=read('pilot.json')
    validation=read('validation.json');timing=read('timing.json');selection=read('selection.json');mechanism=read('mechanism_diagnostics.json')
    assert summary['status']=='PASS' and len(test)==12
    figures(training,test,timing,selection)
    chosen=selection['selected_curve'];selected=next(r for r in test if r['configuration']['id']==chosen['id'])
    selected_val=next(r for r in validation if r['configuration']['id']==chosen['id'])
    m=selected['metrics'];ci=m['delta_nll_document_bootstrap_95ci']
    quality='At least one learned curve passed the predefined validation quality gate.' if selection['passed_gate'] else 'No learned curve passed the predefined validation quality gate in this run.'
    direct=next(r for r in test if r['configuration']['id']=='direct_768_n1')
    testrows=[];flat=[]
    for row in test:
        c=row['configuration'];v=row['metrics'];a=row['accounting'];interval=v['delta_nll_document_bootstrap_95ci']
        testrows.append([name(c),f"{v['delta_nll']:+.5f}",f"[{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
                        f"{100*(v['perplexity_ratio_to_original']-1):+.2f}%",f"{v['mean_kl']:.5f}",f"{v['top1_agreement']:.1%}",
                        f"{a['block_coefficient_values']:,}",f"{a['block_projection_macs_per_token']/3538944:.1%}"])
        flat.append({'candidate':c['id'],'delta_nll':v['delta_nll'],'ci_lower':interval['lower'],'ci_upper':interval['upper'],
          'perplexity_ratio':v['perplexity_ratio_to_original'],'kl':v['mean_kl'],'top1_agreement':v['top1_agreement'],
          'block_values':a['block_coefficient_values'],'projection_mac_fraction':a['block_projection_macs_per_token']/3538944})
    with (OUT/'test_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    trainrows=[[NAMES[r['spec']['id']],r['updates'],r['best_update'],f"{r['token_presentations']:,}",
        f"{r['field_token_evaluations']:,}",f"{math.sqrt(r['best_validation_relative_mse']):.4f}",f"{r['elapsed_seconds']:.1f}"] for r in training]
    timerows=[]
    for r in timing:
        for scope,v in r['measurements'].items():
            ratios=[p['candidate']/p['original'] for p in v['pairs_ms']]
            timerows.append([name(r['configuration']),'cached' if r['materialized'] else 'on demand',scope,
                f"{v['original_median_ms']:.3f}",f"{v['candidate_median_ms']:.3f}",f"{v['median_paired_ratio']:.3f}×",
                f"{min(ratios):.3f}–{max(ratios):.3f}×",f"{r['accounting']['sampled_matrix_cache_values']*4/1024**2:.2f}"])
    diag=mechanism['metrics'];diagrows=[[n,f"{diag[str(n)]['relative_rms_error']:.4f}"] for n in [1,2,4,8,16,32,64]]
    diagrows.append(['8, frozen state',f"{diag['frozen_state_n8']['relative_rms_error']:.4f}"])
    images={n:base64.b64encode((OUT/n).read_bytes()).decode() for n in ['phase3_summary.png','depth_curve_paths.png']}
    generation_rows=[]
    for g in read('generations.json'):
        generation_rows.append('<h3>'+html.escape(g['prompt'])+'</h3>'+''.join('<p><strong>'+label+':</strong> '+html.escape(g[key])+'</p>' for key,label in [('original','Original'),('curve_n8','Curve, 8 steps'),('curve_n1','Same curve, 1 step')]))
    selected_latency=next(r for r in timing if r['configuration']['id']==chosen['id'] and not r['materialized'])['measurements']
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmolLM2-135M — learned depth curves</title>
<style>*{{box-sizing:border-box}}:root{{color-scheme:light}}body{{margin:0;background:#edf2f5;color:#193246;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1240px;margin:30px auto;background:white;padding:48px 54px;border-radius:16px}}h1{{font-size:42px;line-height:1.12;margin:12px 0}}h2{{font-size:25px;line-height:1.3;margin-top:40px}}h3{{font-size:18px}}.eyebrow{{color:#007d88;font-size:12px;font-weight:700;letter-spacing:.12em}}.lead{{font-size:20px}}.meta{{color:#64748b}}.box{{background:#edf7f7;border-left:4px solid #007d88;padding:20px 24px;border-radius:5px}}img{{width:100%;height:auto}}.scroll{{overflow-x:auto;margin:22px 0}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:10px 12px;border-bottom:1px solid #dde5eb;text-align:left;white-space:nowrap}}th{{background:#edf2f5}}tr:nth-child(even) td{{background:#f8fafb}}pre{{background:#edf2f5;padding:20px;white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace}}a{{color:#006f81}}details{{margin:20px 0}}li{{margin:7px 0}}@media(max-width:700px){{main{{margin:0;padding:24px 18px;border-radius:0}}h1{{font-size:32px}}.lead{{font-size:18px}}}}</style></head><body><main>
<div class="eyebrow">EXPERIMENT 03 · LEARNED MLP WEIGHTS AS A FUNCTION OF DEPTH</div>
<h1>SmolLM2-135M<br>One learned curve, four evaluation depths</h1>
<p class="meta">131,072 training tokens · 4,096 validation tokens · 16,384 test tokens from 16 previously unused test articles</p>
<p class="lead">{quality} The experiment changes the MLP transformation inside the first block; attention still runs once and all 30 outer transformer blocks remain.</p>
<div class="box"><strong>Validation-selected curve:</strong> {html.escape(name(chosen))}. On the fresh test sample, Δ NLL is {m['delta_nll']:+.6f} nats/token, with a 95% document-bootstrap interval [{ci['lower']:+.6f}, {ci['upper']:+.6f}]. Perplexity changes by {100*(m['perplexity_ratio_to_original']-1):+.2f}%; argmax agreement is {m['top1_agreement']:.1%}. {'This candidate passed the validation gate.' if selection['passed_gate'] else 'This is the lowest-KL failed-gate representative, not a successful quality-preserving replacement.'}</div>
<p>The parameter-matched one-pass width-768 student changes test perplexity by {100*(direct['metrics']['perplexity_ratio_to_original']-1):+.2f}%. On-demand CPU latency for the selected curve is {selected_latency['block']['median_paired_ratio']:.3f}× the original first-block measurement and {selected_latency['full_model']['median_paired_ratio']:.3f}× the full-model measurement. Quality and runtime are separate outcomes.</p>
<img alt="Training error, held-out quality by depth, measured first-block cost, and projected state trajectories" src="data:image/png;base64,{images['phase3_summary.png']}">
<h2>What changed from the previous phase</h2>
<p>Phase 2 reduced matrix ranks and coefficient resolution while retaining a fixed computational depth. Here, depth t indexes the weights of a SwiGLU vector field. Four learned Catmull-Rom control vectors define gate, up and down matrices over t ∈ [0,1]. The same fitted curve is sampled with N = 1, 2, 4 or 8 Euler steps, with each update scaled by 1/N.</p>
<pre>u₀ = 0
F(h,t) = D(t) [SiLU(G(t) RMSNorm(h)) ⊙ U(t) RMSNorm(h)]
uᵢ₊₁ = uᵢ + F(a + uᵢ, i/N) / N
output = a + uₙ</pre>
<p>a is the state after the original first attention operation. The original post-attention RMSNorm scales remain frozen and are reused inside F. The depth coordinate now controls computation, rather than indexing a flattened array of weights. This is a chosen continuous-depth construction, not a recovery of an undocumented GSA implementation.</p>
<h2>Controls and training</h2>
{table(['Student','Width','Control vectors','Trainable MLP values','Training steps per example'],[['One pass, narrow',192,1,'331,776',1],['One pass, parameter matched',768,1,'1,327,104',1],['Curve, fixed depth',192,4,'1,327,104',8],['Curve, mixed depth',192,4,'1,327,104','1 / 2 / 4 / 8']])}
<p>Each student starts with importance-selected teacher gate/up neurons and a training-only ridge fit for the down projection. Curve controls start identical. All methods use the same seeded minibatch stream and 3,000 AdamW updates. They see the same number of teacher examples, but their training FLOPs and elapsed times differ. The loss fits the MLP residual update; later transformer blocks do not participate in backpropagation. Hidden-state MSE alone cannot establish preserved language quality.</p>
{table(['Student','Updates','Selected update','Token presentations','Field-token evaluations','Selected val. relative RMS','Training seconds'],trainrows)}
<p>The 40-update pilot used a smaller {pilot.get('training_tokens',32768):,}-token sample and a different minibatch seed. Its validation MSE fell from {pilot['history'][0]['selection_score']:.4f} to {pilot['history'][-1]['selection_score']:.4f}. That established learning and affordable runtime; the main run rebuilt its initialization on the larger sample and did not reuse pilot weights. Checkpoints were selected by validation endpoint MSE every 100 updates. Fixed-depth selection uses N=8; mixed-depth selection averages the four depths equally.</p>
<h2>Fresh held-out language evaluation</h2>
<p>Positive Δ NLL is worse. Perplexity ratios use exp(Δ NLL). Argmax agreement compares with the original model, not ground-truth answer accuracy. All curve depths are reported without selecting among test results. The fixed phase-2 Q/K baseline is reevaluated on this phase's new articles.</p>
{table(['Candidate','Δ NLL','95% CI','PPL change','KL','Argmax agreement','Block values','Block projection MACs / original'],testrows)}
<p>The engineering gate was validation Δ NLL ≤ 0.05 and KL ≤ 0.05. Among curves passing both, selection favors the fewest field evaluations and then lowest KL; otherwise it reports the lowest-KL failed candidate. The choice and its Q/K composition were written before reading test metrics. Validation scores for the selected curve were Δ NLL {selected_val['metrics']['delta_nll']:+.6f} and KL {selected_val['metrics']['mean_kl']:.6f}.</p>
<h2>The learned paths</h2>
<img alt="Hidden-state trajectories at several depths and the learned weight curve derivative" src="data:image/png;base64,{images['depth_curve_paths.png']}">
<p>The plot uses token 128 of the first validation window, fixed before seeing the outcome. The endpoint star is the original teacher MLP output. The axes are a two-dimensional projection of 576-dimensional states. A path close to the star in this projection does not establish low full-dimensional error or good language-model quality.</p>
{table(['Euler evaluations','Relative RMS error on the first validation window'],diagrows)}
<p>The 16/32/64-step runs are exploratory numerical diagnostics. A model optimized at eight steps need not improve when its ODE discretization is refined. The frozen-state ablation evaluates every field at the original a instead of the evolving a+u; it is not a trained alternative. Its output differs from the recurrent eight-step version by {diag['frozen_state_n8']['difference_from_recurrent_relative_rms']:.4f} relative to teacher-update RMS. Relative variation of the four learned control vectors around their mean is {diag['control_relative_variation']:.4f}.</p>
<h2>Runtime and memory accounting</h2>
<p>The original first block has 3,540,096 values and 3,538,944 learned projection MACs per token. Each width-192 field evaluation costs 331,776 projection MACs. At eight evaluations the MLP projection count equals the original MLP's; spline assembly, repeated normalization, residual updates and kernel overhead are additional work. Reducing steps reduces this part of compute, not the number of outer blocks or the KV cache size.</p>
<p>On-demand evaluation samples matrices for each step. The optional cached mode keeps N materialized matrix sets in addition to the spline controls; this costs 1.266 MiB per step in FP32. Coefficient storage, sampled-matrix memory and arithmetic counts are distinct. Matrix assembly costs are recorded in the raw accounting. Equal-precision coefficient counts are not compressed-file size comparisons against the original BF16 checkpoint.</p>
{table(['Candidate','Weights','Scope','Original ms','Candidate ms','Paired ratio','Observed ratio range','Extra matrix cache MiB'],timerows)}
<p>Timings use batch 1, 128-token prefill, four CPU threads, FP32/eager attention and 12 alternating-order pairs after warmup. Ratios below 1 are faster. The first-block measurement includes embedding and rotary setup; full-model timing includes the output head. Shared-host timing is noisy and establishes no GPU claim.</p>
<h2>Correctness and interpretation limits</h2>
<ul><li>The full-width one-step control was bit-identical to the original model.</li><li>{summary['outside_parameter_tensors_checked']} learned tensors outside block 0 remained identical by hash.</li><li>Selected-curve cached decoding versus prefill differed by at most {summary['cached_vs_prefill_max_logit']:.9g} in logits.</li><li>Materialized versus on-demand sampled weights produced exactly matching logits.</li><li>All four exported checkpoints are checked against their saved state and measured held-out losses by <code>verify_exports.py</code>.</li></ul>
<p>Training, validation and test use official WikiText-2 article partitions. All articles sampled during phase-2 validation/test are excluded from the corresponding phase-3 evaluation pools. The test set has 16 documents, so the paired 2,000-resample bootstrap intervals have limited breadth. These are pointwise intervals, not simultaneous intervals across all twelve comparisons. The model may have seen WikiText in pretraining. Fixed 128-token windows are not standard full WikiText perplexity. One training minibatch seed is not a seed-robust architecture comparison.</p>
<p>The experiment changes both the MLP representation and its integration depth. Both four-control curves have half the original MLP's parameter budget. The direct width-768 control matches their stored parameters and their four-step projection MACs; the direct width-192 control matches one field's width. The one-step curve accesses only its first control vector. These controls help separate capacity, optimization and recurrence, but do not establish the best possible curve architecture. All selected checkpoints were at the final update, so the bounded training budget does not establish convergence. No conclusions about all-layer conversion, adaptive halting, long-context quality, cache compression or TinyLlama follow from this run.</p>
<h2>Next decision</h2>
<p>The next experiment should begin with a function-preserving decomposition at full resolution. The original width-1,536 SwiGLU MLP is a sum of neuron contributions: split it into eight width-192 groups, evaluate every group at the same original attention-output state a, and sum the outputs. Algebraically this reproduces the original MLP; its implementation must first meet a floating-point equivalence check. It retains all original MLP weights and is a proposed control, not a measured result in this phase.</p>
<p>Then fit an ordered curve through those contributions and test coarser sampling against that verified reference. Compare keeping a available to each field with the current evolving-state construction, and compare intermediate-path supervision with endpoint-only training. This separates conversion fidelity from the additional compression constraint. Require a curve to beat the parameter-matched direct student and pass the language-quality gate before repeat seeds, longer contexts, broader layer coverage or TinyLlama. Smooth weight functions alone do not establish useful depth invariance.</p>
<details><summary>Greedy continuations, unchanged prompts and no sampling</summary>{''.join(generation_rows)}</details>
<details><summary>Frozen protocol</summary><pre>{html.escape(json.dumps(read('protocol.json'),indent=2))}</pre></details>
<h2>Files and research context</h2>
<p>The archive contains source, pinned versions, sample token arrays, data/selection manifests, initialization and trained checkpoints, per-window metrics, raw timing pairs, trajectory arrays and this report. <code>README.md</code> gives reproduction commands. The original model and regenerable teacher caches are excluded.</p>
<ul><li><a href="https://arxiv.org/abs/1806.07366">Neural Ordinary Differential Equations</a>: learned vector fields and numerical depth. This implementation uses unrolled Euler gradients, not the paper's adjoint method.</li><li><a href="https://arxiv.org/abs/1807.03819">Universal Transformers</a>: recurrence over depth; this experiment does not repeat attention or implement adaptive halting.</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M">Official SmolLM2-135M</a> and <a href="https://huggingface.co/datasets/Salesforce/wikitext">WikiText dataset</a>.</li></ul>
<p class="meta">Process time {summary['process_elapsed_seconds']/60:.1f} minutes; peak RSS {summary['peak_rss_mib']:.0f} MiB. A completed verification status is not a declaration that the quality target passed.</p>
</main></body></html>'''
    (OUT/'phase3_report.html').write_text(document)
    print('Wrote phase3_report.html, summary/path figures and test_summary.csv')


if __name__=='__main__':main()
