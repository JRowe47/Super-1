"""Render a self-contained research report from measured JSON results only."""
from __future__ import annotations
import base64
import csv
import html
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
COLORS = {'svd':'#526270', 'activation_svd':'#007b87',
          'joint_svd':'#9484ac', 'joint_activation_svd':'#6846b5',
          'haar':'#d1931c', 'knots':'#c45c46',
          'importance':'#007b87', 'contiguous':'#c45c46',
          'random_0':'#a4afbb', 'random_1':'#8393a6', 'random_2':'#64748b'}
METHODS = {'svd':'Independent SVD', 'activation_svd':'Activation-aware SVD',
           'joint_svd':'Shared Q/K SVD', 'joint_activation_svd':'Shared Q/K, calibrated',
           'haar':'Haar coefficients', 'knots':'Thinned spline knots'}


def read(name):
    return json.loads((OUT/name).read_text())


def label(row):
    s = row['spec']; method = s['method']
    if method == 'mlp_groups':
        order = {'importance':'Importance', 'contiguous':'Contiguous', 'random_0':'Random seed 0'}[s['ordering']]
        return f"{order} MLP · {s['groups']}/16"
    return f"{METHODS[method]} · {100*s['budget']:g}%"


def color(row):
    return COLORS.get(row['spec'].get('ordering', row['spec']['method']), '#64748b')


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def plot(validation, tested):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.titleweight':'bold', 'axes.labelcolor':'#334155',
                         'text.color':'#172b3a', 'xtick.color':'#475569', 'ytick.color':'#475569'})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw={'height_ratios':[1,1.1]})
    fig.subplots_adjust(left=.08, right=.97, top=.86, bottom=.08, wspace=.63, hspace=.58)
    fig.suptitle('SmolLM2-135M', x=.08, y=.965, ha='left', fontsize=27, fontweight='bold')
    fig.text(.08,.925,'First-block curve resolution · 84 validation configurations · 16,384 held-out test tokens',fontsize=13,color='#475569')
    ax = axes[0,0]
    for method in METHODS:
        rows = [r for r in validation if r['spec']['method']==method and
                (r['spec'].get('target')=='qk' or method.startswith('joint_')) and
                'budget' in r['spec'] and r['spec']['budget']<1]
        rows.sort(key=lambda r:r['accounting']['retained_changed_fraction'])
        ax.plot([r['accounting']['retained_changed_fraction'] for r in rows],
                [max(r['metrics']['mean_kl'],1e-8) for r in rows],
                'o-',ms=4,lw=1.8,color=COLORS[method],label=METHODS[method])
    ax.set(title='A  Query + key: preserve distributions',xlabel='Retained Q/K coefficient budget',ylabel='KL from original (nats/token; lower is better)')
    ax.set_yscale('log');ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks([.125,.25,.5,.75],['12.5%','25%','50%','75%']);ax.grid(axis='y',alpha=.18)
    ax.legend(fontsize=7.6,loc='lower left',ncol=2,frameon=False,handlelength=1.7,columnspacing=1)
    ax = axes[0,1]
    for order in ['importance','contiguous','random_0','random_1','random_2']:
        rows = [r for r in validation if r['spec']['method']=='mlp_groups' and r['spec'].get('ordering')==order and r['spec']['groups']<16]
        rows.sort(key=lambda r:r['spec']['groups'])
        name = {'importance':'Importance-ranked','contiguous':'Original order'}.get(order,order.replace('_',' ').capitalize())
        ax.plot([r['spec']['groups'] for r in rows],[r['metrics']['mean_kl'] for r in rows],
                'o-',ms=4,lw=2.2 if order=='importance' else 1.2,color=COLORS[order],label=name)
    ax.set(title='B  MLP: retain coupled neuron groups',xlabel='Groups retained out of 16 (96 neurons each)',ylabel='KL from original (nats/token; lower is better)')
    ax.set_yscale('log');ax.set_xticks([2,4,8,12]);ax.grid(axis='y',alpha=.18)
    ax.legend(fontsize=8,loc='best',frameon=False)
    ax = axes[1,0]
    y = np.arange(len(tested))
    for i,row in enumerate(tested):
        m=row['metrics'];ci=m['delta_nll_document_bootstrap_95ci'];mean=m['delta_nll']
        ax.plot([ci['lower'],ci['upper']],[i,i],color=color(row),lw=2)
        ax.scatter([mean],[i],color=color(row),s=34,zorder=3)
    ax.axvline(0,color='#7c8a97',lw=1,ls='--');ax.set_yticks(y,[label(r) for r in tested]);ax.invert_yaxis()
    ax.set_xscale('symlog',linthresh=.002,linscale=.8)
    upper=max(r['metrics']['delta_nll_document_bootstrap_95ci']['upper'] for r in tested)
    lower=min(r['metrics']['delta_nll_document_bootstrap_95ci']['lower'] for r in tested)
    ax.set_xlim(min(-.002,lower*1.15),upper*1.2)
    ax.set(title='C  Held-out next-token loss',xlabel='Δ NLL (nats/token) · 95% document bootstrap CI')
    ax.grid(axis='x',alpha=.18);ax.tick_params(axis='y',labelsize=8)
    ax.text(0,-.2,'Linear near zero; logarithmic outside ±0.002.\nPositive = worse; negative = better on this sample.',transform=ax.transAxes,fontsize=8,color='#64748b')
    ax = axes[1,1]
    saved=[r['accounting']['projection_mac_reduction_per_token']/3538944 for r in tested]
    bars=ax.barh(y,saved,color=[color(r) for r in tested],height=.58)
    for i,value in enumerate(saved):ax.text(value+.007,i,f'{value:.1%}',va='center',fontsize=9)
    ax.set_yticks(y,[label(r) for r in tested]);ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8)
    ax.set_xlim(0,max(.22,max(saved)+.08));ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='D  First-block projection compute saved',xlabel='Fraction of block-0 linear MACs removed')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.text(0,-.2,'All 30 transformer blocks still execute.\nHaar and spline controls use dense reconstruction caches.',transform=ax.transAxes,fontsize=8,color='#64748b')
    # Reserve space for full candidate names only once; lower panels share rows.
    axes[1,0].set_position([.265,.13,.29,.29])
    axes[1,1].set_position([.66,.13,.29,.29])
    axes[1,1].set_yticks(y, [])
    axes[1,1].tick_params(axis='y',length=0)
    axes[1,1].set_ylabel('Same candidates and row order as panel C',fontsize=8)
    for lower_ax in axes[1]:lower_ax.set_ylim(len(tested)-.5,-.5)
    fig.savefig(OUT/'phase2_summary.png',dpi=180,facecolor='white')
    fig.savefig(OUT/'phase2_summary.svg',facecolor='white')
    plt.close(fig)


def main():
    summary=read('summary.json');validation=read('validation_sweep.json');tested=read('heldout_test.json')
    timing=read('timing.json');selection=read('selection.json');spectra=read('spectra.json')
    assert summary['status']=='PASS' and len(validation)==84 and len(tested)==8
    plot(validation,tested)
    by_method={r['spec']['method']:r for r in tested if r['spec']['method']!='mlp_groups'}
    joint=by_method['joint_activation_svd'];independent=by_method['activation_svd']
    mlp=next(r for r in tested if r['spec'].get('ordering')=='importance')
    j=joint['metrics'];a=joint['accounting'];ci=j['delta_nll_document_bootstrap_95ci']
    mlpgate=next(r for r in validation if r['spec']['id']==mlp['spec']['id'])['metrics']
    mlp_pass=mlpgate['delta_nll']<=.05 and mlpgate['mean_kl']<=.05
    joint_timing=next(r['measurement'] for r in timing if r['spec']['id']==joint['spec']['id'])
    chart=base64.b64encode((OUT/'phase2_summary.png').read_bytes()).decode()
    testrows=[]
    csvrows=[]
    for r in tested:
        m=r['metrics'];ac=r['accounting'];c=m['delta_nll_document_bootstrap_95ci']
        testrows.append([label(r),f"{m['delta_nll']:+.6f}",f"[{c['lower']:+.6f}, {c['upper']:+.6f}]",
            f"{100*(m['perplexity_ratio_to_original']-1):+.3f}%",f"{m['mean_kl']:.6f}",f"{m['top1_agreement']:.2%}",
            f"{ac['retained_changed_fraction']:.2%}",f"{ac['projection_mac_reduction_per_token']/3538944:.2%}"])
        csvrows.append({'candidate':r['spec']['id'],'delta_nll':m['delta_nll'],'ci_lower':c['lower'],'ci_upper':c['upper'],
            'perplexity_ratio':m['perplexity_ratio_to_original'],'kl':m['mean_kl'],'top1_agreement':m['top1_agreement'],
            'retained_target_fraction':ac['retained_changed_fraction'],'block_projection_mac_fraction_saved':ac['projection_mac_reduction_per_token']/3538944,
            'block_stored_values':ac['block_values'],'dense_cache_values':ac['dense_working_cache_values'],
            'block_residual_relative_rms_error':m['block_residual_relative_rms_error']})
    with (OUT/'heldout_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    valrows=[]
    for r in validation:
        m=r['metrics'];ac=r['accounting']
        valrows.append([r['spec']['id'],f"{m['delta_nll']:+.6f}",f"{m['mean_kl']:.7f}",f"{m['top1_agreement']:.2%}",
            f"{ac['block_values']:,}",f"{ac['projection_mac_reduction_per_token']/3538944:.2%}"])
    timerows=[]
    for r in timing:
        for scope,measure in r['measurement'].items():
            ratios=[p['candidate']/p['original'] for p in measure['pairs_ms']]
            timerows.append([r['spec']['id'],scope,f"{measure['original_median_ms']:.3f}",f"{measure['candidate_median_ms']:.3f}",
                            f"{measure['median_paired_ratio']:.3f}×",f"{min(ratios):.3f}–{max(ratios):.3f}×"])
    costs=[]
    for r in tested:
        ac=r['accounting'];removed=3540096-ac['block_values']
        costs.append([label(r),f"{ac['block_values']:,}",f"{ac['dense_working_cache_values']:,}",
                      f"{ac['block_projection_macs_per_token']:,}",f"{removed/134515008:.3%}"])
    controls=[r for r in validation if r['spec']['id'] in ['exact_curves','full_rank_svd_all','full_rank_activation_qk','full_rank_joint_activation_qk','full_haar_qk','full_knots_qk','full_mlp_groups']]
    controlrows=[[r['spec']['id'],f"{r['metrics']['max_logit_abs']:.9g}",f"{r['metrics']['delta_nll']:+.9g}"] for r in controls]
    headings=['Held-out candidate','Δ NLL','95% CI','PPL change','KL','Argmax agreement','Target values kept','Block MACs saved']
    spec_pre=html.escape(json.dumps(selection,indent=2))
    sample=summary['sample']
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmolLM2-135M — curve resolution research</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#edf2f5;color:#183044;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1220px;margin:32px auto;background:white;padding:50px 56px;border-radius:16px}}h1{{font-size:42px;line-height:1.12;margin:12px 0}}h2{{margin-top:42px;font-size:24px;line-height:1.3}}h3{{font-size:18px}}.eyebrow{{letter-spacing:.12em;font-size:12px;font-weight:700;color:#007b87}}.lead{{font-size:20px;max-width:950px}}.meta{{color:#64748b}}.box{{background:#edf7f7;padding:20px 24px;border-left:4px solid #007b87;border-radius:5px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #dde5eb;white-space:nowrap}}th{{background:#edf2f5;color:#294359}}tr:nth-child(even) td{{background:#f8fafb}}.table-scroll{{overflow-x:auto;margin:22px 0}}code,pre{{font:13px/1.5 ui-monospace,monospace}}pre{{background:#edf2f5;padding:20px;overflow:auto;white-space:pre-wrap}}a{{color:#006c81}}li{{margin:7px 0}}small{{color:#64748b}}details{{margin:20px 0}}@media(max-width:700px){{main{{padding:24px 18px;margin:0;border-radius:0}}h1{{font-size:32px}}.lead{{font-size:18px}}}}@media print{{body{{background:white}}main{{max-width:none;margin:0;padding:16px}}details{{display:block}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}</style></head><body><main>
<div class="eyebrow">EXPERIMENT 02 · FIRST TRANSFORMER BLOCK ONLY</div>
<h1>SmolLM2-135M<br>Decomposition and curve resolution</h1>
<p class="meta">84 validation configurations · 8 held-out candidates · no training · FP32 CPU inference</p>
<p class="lead">Structured query/key factors preserve the model's next-token distributions at substantially reduced coefficient budgets. This experiment tests the structure carried by the coefficients; the cubic-curve wrapper itself is not the source of the low-rank benefit.</p>
<div class="box"><strong>Selected shared Q/K result:</strong> retain {a['retained_changed_values']:,} of {a['changed_dense_values']:,} Q/K values ({a['retained_changed_fraction']:.2%}), using rank {a['ranks']['qk']}. Held-out Δ NLL is {j['delta_nll']:+.6f} nats/token, with a 95% document-bootstrap interval [{ci['lower']:+.6f}, {ci['upper']:+.6f}]. KL is {j['mean_kl']:.6f}; argmax agreement is {j['top1_agreement']:.2%}. This removes {a['projection_mac_reduction_per_token']/3538944:.2%} of the first block's linear-projection MACs, not that fraction of whole-model runtime.</div>
<p><strong>Runtime limit:</strong> the shared-Q/K candidate's median paired latency ratio was {joint_timing['block']['median_paired_ratio']:.3f}× for the first-block measurement and {joint_timing['full_model']['median_paired_ratio']:.3f}× for the complete model. Full-model CPU latency was slower in this short run; no end-to-end speedup was demonstrated.</p>
<img alt="Validation resolution curves, held-out loss intervals and first-block compute savings" src="data:image/png;base64,{chart}">
<h2>What was tested</h2>
<p>The exact phase-1 scalar curve indexed 3,540,096 parameters in block 0. Here, query and key matrices are decomposed independently or jointly into two factors, each stored as coefficient curves. Evaluation uses exact integer knots and two matrix multiplications. Joint Q/K evaluates its shared input factor once. Query/key output coordinates, RoPE, attention heads, residual connections and cache dimensions are preserved.</p>
<p>We also tested coarse-to-fine Haar coefficients and uniformly thinned spline knots in flattened matrix order. These reconstruct a dense matrix once, so their reduced coefficient count does not reduce inference MACs. MLP groups retain coupled gate/up rows and down columns, using importance scores calibrated on real text, contiguous order, or three random orderings.</p>
<p><strong>All 30 transformer blocks still execute.</strong> MLP neuron removal reduces width/contributions, not sequential depth. No gradient steps, distillation, residual rescaling, or downstream recovery training were applied. Only block 0 changes.</p>
<h2>Held-out measurements</h2>
<p>Each candidate was selected using validation only, then measured on 16,384 predicted tokens in 128 independent 128-token contexts. The baseline sample NLL is {summary['test_reference_nll']:.6f}. Positive Δ NLL is worse; small negative changes are not evidence of a general capability improvement. Argmax agreement compares with the original model, not with correct answers.</p>
{table(headings,testrows)}
<p>The importance MLP candidate {'passed' if mlp_pass else 'did not pass'} the predefined validation gate of Δ NLL ≤ 0.05 and KL ≤ 0.05. {'It was the cheapest passing MLP candidate.' if mlp_pass else 'It is retained as the lowest-KL failed-gate representative, not as a successful compression result.'} Random and contiguous MLP controls retain the same number of neurons. Half-budget Haar and spline controls were included in the test regardless of quality.</p>
<h2>Cost accounting</h2>
{table(['Candidate','Block coefficient values','Extra dense cache values','Block linear MACs/token','Whole-model value count removed'],costs)}
<p>The original block has 3,540,096 values and 3,538,944 linear-projection MACs per token. Budgets count both low-rank factors: r(m+n), not just retained singular modes. Reported MACs omit attention-score products, softmax, normalization and other kernels. The whole model has 134,515,008 unique learned values. Counting FP32 coefficient values gives an equal-precision comparison; saved FP32 checkpoints are not claimed to be smaller in bytes than the original BF16 weights. Small integer index arrays and metadata are additional overhead.</p>
<h2>Observed CPU timings</h2>
{table(['Candidate','Scope','Original median ms','Candidate median ms','Paired latency ratio','Observed ratio range'],timerows)}
<p>Four CPU threads, batch 1, 128-token prefill, 12 alternating-order pairs after warmup. A ratio below 1 is faster. The block measurement includes embedding and rotary-embedding setup around the first block. Full-model timing includes the output projection. These are short, shared-host measurements, not evidence of a general speedup or GPU throughput. First-block savings affect a small share of all 30 blocks plus the language-model head.</p>
<h2>Correctness controls</h2>
{table(['Full-resolution control','Maximum absolute logit difference','Δ NLL'],controlrows)}
<p>The exact curve control is bit-identical. Full-rank factorization changes arithmetic order, so it has small FP32 rounding differences. All tested outputs were finite. {summary['untouched_parameter_tensors_checked']} parameter tensors outside block 0 were checked by hash and remained identical. For the shared-Q/K candidate, 32 incremental cached steps versus a full prefill differed by at most {summary['joint_cached_vs_prefill_max_abs_logit']:.9g} in logits.</p>
<h2>Dataset and uncertainty</h2>
<p>WikiText-2 raw, official train/validation/test article partitions. Calibration used 16,384 train tokens; selection used 4,096 validation tokens. Article-title overlap across splits was checked. Each 129-token sample stays within one article and supplies 128 shifted next-token targets. Context resets at each sample. This is a sampled, fixed-context comparison, not standard complete WikiText-2 perplexity. The test split was excluded from this experiment's calibration and selection; it may have appeared in the original model's pretraining.</p>
<p>The 95% intervals resample {ci['clusters']} sampled test documents, with all their selected windows, in 2,000 paired bootstrap replicates. They describe uncertainty on this sample, not model-family generalization. Dataset sampling has one seed per split; model weights have one pinned revision. Do not extrapolate to later layers, long contexts, instruction following, or TinyLlama without testing.</p>
<h2>What this supports next</h2>
<ol><li>Retain a structured Q/K factor family as the working representation. Test its calibration stability and longer contexts before moving to TinyLlama.</li><li>For MLP reduction, test activation-aware factorization of the down projection and a small teacher-distillation or least-squares recovery step. Removing coupled neurons alone need not preserve their combined function.</li><li>To test actual depth, train a shared residual operator or continuous-depth surrogate for block 0, then compare 8, 4, 2 and 1 evaluation steps. Keep this distinct from reducing spline knots, matrix rank or MLP width.</li></ol>
<p>No result here shows that raw flattened weights form a naturally smooth semantic curve, or that ordinary cubic interpolation can replace attention/MLP computation. Low-rank approximation supplies the demonstrated structure.</p>
<details><summary>Validation-selected specifications and reasons</summary><pre>{spec_pre}</pre></details>
<details><summary>All 84 validation measurements</summary>{table(['Configuration','Δ NLL','KL','Argmax agreement','Block values','Block MACs saved'],valrows)}</details>
<details><summary>Singular-spectrum diagnostics</summary>{table(['Matrix','Shape','Rank for 90% energy','Rank for 99% energy'],[[r['matrix'],' × '.join(map(str,r['shape'])),r['rank_90_percent_energy'],r['rank_99_percent_energy']] for r in spectra])}</details>
<details><summary>One illustrative greedy continuation (not a quality benchmark)</summary><p><strong>Prompt:</strong> {html.escape(sample['prompt'])}</p><p><strong>Original:</strong> {html.escape(sample['original'])}</p><p><strong>Shared Q/K:</strong> {html.escape(sample['joint_curve'])}</p></details>
<h2>Reproduction and sources</h2>
<p>The accompanying archive contains exact sample token arrays and manifests, protocol, all per-window results, selected block checkpoints, source code, environment versions, and this report. <code>README.md</code> provides reproduction commands. <code>load_candidate.py</code> restores a selected first block into the pinned reference model without recalibration.</p>
<ul><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M">HuggingFaceTB/SmolLM2-135M</a>, revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2.</li><li><a href="https://huggingface.co/datasets/Salesforce/wikitext">Salesforce WikiText dataset</a>; pinned revision and file hashes in the manifest. Source text attribution: Wikipedia contributors.</li><li><a href="https://arxiv.org/abs/2403.07378">SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression</a>. Inspiration for activation-aware weighting; this is not a reproduction of the full published pipeline.</li><li><a href="https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py">Transformers 4.46.3 Llama implementation</a>.</li></ul>
<p class="meta">Measured run completed in {summary['run_elapsed_seconds']/60:.1f} minutes; peak process RSS {summary['peak_process_rss_mib']:.0f} MiB. Findings apply to this pinned checkpoint, implementation, and sample.</p>
</main></body></html>'''
    (OUT/'phase2_report.html').write_text(body)
    print('Wrote phase2_report.html, phase2_summary.png/.svg and heldout_summary.csv')


if __name__=='__main__':
    main()
