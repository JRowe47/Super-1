"""Render separate quality and memory measurements from immutable JSON evidence."""
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / '.super1-plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    if not os.path.ismount('/Volumes/SP AI 01_16') or not os.access(ROOT, os.W_OK):
        raise RuntimeError('Mounted writable SSD workspace required')
    quality_path = ROOT / 'auto-research/results/function-kv-text-descent-512-20261018-v1/execution/RESULT.json'
    memory_dir = ROOT / 'auto-research/results/function-kv-text-descent-20261018-v1/execution/long-context'
    exact_path, hierarchy_path = memory_dir / 'exact.json', memory_dir / 'h16.json'
    quality = json.loads(quality_path.read_text())['rows']
    exact, hierarchy = [json.loads(p.read_text()) for p in (exact_path, hierarchy_path)]
    groups = ('nano', 'curve', 'h16')
    samples = [[r['confirmation']['ce'] for r in quality
                if r['arm'] == arm and r.get('variant', 'selected') == 'selected'] for arm in groups]
    assert all(len(s) == 3 for s in samples)
    means = [statistics.mean(s) for s in samples]
    memory = [exact['peak_rss_bytes'] / 2**20, hierarchy['peak_rss_bytes'] / 2**20]
    ratio = memory[0] / memory[1]
    colors = ['#9aa8b5', '#425a73', '#138b7b']
    ink, muted = '#152b3b', '#536674'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.labelcolor': muted, 'xtick.color': ink, 'ytick.color': muted,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(15, 8.4), facecolor='#f7f9fb')
    grid = fig.add_gridspec(1, 2, left=.07, right=.96, bottom=.30, top=.72, wspace=.30)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    fig.text(.07, .94, 'Learned hierarchical KV memory', fontsize=26, weight='bold', color=ink)
    fig.text(.07, .889, 'Function weights and inference layer sampling preserved', fontsize=15, color=muted)
    for ax in axes:
        ax.set_facecolor('#f7f9fb')
        ax.grid(axis='y', color='#dfe6eb', linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#c8d3dc')
        ax.tick_params(axis='both', length=0, pad=9)
    left, right = axes
    left.set_title('Text quality at 512-byte context', loc='left', fontsize=17, weight='bold', color=ink, pad=25)
    left.bar(range(3), means, color=colors, width=.58, zorder=2)
    for i, values in enumerate(samples):
        left.scatter([i-.13, i, i+.13], values, color=ink, s=31, edgecolors='white', linewidth=.7, zorder=4)
        left.text(i, max(values)+.11, f'{means[i]:.3f}', ha='center', fontsize=16, weight='bold', color=ink)
    left.set_xticks(range(3), ['Pinned\nnanoGPT', 'Function weights\nFull KV', 'Function weights\nHierarchy (16 entries)'])
    left.set_ylim(0, max(max(s) for s in samples)*1.24)
    left.set_ylabel('Next-byte cross-entropy (lower is better)', labelpad=13)
    right.set_title('Peak RAM at 100K byte tokens', loc='left', fontsize=17, weight='bold', color=ink, pad=25)
    right.bar(range(2), memory, color=colors[1:], width=.55, zorder=2)
    for i, value in enumerate(memory):
        right.text(i, value+75, f'{value:,.1f} MiB', ha='center', fontsize=16, weight='bold', color=ink)
    right.set_xticks(range(2), ['Function weights\nFull KV', 'Function weights\nHierarchy (16 entries)'])
    right.set_ylim(0, max(memory)*1.24)
    right.set_ylabel('Total process peak RSS (MiB)', labelpad=13)
    right.text(.96, .91, f'{ratio:.2f}× lower\npeak RAM', transform=right.transAxes,
               ha='right', va='top', fontsize=16, weight='bold', color='#087668',
               bbox={'facecolor':'#e3f2ee','edgecolor':'none','pad':12})
    fig.text(.07, .195, 'Matched base training batches; mean and all 3 seeds shown.', fontsize=11, color=muted)
    fig.text(.07, .158, 'Models trained at 512 bytes; hierarchy gets extra pooling training.', fontsize=11, color=muted)
    fig.text(.574, .195, 'Separate model trained at 128 bytes; one paired CPU run.', fontsize=11, color=muted)
    fig.text(.574, .158, 'FP32; 100,000 input tokens + 16 decode steps; chunk size 128.', fontsize=11, color=muted)
    fig.text(.07, .075, 'Tiny byte models, not SOTA or 70B. Memory at 100K does not establish quality or recall at 100K.',
             fontsize=12, weight='bold', color=ink)
    fig.text(.07, .037, 'Peak RAM includes runtime, allocator and attention temporaries. The two panels are separate experiments.',
             fontsize=11, color=muted)
    target = ROOT / 'graphs/function_weight_kv_text_descent'
    fig.savefig(target.with_suffix('.png'), dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(target.with_suffix('.svg'), facecolor=fig.get_facecolor(), metadata={'Date':None})
    plt.close(fig)
    receipt = {'inputs': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (quality_path,exact_path,hierarchy_path)},
               'quality_means':dict(zip(groups,means)), 'quality_seeds':samples,
               'peak_ram_mib':dict(zip(('exact','h16'),memory)), 'peak_ram_reduction':ratio,
               'outputs':{str(target.with_suffix(ext).relative_to(ROOT)):hashlib.sha256(target.with_suffix(ext).read_bytes()).hexdigest()
                          for ext in ('.png','.svg')}}
    target.with_suffix('.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'chart':str(target.with_suffix('.png')),'peak_ram_reduction':ratio}))

if __name__ == '__main__':
    main()
