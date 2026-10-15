"""Render completed receipt metrics only; never evaluate models or read datasets."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
receipt = root / 'regularized-language-run' / 'COMPLETION.json'
assert hashlib.sha256(receipt.read_bytes()).hexdigest() == 'cf32edede7e22bb260d80b772d460b3d01bdce22879380de905fd0b02580f97e'
result = json.loads(receipt.read_text())
assert result['advance_gate_passed'] and all(r['first'] == r['second'] for r in result['results'])
arms = ['original', 'local', 'cross']
labels = ['Original hierarchy', 'Local rotation', 'Cross-scale query rotation']
colors = ['#677789', '#bd7834', '#126b61']
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for arm, label, color in zip(arms, labels, colors):
    values = [r['ce'][arm] for r in result['paired']]
    axes[0].plot(range(3), values, 'o-', label=label, color=color, linewidth=2)
axes[0].set_xticks(range(3), [str(r['seed']) for r in result['paired']])
axes[0].set_xlabel('Training seed')
axes[0].set_ylabel('Confirmation byte cross-entropy (nats; lower is better)')
axes[0].set_ylim(1.69, 1.82)
axes[0].set_title('All seeds; zoomed CE axis')
axes[0].legend(fontsize=8, loc='upper left')
for arm, label, color in zip(arms[:2], labels[:2], colors[:2]):
    differences = [r['ce'][arm] - r['ce']['cross'] for r in result['paired']]
    axes[1].plot(range(3), differences, 'o-', label=f'{label} minus cross', color=color, linewidth=2)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_xticks(range(3), [str(r['seed']) for r in result['paired']])
axes[1].set_xlabel('Training seed')
axes[1].set_ylabel('Paired CE reduction (nats; positive favors cross)')
axes[1].set_title('Cross wins both controls on all three seeds')
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.2)
fig.suptitle('Promising local hierarchy rotation pilot — not SOTA', fontsize=16)
fig.text(.5, .035, '18 trials; 6,000 updates each; 3.072M byte targets/trial; T=128; 9 checkpoints confirmed twice in FP32.\n54,016 baseline / 54,024 rotation parameters. No matched FLOPs or measured latency; confirmation region not globally pristine.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .12, 1, .94))
fig.savefig(root / 'CONFIRMED_ROTATION_CHART.png', dpi=160)
plt.close(fig)
