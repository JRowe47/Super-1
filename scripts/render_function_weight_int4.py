"""Render measured packed bytes and same-split quality, retaining all seed points."""
import hashlib,json,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'.super1-plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=ROOT/'auto-research/results/function-weight-int4-20261018-v1/execution'
rows=json.loads((H/'RESULT.json').read_text())['rows']; summary=json.loads((H/'SUMMARY.json').read_text())
compact=json.loads((H/'compact/RESULT.json').read_text())
arms=['nano_fp16','nano_q4','curve_fp16','curve_b4','curve_be4']
labels=['nanoGPT\nFP16','nanoGPT\nint4','Curve weights\nFP16','Curve weights\nint4 bases','Curve weights\nint4 bases\n+ embedding']
colors=['#8795a6','#516a89','#bba67b','#45a795','#087c70']
fig,axes=plt.subplots(1,2,figsize=(15,7.5),facecolor='#f7f9fb')
fig.subplots_adjust(left=.065,right=.98,top=.76,bottom=.28,wspace=.22)
fig.text(.065,.92,'Function-weight compression with recovery training',fontsize=25,weight='bold',color='#152b3b')
fig.text(.065,.86,'Matched extra training • Three seeds • Full attention at 512-byte context',fontsize=14,color='#536674')
for ax in axes:
    ax.set_facecolor('#f7f9fb'); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    ax.set_xticks(range(len(arms)),labels,fontsize=10)
values=[next(r['file_bytes'] for r in compact['rows'] if r['arm']==a)/1024 for a in arms]
axes[0].bar(range(len(arms)),values,color=colors,width=.62)
for i,v in enumerate(values): axes[0].text(i,v+6,f'{v:.1f}',ha='center',weight='bold')
axes[0].set_ylim(0,max(values)*1.2); axes[0].set_ylabel('Serialized deployment file (KiB), lower is better')
axes[0].set_title('Compact files, including metadata and scales',loc='left',fontsize=14,pad=15)
for i,a in enumerate(arms):
    values=[r['confirmation']['ce'] for r in rows if r['arm']==a and r['variant']=='selected']
    axes[1].scatter([i-.10,i,i+.10],values,color=colors[i],s=50,zorder=3)
    axes[1].plot([i-.25,i+.25],[statistics.mean(values)]*2,color=colors[i],lw=3)
    axes[1].text(i,max(values)+.014,f'{statistics.mean(values):.3f}',ha='center',weight='bold')
axes[1].set_ylim(min(r['confirmation']['ce'] for r in rows if r['variant']=='selected')-.035,max(r['confirmation']['ce'] for r in rows if r['variant']=='selected')+.065)
axes[1].set_ylabel('Confirmation next-byte cross-entropy, lower is better')
axes[1].set_title('Dots: all seeds; line: mean',loc='left',fontsize=14,pad=15)
gate=summary['gates']['curve_be4']
fig.text(.065,.13,f"Bases + embedding: {gate['payload_reductions'][0]:.2f}× smaller weight payload; {compact['compact_file_reductions']['curve_be4'][0]:.2f}× smaller deployment file versus FP16 curve.",fontsize=13,weight='bold',color='#087c70')
fig.text(.065,.075,'Single-book tiny-model pilot. Weight bytes are not total RAM. Curves, learned t and selectable depth retained.\nOriginal file gate failed (2.81×); lossless packaging follow-up shown here. No SOTA or long-context quality claim.',fontsize=11,color='#536674',linespacing=1.5)
png=ROOT/'graphs/function_weight_int4.png'; svg=png.with_suffix('.svg')
fig.savefig(png,dpi=170); fig.savefig(svg); plt.close(fig)
receipt=dict(inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [H/'RESULT.json',H/'SUMMARY.json',H/'compact/RESULT.json']},outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [png,svg]})
png.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(png)
