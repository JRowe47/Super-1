"""Publication-style static figures generated only from measured result files."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
read=lambda name:json.loads((ROOT/'results'/name).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.titleweight':'bold','axes.labelsize':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.edgecolor':'#c8ced6','text.color':'#162735',
    'axes.labelcolor':'#162735','xtick.color':'#43596a','ytick.color':'#43596a',
    'figure.facecolor':'white','axes.facecolor':'white','grid.color':'#e3e8ec',
    'savefig.facecolor':'white','svg.fonttype':'none'})
BLUE='#1965b0';TEAL='#078c83';ORANGE='#d47219';PURPLE='#8257a6';GRAY='#667b89'
def style(ax,title,xlabel,ylabel):
    ax.set_title(title,loc='left',pad=12);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.7);ax.set_axisbelow(True)
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def pythia():
    rows=read('pythia_evaluations.json');checks=read('pythia_first_step_checks.json')['update_comparisons']
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.1),layout='constrained')
    fig.suptitle('Pythia-70m: curve continuation works; the optimizer changes learning',fontsize=15,fontweight='bold')
    configs=[('native_adamw','Native Adam',BLUE,'-','o'),
        ('curve_transport_adamw','Curve + transported Adam',TEAL,'--','x'),
        ('curve_adamw','Adam on curve coefficients',ORANGE,'-','s')]
    for arm,label,color,ls,marker in configs:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['step'])
        xx=[r['step'] for r in rr]
        axs[0,0].plot(xx,[r['validation']['nll'] for r in rr],label=label,color=color,ls=ls,marker=marker,lw=2,ms=5)
        axs[0,1].plot(xx,[100*r['retention']['repeat']['accuracy'] for r in rr],label=label,color=color,ls=ls,marker=marker,lw=2,ms=5)
    style(axs[0,0],'A  Held-out text loss improves','Continuation step','Mean NLL (nats / target)')
    style(axs[0,1],'B  Copying retention deteriorates','Continuation step','Repeated-sequence accuracy (%)')
    axs[0,0].legend(fontsize=8.5,frameon=False,loc='upper right')
    axs[0,1].set_ylim(35,69)
    vals=[1.,checks[1]['cosine'],checks[2]['cosine']]
    bars=axs[1,0].barh([2,1,0],vals,color=[BLUE,TEAL,ORANGE],height=.55)
    axs[1,0].set_yticks([2,1,0],['Native Adam','Transported Adam','Coefficient Adam']);axs[1,0].set_xlim(0,1.13)
    for b,v in zip(bars,vals):axs[1,0].text(v+.018,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=10)
    style(axs[1,0],'C  Same first-step norm, different direction','Cosine with native Adam update','')
    axs[1,0].set_ylim(-.7,2.6)
    axs[1,0].text(.03,-.57,'All three update norms ≈ 0.08697',fontsize=9,color=GRAY)
    ax=axs[1,1]
    eq=[r for r in read('pythia_equivalence.json') if r['arm']=='curve_transport_adamw']
    stable=read('stable_sgd_equivalence.json')
    for rr,label,color,ls in [(eq,'Transported Adam',TEAL,'--'),(stable,'Stable-step SGD',PURPLE,'-')]:
        ax.plot([r['step'] for r in rr],[r['logit_rms_difference'] for r in rr],marker='o',ms=4,color=color,ls=ls,label=label)
    style(ax,'D  Equivalent updates accumulate numerical drift','Continuation step','Logit RMS difference vs native')
    ax.legend(frameon=False,fontsize=9,loc='upper left');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    fig.get_layout_engine().set(h_pad=.13,w_pad=.13,hspace=.10,wspace=.08)
    save(fig,'pythia_curve_continuation')

def toy():
    diagnostics=read('toy_conversion_diagnostics.json');rows=read('toy_summary.json')
    zero=read('toy_posthoc_zero_residual.json')['rows']
    fig,axs=plt.subplots(1,3,figsize=(14.2,4.5),layout='constrained')
    fig.suptitle('Shared depth curves: finite-task success, with a substantial ceiling limitation',fontsize=15,fontweight='bold')
    rms=[r['curve2_hard']['initial_logit_rms_error'] for r in diagnostics]
    axs[0].bar(range(5),rms,color=ORANGE,width=.62)
    style(axs[0],'A  Hard projection changes confidence','Paired seed','Immediate logit RMS error')
    axs[0].set_xticks(range(5));axs[0].set_ylim(0,2.45)
    axs[0].text(.03,.94,'Accuracy stays 100% in every seed',transform=axs[0].transAxes,va='top',fontsize=9,color=GRAY)
    configs=[('native6','Native six stages',BLUE),('curve6','Full six-mode curve',PURPLE),
        ('curve2','Two-mode curve',ORANGE),('curve2_multidepth','Two-mode, multi-depth',TEAL)]
    for arm,label,col in configs:
        rr=[r for r in rows if r['arm']==arm and r['step']==1200]
        axs[1].plot([r['depth'] for r in rr],[r['logit_rms_from_depth6']['mean'] for r in rr],color=col,marker='o',ms=4,label=label)
    style(axs[1],'B  Changing depth changes the function','Number of execution steps','Mean logit RMS vs six steps')
    axs[1].set_xticks([3,6,12,24]);axs[1].legend(frameon=False,fontsize=8)
    names=['Native six','Full curve','Two modes','Multi-depth']
    for i,(arm,_,col) in enumerate(configs):
        vv=np.array([r['accuracy']*100 for r in zero if r['arm']==arm]);mean=vv.mean()
        axs[2].scatter(np.full(5,i)+np.linspace(-.08,.08,5),vv,color=col,s=25,alpha=.8)
        axs[2].plot([i-.24,i+.24],[mean,mean],color=col,lw=3)
    axs[2].axhline(100,color=GRAY,ls='--',lw=1)
    axs[2].set_xticks(range(4),names,rotation=25,ha='right');axs[2].set_ylim(84,102)
    style(axs[2],'C  Input/output maps carry most labels','Trained input/output maps only','Accuracy after removing all stages (%)')
    fig.get_layout_engine().set(h_pad=.12,w_pad=.13,wspace=.07)
    save(fig,'toy_shared_depth_limits')

def stress():
    fig,axs=plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
    fig.suptitle('SGD: retain the unstable original run and distinguish the disclosed stability control',fontsize=12,fontweight='bold')
    for ax,file,title in [(axs[0],'pythia_evaluations.json','Original LR = 0.005'),
        (axs[1],'stable_sgd_evaluations.json','Added LR = 2.6043 × 10⁻⁶')]:
        rows=read(file)
        for arm,label,col,ls in [('native_sgd','Native',BLUE,'-'),('curve_sgd','Full Haar curve',ORANGE,'--')]:
            rr=[r for r in rows if r['arm']==arm]
            ax.plot([r['step'] for r in rr],[r['validation']['nll'] for r in rr],marker='o',color=col,ls=ls,label=label,ms=4)
        style(ax,title,'Continuation step','Held-out NLL (nats / target)');ax.legend(frameon=False,fontsize=9)
    save(fig,'sgd_stability_control')

if __name__=='__main__':pythia();toy();stress()
