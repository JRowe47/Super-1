"""Standalone scientific figures from saved measurements only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/'results'/n).read_text())
C={'full':'#2766ad','targeted':'#078c83','random0':'#8b97a1','random1':'#a7afb7','random_mean':'#7b8793','opposite':'#c46538'}
LABEL={'full':'All coefficients','targeted':'Targeted mask','random0':'Random mask 1','random1':'Random mask 2','random_mean':'Random masks (mean)','opposite':'Opposite mask'}
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263c4c','text.color':'#263c4c','axes.edgecolor':'#aab7c0','xtick.color':'#536b7b','ytick.color':'#536b7b','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def main():
    style();e=read('effects.json');base=e[0];contrast=read('contrasts.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
    for arm in ['full','targeted','random_mean','opposite']:
        rows=[base]+sorted([r for r in e if r['arm']==arm],key=lambda r:r['step']);xs=[r['step'] for r in rows]
        axes[0].plot(xs,[100*r['repeat']['accuracy'] for r in rows],'-o',color=C[arm],label=LABEL[arm],markersize=4)
        axes[1].plot(xs,[r['validation_nll'] for r in rows],'-o',color=C[arm],markersize=4)
    arms=['full','targeted','random0','random1','opposite']
    for i,arm in enumerate(arms):
        r=next(r for r in e if r['arm']==arm and r['step']==24)
        axes[2].scatter(r['validation_nll'],100*r['repeat']['accuracy'],s=55,color=C[arm],zorder=4)
        offset={'full':(6,-13),'targeted':(6,8),'random0':(-96,-16),'random1':(6,6),'opposite':(6,-13)}[arm]
        axes[2].annotate(LABEL[arm],(r['validation_nll'],100*r['repeat']['accuracy']),xytext=offset,textcoords='offset points',fontsize=8.5,color=C[arm])
    axes[0].set(xlabel='Training update',ylabel='Held-out copying accuracy (%)',title='A  Copying retention');axes[0].legend(fontsize=8.5,loc='lower left')
    axes[1].set(xlabel='Training update',ylabel='Held-out text NLL (lower is better)',title='B  Text adaptation')
    axes[2].set(xlabel='Held-out text NLL (lower is better)',ylabel='Copying accuracy (%)',title='C  Final tradeoff');axes[2].margins(x=.4,y=.25)
    for ax in axes:ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Haar update targeting · masked runs match band counts; all runs match step norms',fontsize=14,fontweight='bold')
    fig.supxlabel('120 Pythia updates total • 48 fresh text blocks + 48 short copy sequences • fixed training stream',fontsize=9,color='#536b7b')
    save(fig,'masked_update_tradeoff')

    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    pred=read('mask_calibration.json')['predicted'];arms=['full','targeted','random0','random1','opposite'];xx=np.arange(5)
    axes[0].bar(xx,[pred[a]['predicted_copy_delta_at_lr2e5'] for a in arms],color=[C[a] for a in arms]);axes[0].axhline(0,color='#536b7b',lw=.8)
    axes[0].set_xticks(xx,[LABEL[a].replace(' ','\n',1) for a in arms],fontsize=8);axes[0].set(ylabel='Predicted copying NLL change',title='A  Calibration prediction')
    for arm in arms:
        r=next(r for r in e if r['arm']==arm and r['step']==1);v=r['changes']['repeat']['target_nll'];i=arms.index(arm)
        axes[1].bar(i,v['estimate'],color=C[arm]);axes[1].errorbar(i,v['estimate'],yerr=np.array([[v['estimate']-v['ci95'][0]],[v['ci95'][1]-v['estimate']]]),fmt='none',ecolor='#263c4c',capsize=3)
    axes[1].axhline(0,color='#536b7b',lw=.8);axes[1].set_xticks(xx,[LABEL[a].replace(' ','\n',1) for a in arms],fontsize=8);axes[1].set(ylabel='Held-out copying NLL change',title='B  Observed first update')
    bands=read('haar_band_summary.json')
    for arm in arms:
        rows=[r for r in bands if r['arm']==arm];axes[2].plot([r['level'] for r in rows],[r['enrichment'] for r in rows],'-o',color=C[arm],markersize=2.5,label=LABEL[arm])
    axes[2].axhline(1,color='#536b7b',lw=.8,ls='--');axes[2].set(xlabel='Haar detail level j (support = 2ʲ weights)',ylabel='Mean update² / global mean update²',title='C  Full-resolution final updates',yscale='log');axes[2].legend(fontsize=7.5)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Local gradient scores tested by intervention',fontsize=14,fontweight='bold')
    fig.supxlabel('Panel A uses calibration mean gradients; B uses the first training batch and fresh repeated-copy tests. Error bars: paired 95% bootstrap intervals.',fontsize=9,color='#536b7b')
    save(fig,'recursive_targeting_diagnostics')

    oracle=read('transport_oracle.json');toy=read('transport_evaluations.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
    grids=[1,6,12,17,24,31,48,96];xs=np.arange(len(grids))
    for solver,label,color in [('exact','Boundary-aware exact rotations','#078c83'),('midpoint','Uniform midpoint rotations','#2766ad'),('euler','Boundary-aware Euler','#c46538')]:
        r=[next(r for r in oracle if r['solver']==solver and r['grid']==g) for g in grids]
        axes[0].plot(xs,[max(1e-28,d['expected_mse']) for d in r],'-o',label=label,color=color,markersize=4)
    axes[0].set(yscale='log',ylabel='Expected output MSE (display floor 10⁻²⁸)',xlabel='Requested integration grid',title='A  Operator and solver both matter');axes[0].set_xticks(xs,grids);axes[0].legend(fontsize=7.5,loc='center right');axes[0].grid(alpha=.18)
    traces=read('transport_training.json')
    for arm,label,color in [('full12','12 pulse controls','#2766ad'),('shared2','2 shared controls','#078c83')]:
        steps=[1,100,200,400];values=np.array([[r['training_loss'] for r in traces if r['arm']==arm and r['step']==step] for step in steps])
        axes[1].plot(steps,values.mean(1),'-o',markersize=4,color=color,label=label)
        axes[1].fill_between(steps,values.min(1),values.max(1),color=color,alpha=.15)
    axes[1].set(yscale='log',ylabel='Training stage loss (mean and seed range)',xlabel='Training update',title='B  Both parameterizations learn');axes[1].legend(fontsize=8);axes[1].grid(alpha=.18)
    exact=[r for r in oracle if r['solver']=='exact'];axes[2].plot(xs,[r['rotation_operations'] for r in exact],'-o',color='#078c83',label='Executed local rotations');axes[2].plot(xs,grids,'--',color='#8b97a1',label='Requested grid count');axes[2].set_xticks(xs,grids)
    axes[2].set(xlabel='Requested integration grid',ylabel='Operations',title='C  Boundary handling has a cost');axes[2].legend(fontsize=8);axes[2].grid(alpha=.18)
    fig.suptitle('A real two-channel local flow removes the scalar routing obstruction',fontsize=14,fontweight='bold')
    fig.supxlabel('12 sites, six-hop target, fixed input/readout • state doubles from 12 to 24 scalars • at least 12 rotations • not a full GSA conversion',fontsize=9,color='#536b7b')
    save(fig,'augmented_depth_transport')
if __name__=='__main__':main()
