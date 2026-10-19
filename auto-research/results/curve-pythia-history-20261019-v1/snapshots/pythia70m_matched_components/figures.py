"""Standalone scientific figures; exact curves remain available in NPZ."""
import itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text())
C={'early':'#196caa','late':'#d26730','repeat':'#196caa','scrambled':'#ae4c89'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','savefig.facecolor':'white'})
def find(rows,step,path='aaaaad',direction='early',lam=1):
    return next(r for r in rows if r['step']==step and r['path']==path and r['direction']==direction and abs(r['lam']-lam)<1e-7)
def error(ax,x,stat,**kw):
    y=stat['estimate'];lo,hi=stat['ci95'];ax.errorbar(x,y,yerr=np.array([[y-lo],[hi-y]]),capsize=3,**kw)
def main():
    se=read('scan_effects.json');ce=read('confirmation_effects.json');sel=read('selection_locked.json');meta=read('curve_metadata.json')
    paths=[''.join(p) for p in itertools.product('ad',repeat=6)];short=[''.join(p) for p in itertools.product('ad',repeat=3)]
    fig,axs=plt.subplots(2,2,figsize=(14,11),gridspec_kw={'height_ratios':[1.05,1]})
    vmax=max(abs(r[k]['target_nll_delta']['estimate']) for r in se for k in ['repeat','scrambled'])
    for ax,kind,title in zip(axs[0],['repeat','scrambled'],['A  Repeat queries: all 64 equal-norm branches','B  Scrambled queries: the same branches']):
        a=np.array([next(r for r in se if r['path']==p)[kind]['target_nll_delta']['estimate'] for p in paths]).reshape(8,8)
        im=ax.imshow(a,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set(xticks=range(8),xticklabels=short,yticks=range(8),yticklabels=short,xlabel='Final three packet decisions',ylabel='First three packet decisions',title=title)
        for path,color in [('aaaaad','#132635'),(sel['scan_winner'],'#f0aa13')]:
            i=paths.index(path);ax.add_patch(Rectangle((i%8-.48,i//8-.48),.96,.96,fill=False,edgecolor=color,lw=2.5))
        fig.colorbar(im,ax=ax,shrink=.8,label='Δ target NLL (nats; + means worse)')
    ax=axs[1,0]
    for direction in ['early','late']:
        for kind,offset,marker in [('repeat',-.035,'o'),('scrambled',.035,'s')]:
            yy=[]
            for i,step in enumerate([512,1000,8000]):
                r=find(ce,step,direction=direction);st=r[kind]['target_nll_delta'];yy.append(st['estimate'])
                error(ax,i+offset,st,color=C[direction],marker=marker,ls='none',alpha=.9)
            ax.plot(np.arange(3)+offset,yy,color=C[direction],ls='-' if kind=='repeat' else '--',alpha=.8,label=direction.capitalize()+' direction / '+kind)
    ax.axhline(0,color='#89949e',lw=.8);ax.set(xticks=range(3),xticklabels=['512','1,000','8,000'],xlabel='Background checkpoint',ylabel='Δ target NLL (nats)',title='C  Identical vectors across checkpoints')
    ax.legend(fontsize=9,loc='best');ax.grid(axis='y',alpha=.18)
    ax=axs[1,1]
    for i,path in enumerate(sel['paths']):
        r=find(ce,1000,path=path)
        for kind,offset,marker in [('repeat',-.1,'o'),('scrambled',.1,'s')]:
            st=r[kind]['target_nll_delta'];y=st['estimate'];lo,hi=st['ci95']
            ax.errorbar(y,i+offset,xerr=[[y-lo],[hi-y]],fmt=marker,color=C[kind],capsize=3,label=kind.capitalize() if i==0 else None)
    labels=[p+('  prior selected' if p=='aaaaad' else '  scan winner' if p==sel['scan_winner'] else '  sibling' if p=='aaaaaa' else '') for p in sel['paths']]
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Δ target NLL (nats)',title='D  Independent confirmation at step 1,000');ax.invert_yaxis();ax.axvline(0,color='#89949e',lw=.8);ax.grid(axis='x',alpha=.18);ax.legend(fontsize=9)
    fig.suptitle('Pythia-70M · native component structure and checkpoint context',fontsize=18,y=.99)
    fig.text(.5,.025,'Each unit intervention has L2 norm %.4f and retains all 1,024 coefficients in its depth-six group.\nBlack outline: prior branch; gold: scan winner. Top: 24 scan sequences. Bottom: 48 fresh sequences, paired 95%% intervals.'%meta['common_norm'],ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(OUT/'pythia_matched_components.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for j,step in enumerate([512,1000,8000]):
        for i,kind in enumerate(['repeat','scrambled']):
            ax=axs[i,j]
            for direction in ['early','late']:
                rr=sorted([r for r in ce if r['step']==step and r['path']=='aaaaad' and r['direction']==direction],key=lambda r:r['lam'])
                data=[(r['lam'],r[kind]['target_nll_delta']) for r in rr]+[(0,dict(estimate=0,ci95=[0,0]))];data.sort()
                x=np.array([v[0] for v in data]);y=np.array([v[1]['estimate'] for v in data]);lo=np.array([v[1]['ci95'][0] for v in data]);hi=np.array([v[1]['ci95'][1] for v in data])
                ax.plot(x,y,'o-',color=C[direction],label=direction.capitalize()+' direction');ax.fill_between(x,lo,hi,color=C[direction],alpha=.14)
            ax.axhline(0,color='#a0a9af',lw=.8);ax.axvline(0,color='#a0a9af',lw=.8);ax.grid(alpha=.14)
            ax.set_title(f'Step {step:,} · {kind}')
            if j==0:ax.set_ylabel('Δ target NLL (nats)')
            if i==1:ax.set_xlabel('λ: negative adds; positive subtracts')
            if i==0 and j==0:ax.legend(fontsize=9)
    fig.suptitle('Dose response of the fixed aaaaad component group',fontsize=18)
    fig.text(.5,.015,'θ′ = θ − λv. Both directions share the same norm. λ = %.3f matches the natural late update magnitude.\nShading: paired 95%% intervals over 48 sequences; each panel uses its own vertical scale.'%meta['late_to_early_norm_ratio'],ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(OUT/'pythia_component_dose_response.png',dpi=170);plt.close(fig)
    z=np.load(ROOT/'native_local_curves.npz');e=z['selected_early'];l=z['selected_late_matched'];ec=z['early_coefficients'][paths.index('aaaaad')];lc=z['late_coefficients'][paths.index('aaaaad')]/meta['late_to_early_norm_ratio']
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    ax=axs[0,0];x=np.arange(len(ec));ax.plot(x,ec,lw=.75,color=C['early'],label='Early');ax.plot(x,lc,lw=.75,color=C['late'],alpha=.7,label='Late, norm matched');ax.set(title='A  All 1,024 selected coefficients',xlabel='Coefficient index in native order',ylabel='Coefficient value');ax.legend(fontsize=9)
    ax=axs[0,1];ax.scatter(ec,lc,s=10,alpha=.55,color='#536f83');ax.set(title='B  Coefficient orientation changes',xlabel='Early coefficient',ylabel='Late coefficient, norm matched');ax.axhline(0,lw=.6,color='#aaa');ax.axvline(0,lw=.6,color='#aaa');ax.text(.03,.94,'Cosine = %.3f · angle = %.1f°'%(meta['selected_early_late_cosine'],meta['selected_angle_degrees']),transform=ax.transAxes,va='top',fontsize=10)
    ax=axs[1,0];x=z['native_flat_indices'][:512];ax.step(x,e[:512],where='mid',color=C['early'],label='Early');ax.step(x,l[:512],where='mid',color=C['late'],label='Late, norm matched');ax.set(title='C  Native curve detail: first 512 positions',xlabel='Original QKV flat coordinate',ylabel='Reconstructed update');ax.ticklabel_format(style='plain',useOffset=False,axis='x');ax.legend(fontsize=9)
    ax=axs[1,1];norm=np.array([meta['early_norms'][p] for p in paths]);ax.bar(range(64),norm,color='#aebfcc');i=paths.index('aaaaad');ax.bar(i,norm[i],color=C['early']);ax.axhline(meta['common_norm'],color=C['late'],ls='--',label='Common intervention norm');ax.set(title='D  Original update sizes before matching',xlabel='Depth-six path in binary tree order',ylabel='Natural update L2 norm');ax.legend(fontsize=9)
    fig.suptitle('Native coordinates and full coefficient resolution',fontsize=18)
    fig.text(.5,.015,'The NPZ includes every weight in the native window at all three checkpoints and every depth-six coefficient.\nPanel C is a labeled visual crop only; no analysis or intervention uses downsampled curves.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(OUT/'pythia_native_matched_curves.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
