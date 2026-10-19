"""Measured retention, exact depth response and normalized Haar update spectra."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
read=lambda n:json.loads((ROOT/'results'/n).read_text())
BLUE='#1965b0';ORANGE='#d47219';TEAL='#078c83';PURPLE='#8257a6';GRAY='#657b89'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c8d3db','text.color':'#162735','axes.labelcolor':'#162735','xtick.color':'#43596a','ytick.color':'#43596a','grid.color':'#e2e9ee','svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
def style(ax,title,xlabel,ylabel):
    ax.set_title(title,loc='left',pad=12);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.7);ax.set_axisbelow(True)
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def retention():
    s=read('summary.json');final={r['arm']:r for r in s['final']};base=s['baseline'];fig,axs=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
    fig.suptitle('Pythia-70m: retention-aware updates improve the measured tradeoff',fontsize=15,fontweight='bold')
    for coord,color,marker in [('native',BLUE,'o'),('curve',ORANGE,'s')]:
        rr=[final[f'{coord}_lambda{v}'] for v in [0,1,4]];xx=[r['validation_nll'] for r in rr];yy=[100*r['repeat']['accuracy'] for r in rr]
        axs[0].plot(xx,yy,marker=marker,ms=7,color=color,lw=1.6,label='Native Adam' if coord=='native' else 'Coefficient Adam')
        for r,x,y,lam in zip(rr,xx,yy,[0,1,4]):
            offset=(7,-12) if coord=='native' else (7,7);axs[0].annotate('λ='+str(lam),(x,y),textcoords='offset points',xytext=offset,color=color,fontsize=9)
    axs[0].scatter([base['validation_nll']],[100*base['repeat']['accuracy']],marker='*',s=125,color=TEAL,label='Starting model',zorder=3)
    style(axs[0],'A  Lower loss, higher retention is better','Held-out text NLL (nats / target)','Held-out copy accuracy (%)');axs[0].legend(frameon=False,fontsize=9,loc='lower right');axs[0].set_xlim(4.46,4.96);axs[0].set_ylim(34,68)
    comparisons=[r for r in s['contrasts'] if r['b'].endswith('lambda0')]
    names=[];ys=[];es=[];colors=[]
    for coord in ['native','curve']:
        for lam in [1,4]:
            r=next(r for r in comparisons if r['a']==f'{coord}_lambda{lam}')['contrast']['repeat']['accuracy'];names.append(('Native' if coord=='native' else 'Curve')+f' λ={lam}');ys.append(r['estimate']);es.append([r['estimate']-r['ci95'][0],r['ci95'][1]-r['estimate']]);colors.append(BLUE if coord=='native' else ORANGE)
    axs[1].bar(range(4),ys,color=colors,width=.6);axs[1].errorbar(range(4),ys,yerr=np.array(es).T,fmt='none',ecolor='#233745',capsize=4)
    axs[1].set_xticks(range(4),names);style(axs[1],'B  Copying gains over each unregularized control','Same coordinate system, same 24 adaptation batches','Accuracy gain (percentage points)')
    axs[1].axhline(0,color=GRAY,lw=1);axs[1].set_ylim(0,25)
    save(fig,'retention_tradeoff')
def transport():
    rows=read('transport_summary.json');oracle=read('transport_oracle.json');limit=read('transport_continuous_limit.json')
    fig,axs=plt.subplots(2,2,figsize=(12.2,8.4),layout='constrained');fig.suptitle('Six-hop retrieval: an exact depth curve does not guarantee an equivalent computation at new depths',fontsize=14,fontweight='bold')
    chosen=[('native_fixed','Native teacher continued',BLUE),('curve2_fixed','Two-mode, fixed-depth',ORANGE),('curve2_multidepth','Two-mode, multi-depth',TEAL),('curve6_multidepth','Six-mode, multi-depth',PURPLE)]
    for arm,label,color in chosen:
        rr=[r for r in rows if r['arm']==arm and r['depth']>=6];axs[0,0].plot([r['depth'] for r in rr],[r['expected_mse']['mean'] for r in rr],color=color,label=label,marker='o',ms=4)
    axs[0,0].set_xticks([6,12,24,48]);style(axs[0,0],'A  More steps change retrieval','Execution steps','Expected MSE (unit-variance inputs)');axs[0,0].legend(frameon=False,fontsize=8,loc='lower right')
    for n,col,ls in [(6,BLUE,'-'),(12,ORANGE,'-'),(48,TEAL,'--')]:
        a=np.array(next(r['operator'] for r in oracle if r['depth']==n));axs[0,1].plot(range(12),a[0],color=col,ls=ls,marker='o',ms=4,label=f'{n} steps')
    style(axs[0,1],'B  Constant-curve oracle: routing becomes mixing','Output site after an impulse at site 0','Impulse response');axs[0,1].set_xticks(range(0,12,2));axs[0,1].legend(frameon=False,fontsize=9)
    for n,col,ls in [(6,BLUE,'-'),(12,ORANGE,'-'),(48,TEAL,'--')]:
        r=next(r for r in limit['finite_frequency_responses'] if r['depth']==n);axs[1,0].plot(range(7),r['magnitudes'][:7],color=col,ls=ls,marker='o',ms=4,label=f'{n} steps')
    style(axs[1,0],'C  Fine spatial modes are attenuated','Spatial Fourier mode on the 12-site ring','Operator gain magnitude');axs[1,0].legend(frameon=False,fontsize=9);axs[1,0].set_ylim(-.04,1.08)
    for i,(arm,name,color) in enumerate([('native6','Native six',BLUE),('curve6','Full six-mode',PURPLE),('curve2','Two-mode',ORANGE)]):
        r=next(r for r in rows if r['arm']==arm and r['depth']==6);vv=r['expected_mse']['values'];axs[1,1].scatter(np.full(5,i)+np.linspace(-.08,.08,5),vv,color=color,s=34)
    axs[1,1].set_yscale('log');axs[1,1].set_xticks(range(3),['Native six','Full six-mode','Two-mode']);style(axs[1,1],'D  Optimization differs despite exact capacity','From scratch; same Adam rate, five paired seeds','Expected six-step MSE')
    save(fig,'depth_routing_and_spectrum')
def wavelets():
    rows=read('haar_band_summary.json');fig,axs=plt.subplots(1,2,figsize=(12.2,4.6),layout='constrained');fig.suptitle('Native-order recursive Haar analysis: normalize by the number of coefficients',fontsize=14,fontweight='bold')
    for coord,color in [('native',BLUE),('curve',ORANGE)]:
        for lam,ls in [(0,'-'),(1,'--'),(4,':')]:
            rr=[r for r in rows if r['arm']==f'{coord}_lambda{lam}' and r['kind']=='detail'];rr=sorted(rr,key=lambda r:r['level'])
            axs[0].plot([r['level'] for r in rr],[r['enrichment'] for r in rr],ls=ls,color=color,label=f'{coord.capitalize()} λ={lam}',lw=1.8)
    axs[0].set_yscale('log');axs[0].axhline(1,color=GRAY,lw=.8);axs[0].set_xticks([1,4,8,12,16,20]);style(axs[0],'A  Mean coefficient energy relative to global mean','Haar support = 2^level native tensor entries','Enrichment (log scale)');axs[0].legend(frameon=False,fontsize=8,ncol=2)
    layer=read('haar_layer_band_summary.json');arr=np.full((6,20),np.nan)
    for r in layer:
        if r['arm']=='curve_lambda1' and r['kind']=='detail':arr[r['layer'],r['level']-1]=np.log2(r['enrichment'])
    vmax=max(1.,float(np.nanmax(np.abs(arr))));im=axs[1].imshow(arr,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,origin='upper')
    axs[1].set_xticks([0,3,7,11,15,19],[1,4,8,12,16,20]);axs[1].set_yticks(range(6));axs[1].set_title('B  Coefficient Adam λ=1: layer and scale',loc='left',pad=12);axs[1].set_xlabel('Haar detail level');axs[1].set_ylabel('Original decoder layer');fig.colorbar(im,ax=axs[1],label='log₂ enrichment',shrink=.85)
    save(fig,'haar_update_structure')
if __name__=='__main__':retention();transport();wavelets()
