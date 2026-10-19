"""Scientific figures for the frozen native-component experiment."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from causal_probe import ROOT,OUT,read,project

BLUE='#2563a6';GREEN='#168773';ORANGE='#cc6439';GRAY='#8c99a2';INK='#172637'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.labelcolor':INK,'text.color':INK,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def errors(d):
    p=d['estimate'];lo,hi=d['ci95'];return np.array([[max(0,p-lo)],[max(0,hi-p)]])

def overview():
    s=read('summary.json');e={(r['phase'],r['id']):r for r in read('confirmation_effects.json')}
    cases=read('confirmation.json');c={(r['phase'],r['id']):r for r in cases};leaf=s['selected_leaf']
    fig,ax=plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
    pairs=[('transplant','baseline'),('transplant','H6_root'),('rollback','baseline'),('rollback','H6_root')]
    labels=['Step 512','512 + interval\ntraining change','Step 1,000','1,000 − interval\ntraining change']
    xx=np.arange(4)
    for j,(kind,col,label) in enumerate([('repeat',BLUE,'Repeated sequence'),('scrambled',GREEN,'Scrambled queries')]):
        vals=[c[k][kind]['accuracy']*100 for k in pairs]
        ax[0,0].bar(xx+(j-.5)*.34,vals,.34,color=col,label=label)
        for x,v in zip(xx+(j-.5)*.34,vals):ax[0,0].text(x,v+.7,f'{v:.1f}',ha='center',fontsize=8)
    ax[0,0].set(xticks=xx,xticklabels=labels,ylabel='Successor-token accuracy (%)',title='A. Causal effects of one native Q/K interval')
    ax[0,0].set_ylim(0,max(c[k]['repeat']['accuracy']*100 for k in pairs)+13)
    ax[0,0].legend(loc='upper left',frameon=False,fontsize=9);ax[0,0].tick_params(axis='x',labelsize=8)
    cats=read('component_metadata.json');path=next(x['path'] for x in cats if x['id']==leaf)
    for x in cats:
        if x['role'] not in ['local_packet','adaptive_local_packet']:continue
        r=e[('rollback',x['id'])]['repeat']['target_nll_delta'];depth=len(x['path']);onpath=path.startswith(x['path'])
        ax[0,1].errorbar(depth+(.04 if onpath else -.04),r['estimate'],yerr=errors(r),fmt='o',ms=5 if onpath else 4,color=BLUE if onpath else GRAY,capsize=2,alpha=1 if onpath else .7)
    vals=[e[('rollback','H6_root' if k==0 else 'H6_'+path[:k])]['repeat']['target_nll_delta']['estimate'] for k in range(len(path)+1)]
    ax[0,1].plot(range(len(path)+1),vals,'-',color=BLUE,lw=1,alpha=.6)
    ax[0,1].set(xticks=range(7),xlabel='Wavelet-packet recursion depth',ylabel='Repeated-query NLL increase (nats)',title='B. Effects persist along the locked descent')
    ax[0,1].set_yscale('log');ax[0,1].set_ylim(.025,6);ax[0,1].grid(axis='y',alpha=.15)
    ax[0,1].text(.03,.04,'Blue: discovery-selected path · gray: tested siblings\nBars: 95% paired bootstrap intervals on fresh sequences',transform=ax[0,1].transAxes,fontsize=8)
    ids=['H6_root','H6_query','H6_key',leaf,'H5_root','H7_root','L3_fourier_atom','L3_haar_j6']
    names=['Head 6: Q/K interval','Head 6: Q interval','Head 6: K interval','Depth-6 component group','Head 5: Q/K control','Head 7: Q/K control','Prior Fourier mode','Prior Haar component']
    for col,kind in [(0,'repeat'),(1,'prose')]:
        a=ax[1,col]
        for yi,cid in enumerate(ids):
            r=e[('rollback',cid)][kind]['target_nll_delta' if kind=='repeat' else 'nll_delta']
            a.errorbar(r['estimate'],yi,xerr=errors(r),fmt='o',color=BLUE if col==0 else ORANGE,capsize=3,ms=5)
        a.axvline(0,color=GRAY,lw=1);a.set(yticks=range(len(ids)),yticklabels=names if col==0 else [],ylim=(len(ids)-.5,-.5),
          xlabel='NLL change (nats)',title='C. Copying effect by native component' if col==0 else 'D. Collateral effect on eight prose passages')
        a.grid(axis='x',alpha=.15)
    fig.suptitle('Pythia-70M: native component changes have measurable behavioral effects',fontsize=16)
    fig.supxlabel('Confirmation: 48 prefixes with disjoint token identities and lengths 16, 32 and 48. No training or weight reordering.',fontsize=9)
    save(fig,'pythia_component_causality')

def native_curves():
    s=read('summary.json');leaf=s['selected_leaf'];cats={c['id']:c for c in read('candidate_catalog.json')}
    tr=next(t for t in read('sibling_triples.json') if leaf in [t['left'],t['right']]);ids=[tr['parent'],tr['left'],tr['right']]
    z=np.load(ROOT/'native_qkv_curves.npz');delta=[z[f'step1000_layer{l}'].astype(np.float64)-z[f'step512_layer{l}'] for l in range(6)]
    lo,hi=589824,655360;before=z['step512_layer3'].ravel()[lo:hi];after=z['step1000_layer3'].ravel()[lo:hi]
    curves={cid:project(cats[cid],delta).ravel()[lo:hi] for cid in ids}
    np.savez_compressed(ROOT/'selected_local_component_curves.npz',native_indices=np.arange(lo,hi),before=before,after=after,full_interval_update=after.astype(np.float64)-before,
        **curves,metadata_json=np.array(json.dumps(dict(interval=[512,1000],support=[lo,hi],parent=tr['parent'],children=[tr['left'],tr['right']],selected_leaf=leaf))))
    fig,ax=plt.subplots(3,1,figsize=(13,7.7),sharex=True,layout='constrained');x=np.arange(lo,lo+256)
    ax[0].plot(x,before[:256],color=BLUE,lw=1,label='Step 512');ax[0].plot(x,after[:256],color=GREEN,lw=1,label='Step 1,000')
    ax[0].set(ylabel='Native weight',title='A. Exact original parameter curves, first 256 entries of the Q/K window')
    ax[0].legend(frameon=False,ncol=2,fontsize=9)
    ax[1].plot(x,(after.astype(np.float64)-before)[:256],color=GRAY,lw=1,label='Full interval training update')
    ax[1].plot(x,curves[tr['parent']][:256],color=ORANGE,lw=2,label='Depth-5 parent component')
    ax[1].set(ylabel='Weight change',title='B. One recursive component of the same full-resolution update');ax[1].legend(frameon=False,ncol=2,fontsize=9)
    for cid,col in [(tr['left'],BLUE),(tr['right'],GREEN)]:ax[2].plot(x,curves[cid][:256],color=col,lw=2,label=cid.replace('H6_','Branch ')+(' · selected' if cid==leaf else ''))
    ax[2].plot(x,(curves[tr['left']]+curves[tr['right']])[:256],'--',color=ORANGE,lw=1.3,label='Children summed = parent')
    ax[2].set(ylabel='Component change',xlabel='Original flat QKV weight index',title='C. Two depth-6 children reconstruct their parent exactly')
    ax[2].legend(frameon=True,facecolor='white',edgecolor='white',framealpha=.95,ncol=3,fontsize=9)
    for a in ax:a.grid(axis='y',alpha=.15);a.ticklabel_format(style='plain',axis='x',useOffset=False)
    fig.suptitle('The behavioral intervention is a mask on the existing native curve hierarchy',fontsize=15)
    fig.supxlabel('Every native sample is shown in this fixed illustrative window. Saved arrays contain all 65,536 positions.',fontsize=9)
    save(fig,'pythia_native_component_curves')

def interactions_and_later():
    s=read('summary.json');leaf=s['selected_leaf'];rows=read('behavioral_interactions.json')
    fig,ax=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    wanted=[('L3_all','packet','Whole tensor:\npacket split'),('L3_all','fourier','Whole tensor:\nFourier split'),('H6_root','packet','Q/K window:\npacket split'),('H6_root','native_spatial_split','Q/K window:\nQ versus K')]
    for j,(phase,col,label) in enumerate([('rollback',BLUE,'Rollback at step 1,000'),('transplant',ORANGE,'Transplant at step 512')]):
        vals=[100*next(r['interaction_fraction'] for r in rows if r['phase']==phase and r['probe']=='repeat' and r['parent']==par and r['kind']==kind) for par,kind,_ in wanted]
        ax[0].bar(np.arange(4)+(j-.5)*.34,vals,.34,color=col,label=label)
    ax[0].set(xticks=range(4),xticklabels=[x[2] for x in wanted],ylabel='Logit interaction / sum of child-effect RMS (%)',title='Parent and child effects are not additive')
    ax[0].legend(frameon=False,fontsize=8);ax[0].tick_params(axis='x',labelsize=8)
    late={r['id']:r for r in read('later_context.json')};ids=['baseline','H6_root',leaf,'L3_fourier_atom','L3_haar_j6']
    labels=['Step 8,000','Q/K interval\nrolled back','Depth-6 group\nrolled back','Fourier mode\nrolled back','Haar atom\nrolled back']
    for j,(kind,col,label) in enumerate([('repeat',BLUE,'Repeated sequence'),('scrambled',GREEN,'Scrambled queries')]):
        vals=[late[cid][kind]['accuracy']*100 for cid in ids]
        ax[1].bar(np.arange(5)+(j-.5)*.34,vals,.34,color=col,label=label)
    ax[1].set(xticks=range(5),xticklabels=labels,ylabel='Successor-token accuracy (%)',title='The same native addresses in a later network')
    ax[1].legend(frameon=False,fontsize=8);ax[1].set_ylim(0,100);ax[1].tick_params(axis='x',labelsize=8)
    fig.supxlabel('Interactions use all-vocabulary centered logits at the final query. Later rollbacks restore the selected coefficients to step 512.',fontsize=9)
    save(fig,'pythia_interactions_later_context')

if __name__=='__main__':overview();native_curves();interactions_and_later()
