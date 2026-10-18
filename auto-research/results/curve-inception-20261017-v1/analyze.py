"""Replay frozen endpoints, perform registered interventions, and plot all seeds."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ARMS, CurveCell
from tasks import episode
from experiment import ROOT, save, sha, evaluate

LABELS={'fixed48':'Fixed curves · 48 state', 'wide64':'Wider fixed · 64 state',
        'context64':'Ordinary context · 64 state', 'adaptive64':'Adaptive curves · 64 state',
        'gru64':'Standard GRU · 64 state'}
COLORS={'fixed48':'#9b9b9b','wide64':'#6589ad','context64':'#e49a35','adaptive64':'#008b80','gru64':'#8a5da8'}

def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    reg=json.loads((ROOT/'REGISTRATION.json').read_text())
    assert sha(ROOT/'PROTOCOL.json')==reg['protocol_sha256']
    for f,s in protocol['evidence']['source_manifest'].items():assert sha(ROOT/f)==s
    seeds=protocol['design']['seeds'];data={};terminals=[];manifest={};replays=[];interventions=[]
    start=time.perf_counter()
    for seed in seeds:
        hashes=[];counts=[]
        for arm in ARMS:
            d=ROOT/'main'/f'{arm}_s{seed}'
            terminal=json.loads((d/'TERMINAL.json').read_text());terminals.append(terminal)
            assert terminal['status']=='COMPLETE' and terminal['completed_steps']==protocol['training']['updates']
            assert sha(d/'final.pt')==terminal['checkpoint_sha256']
            manifest[str((d/'final.pt').relative_to(ROOT))]=sha(d/'final.pt')
            hashes.append(terminal['training_stream_sha256']);counts.append((terminal['targets'],terminal['input_tokens']))
            data[arm,seed]=json.loads((d/'scores.json').read_text())
            m=CurveCell(arm,seed);m.load_state_dict(torch.load(d/'final.pt',weights_only=True))
            repeat=evaluate(m,[990001,990002],(256,))
            saved=[r for r in data[arm,seed] if r['delay']==256]
            assert repeat==saved, (arm,seed,'replay drift')
            replays.append(dict(arm=arm,seed=seed,exact=True))
            print('Replay',arm,seed,'PASS',flush=True)
            if arm=='adaptive64':
                for intervention in ('zero_p','no_gate','no_mix'):
                    rows=evaluate(m,[990001,990002],(256,),intervention=intervention)
                    interventions.append(dict(arm=arm,seed=seed,intervention=intervention,scores=rows))
        assert len(set(hashes))==1 and len(set(counts))==1
    means=[]
    for arm in ARMS:
        for mode in ('retention','conflict','reopen'):
            for delay in (32,256,1024):
                rows=[next(r for r in data[arm,s] if r['mode']==mode and r['delay']==delay) for s in seeds]
                vals=[r['accuracy'] for r in rows]
                means.append(dict(arm=arm,mode=mode,delay=delay,mean_accuracy=float(np.mean(vals)),
                    per_seed_accuracy=vals,mean_locked_accuracy=float(np.mean([r['locked_accuracy'] for r in rows])),
                    mean_unlocked_accuracy=float(np.mean([r['unlocked_accuracy'] for r in rows])),
                    mean_reopened_accuracy=float(np.mean([r['reopened_accuracy'] for r in rows])) if mode=='reopen' else None))
    def row(arm,mode='conflict',delay=256):return next(r for r in means if r['arm']==arm and r['mode']==mode and r['delay']==delay)
    paired=np.array(row('adaptive64')['per_seed_accuracy'])-row('context64')['per_seed_accuracy']
    reopen_gap=row('adaptive64','reopen',32)['mean_reopened_accuracy']-row('context64','reopen',32)['mean_reopened_accuracy']
    passed=bool(paired.mean()>=.05 and (paired>0).all() and reopen_gap>=-.02)
    summary=dict(primary=dict(paired_seed_differences=paired.tolist(),mean_difference=float(paired.mean()),
                              min_difference=float(paired.min()),max_difference=float(paired.max()),
                              short_reopened_accuracy_gap=reopen_gap,gate_passed=passed),
                 aggregates=means,training_cpu_seconds=sum(r['training_cpu_seconds'] for r in terminals),
                 training_wall_seconds_sum=sum(r['training_wall_seconds'] for r in terminals),
                 counts=protocol['design']['parameter_counts'],trials=len(terminals),
                 updates_per_trial=protocol['training']['updates'],
                 status='FOLLOWUP_SIGNAL' if passed else 'NO_PROMOTION',
                 all_outcomes='All15 final checkpoints included;250-step pilot excluded from main summaries.')
    save(ROOT/'SUMMARY.json',summary);save(ROOT/'INTERVENTIONS.json',interventions)
    save(ROOT/'CHECKPOINT_MANIFEST.json',manifest)
    save(ROOT/'REPLAY_CHECKS.json',dict(replays=replays,paired_data_counts_and_hashes_match=True,
                                       protocol_source_hashes_match=True,analysis_seconds=time.perf_counter()-start))
    graphs=ROOT.parents[1]/'graphs'/ROOT.name;graphs.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9'})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),sharey=True)
    for ax,mode,title in zip(axes,('retention','conflict','reopen'),('No conflicting writes','Conflicting writes','Reopen and replace')):
        for arm in ARMS:
            rs=[row(arm,mode,d) for d in (32,256,1024)]
            y=np.array([r['mean_accuracy'] for r in rs])*100
            lo=np.array([min(r['per_seed_accuracy']) for r in rs])*100
            hi=np.array([max(r['per_seed_accuracy']) for r in rs])*100
            ax.plot([32,256,1024],y,'o-',color=COLORS[arm],label=LABELS[arm],linewidth=2 if arm=='adaptive64' else 1.5)
            ax.fill_between([32,256,1024],lo,hi,color=COLORS[arm],alpha=.09)
        ax.axhline(12.5,color='#aaa',linestyle=':',linewidth=1)
        ax.set_xscale('log',base=2);ax.set_xticks([32,256,1024],[32,256,1024]);ax.set_ylim(0,102)
        ax.set_title(title);ax.set_xlabel('Intervening tokens');ax.grid(alpha=.18)
    axes[0].set_ylabel('Final-query accuracy (%)')
    fig.suptitle('Do persistent parameter curves improve retention?',fontsize=17,x=.06,ha='left')
    fig.text(.06,.89,'Synthetic locked registers · 3 training seeds · 3,000 updates each · shading shows seed range',fontsize=10,color='#555')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.subplots_adjust(top=.80,bottom=.23,wspace=.12,left=.06,right=.98)
    fig.savefig(graphs/'retention.png',dpi=180,bbox_inches='tight');fig.savefig(graphs/'retention.svg',bbox_inches='tight');plt.close(fig)
    # Predetermined trajectory: first main seed, new illustrative stream, no selection for success.
    m=CurveCell('adaptive64',seeds[0]);m.load_state_dict(torch.load(ROOT/'main'/f'adaptive64_s{seeds[0]}'/'final.pt',weights_only=True));m.eval()
    x,y,meta=episode(1234567,1,64,'reopen')
    with torch.no_grad():out,state,h,p=m(x,trace=True)
    fields=p[0].reshape(-1,2,8).numpy()@m.basis.numpy().T
    np.savez_compressed(ROOT/'curve_trace.npz',tokens=x.numpy(),targets=y.numpy(),predictions=out.argmax(-1).numpy(),h=h.numpy(),p=p.numpy(),fields=fields)
    fig,ax=plt.subplots(2,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.6,1]})
    for channel,name in enumerate(('Gate-logit coefficient field','Mixing coefficient field')):
        mat=fields[:,channel].T;vmax=max(float(abs(mat).max()),1e-5)
        im=ax[channel,0].imshow(mat,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax,extent=[1,len(x[0]),0,1])
        ax[channel,0].set_ylabel('Curve coordinate');ax[channel,0].set_title(name);fig.colorbar(im,ax=ax[channel,0],fraction=.025)
        s=(np.arange(24)+.5)/24
        for t in (7,15,31,63,len(x[0])-1):ax[channel,1].plot(s,fields[t,channel],label=f't={t+1}')
        ax[channel,1].set_title('Sampled curve shapes');ax[channel,1].set_xlabel('Curve coordinate');ax[channel,1].grid(alpha=.15)
    ax[1,0].set_xlabel('Input token');ax[0,1].legend(ncol=3,fontsize=8,frameon=False)
    fig.suptitle('A writable curve controls subsequent memory updates',fontsize=16)
    fig.text(.08,.01,'Illustrative stream fixed in advance. These are learned gate/mixing offsets, not the entire model or a depth coordinate.',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(graphs/'curve_evolution.png',dpi=160);plt.close(fig)
    print(json.dumps(summary['primary'],indent=2),flush=True)

if __name__=='__main__':main()
