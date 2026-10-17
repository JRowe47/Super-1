"""Check observed clocks/targets, replay endpoints, and visualize every trial."""
import argparse,gzip,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import experiment as ex
import torch

@torch.no_grad()
def detailed_eval(m,q,data,ends,swap=False):
    """Replay the declared mean and add a post-hoc position-specific diagnostic.

    The first scored suffix token is adjacent to the changed prefix; only the
    final prediction places that entire prefix beyond its latest512 inputs.
    """
    m.eval();window=[];profiles=[];last=[];ctx=q['context']
    for i,end in enumerate(ends):
        x,y=ex.eval_batch(data,[end],ctx)
        if swap and ctx>512:
            donor,_=ex.eval_batch(data,[ends[(i+1)%len(ends)]],ctx)
            x[:,:ctx-512]=donor[:,:ctx-512]
        z,_=ex.parent.forward(m,q['arm'],x,y);z=z[:,-512:];y=y[:,-512:]
        window.append(float(ex.F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))))
        losses=ex.F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction='none')
        profiles.append(losses.reshape(8,64).mean(1).tolist());last.append(float(losses[-1]))
    return dict(window_ce=window,mean_ce=sum(window)/len(window),
        bins64_mean_ce=ex.np.mean(profiles,axis=0).tolist(),last_target_window_ce=last,last_target_mean_ce=sum(last)/len(last))

def label(q):
    family='nanoGPT' if q['arm']=='nanogpt' else 'GRU' if q['arm']=='native' else 'lin-minGRU'
    return f"{family} w{q['width']} / {q['context']}"+(' / chunk64' if q['segment_length']==64 else '')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);args=ap.parse_args()
    p,train,val,starts=ex.inputs(args);rows=[];checks={};failures=[];manifest=[];profiles={}
    composition=[]
    for end in p['final_ends']:
        prefix=val[end-2048:end];eos=ex.np.flatnonzero(prefix==0)
        composition.append(dict(final_target_index=end,
            preceding_non_EOS_inputs_capped_at2048=2048-int(eos[-1])-1 if len(eos) else 2048,
            EOS_in_latest512_inputs=bool(ex.np.any(prefix[-512:]==0))))
    ex.save(ROOT/'DATA_CONTEXT.json',dict(EOS_id=0,
        scope='Post-hoc composition check of existing16 final validation windows; no new evaluation region.',
        windows_without_EOS_in_latest512_inputs=sum(not w['EOS_in_latest512_inputs'] for w in composition),windows=composition))
    for q in p['runs']:
        out=ROOT/'execution'/q['run_id']
        if (out/'FAILURE.json').exists():failures.append(json.loads((out/'FAILURE.json').read_text()));continue
        r=json.loads((out/'RESULT.json').read_text());state=json.loads((out/'curves.json').read_text())
        assert r['protocol_sha256']==ex.sha(ROOT/'PROTOCOL.json') and r['checkpoint_sha256']==ex.sha(out/'checkpoint.pt')
        cp=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert cp['protocol_sha256']==r['protocol_sha256'] and cp['state']==state
        assert len(state['train'])==r['updates']==state['step']
        assert r['training_targets']==r['updates']*q['blocks_per_update']*2048==state['training_targets']
        for key,field in [('train_wall_seconds','wall'),('train_cpu_seconds','cpu')]:assert abs(sum(t[field] for t in state['train'])-r[key])<1e-6
        assert 0<r['train_wall_seconds']<=p['train_wall_seconds'];previous=0.
        for i,t in enumerate(state['train']):
            progress=previous/p['train_wall_seconds'];factor=max(.04,progress/.05) if progress<.05 else .1+.45*(1+math.cos(math.pi*(progress-.05)/.95))
            assert abs(t['lr']-q['peak_lr']*factor)<1e-12
            assert t['step']==i+1 and t['training_targets']==(i+1)*q['blocks_per_update']*2048
            assert math.isfinite(t['ce']) and math.isfinite(t['gradient_norm']);previous=t['train_wall_seconds']
        m=ex.build(q,p);m.load_state_dict(cp['model']);replay=detailed_eval(m,q,val,p['final_ends'])
        err=max(abs(a-b) for a,b in zip(replay['window_ce'],r['final']['window_ce']));assert err<5e-6,(q['run_id'],err)
        donor_error=None
        if q['context']>512:
            swapped=detailed_eval(m,q,val,p['final_ends'],True)
            donor_error=max(abs(a-b) for a,b in zip(swapped['window_ce'],r['prefix_swapped']['window_ce']));assert donor_error<5e-6
            profiles[q['run_id']]=dict(authentic=replay,swapped=swapped,
                bins64_benefit_ce=[b-a for a,b in zip(replay['bins64_mean_ce'],swapped['bins64_mean_ce'])],
                last_target_benefit_ce=swapped['last_target_mean_ce']-replay['last_target_mean_ce'])
        checks[q['run_id']]=dict(final_window_replay_max_error=err,prefix_swap_replay_max_error=donor_error,targets_clocks_lr_reconciled=True)
        keys=('run_id','arm','width','context','segment_length','parameters','batch','peak_lr','updates','training_targets',
              'train_wall_seconds','train_cpu_seconds','peak_rss_kib','prefix_benefit_ce','persistent_state_fp32_bytes_per_sequence')
        row={k:r[k] for k in keys}
        row.update(label=label(q),ce=r['final']['ce'],perplexity=r['final']['perplexity'],tokens_per_wall_second=r['training_targets']/r['train_wall_seconds'],
            parameters_reached_at_training_length=q['parameters']-(2048-q['context'])*q['width'] if q['arm']=='nanogpt' else q['parameters'],monitor=r['monitor'])
        rows.append(row)
        with (out/'curves.json.gz').open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write((out/'curves.json').read_bytes())
        manifest.append(dict(run_id=q['run_id'],path=f"execution/{q['run_id']}/checkpoint.pt",sha256=r['checkpoint_sha256'],bytes=(out/'checkpoint.pt').stat().st_size))
        print('REPLAY',q['run_id'],err,donor_error,flush=True)
    summary=dict(run_id=p['run_id'],status='COMPLETE_EXPLORATORY' if not failures else 'COMPLETE_WITH_FAILURES',seed=p['seed'],
        train_wall_budget_seconds=p['train_wall_seconds'],scored_targets_per_endpoint=8192,rows=sorted(rows,key=lambda r:r['ce']),failures=failures)
    ex.save(ROOT/'SUMMARY.json',summary);ex.save(ROOT/'CHECKS.json',checks);ex.save(ROOT/'CHECKPOINT_MANIFEST.json',manifest)
    ex.save(ROOT/'PREFIX_PROFILE.json',dict(scope='Post-hoc descriptive diagnostic, not a new selection metric. Last-target result has only16 targets. Prefix swapping is distribution-shifting.',
        interpretation='The declared whole-suffix mean mixes near and distant effects. Only final-target loss isolates a changed prefix entirely beyond512 observed inputs.',rows=profiles))
    ex.save(ROOT/'TERMINAL.json',dict(status=summary['status'],declared=len(p['runs']),completed=len(rows),failures=len(failures),
        primary='Final512-target CE on16 common windows',no_followup_running=True,checkpoint_uploads_require_separate_receipt_check=True))
    f=ROOT.parents[1]/'RUN_REGISTRY.json';reg=json.loads(f.read_text());entry=next(r for r in reg['runs'] if r['run_id']==p['run_id'])
    entry.update(status=summary['status'],summary=f'auto-research/results/{p["run_id"]}/SUMMARY.json',terminal=f'auto-research/results/{p["run_id"]}/TERMINAL.json',seed=p['seed'],completed_trials=len(rows))
    ex.save(f,reg);plot(summary)
    print(json.dumps({r['run_id']:round(r['ce'],6) for r in summary['rows']},indent=2))

def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=ROOT.parents[1]/'graphs'/ROOT.name;dest.mkdir(parents=True,exist_ok=True);ex.save(dest/'inputs.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    groups=[('lin-minGRU w160','#2166ac',lambda r:r['arm']=='mingru_linear' and r['width']==160 and r['segment_length']==32),
        ('lin-minGRU w224','#7b3294',lambda r:r['width']==224),('nanoGPT w108','#d95f02',lambda r:r['arm']=='nanogpt'),
        ('ordinary GRU w147','#666666',lambda r:r['arm']=='native'),('lin-minGRU w160 / chunk64','#00876c',lambda r:r['segment_length']==64)]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,color,select in groups:
        rows=sorted([r for r in summary['rows'] if select(r)],key=lambda r:r['context'])
        if not rows:continue
        for ax,key,scale in [(axes[0],'ce',1),(axes[1],'tokens_per_wall_second',.001)]:
            ax.plot([r['context'] for r in rows],[r[key]*scale for r in rows],'-o',label=name,color=color,lw=2,ms=6)
    axes[0].set_ylabel('Validation cross-entropy (lower is better)');axes[1].set_ylabel('Training targets / second (thousands)')
    for ax in axes:ax.set_xticks([512,1024,2048]);ax.set_xlabel('Training and evaluation context');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='best');fig.suptitle('Context and recurrent capacity / 90-second CPU recovery sweep')
    fig.text(.5,.015,'Same final 512 targets per window / one seed / wide minGRU: 1.77M weights; others approx. 1M stored weights',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ('png','svg'):fig.savefig(dest/f'context_tradeoffs.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for _,color,select in groups:
        for r in sorted([r for r in summary['rows'] if select(r)],key=lambda r:r['context']):
            ax=axes[1] if r['arm'] in ('nanogpt','native') else axes[0]
            ax.plot([t['train_wall_seconds'] for t in r['monitor']],[t['validation_ce'] for t in r['monitor']],
                {512:':',1024:'--',2048:'-'}[r['context']],label=r['label'],color=color,lw=1.7)
    for ax,title in zip(axes,['Linear minGRU variants','Ordinary GRU and nanoGPT controls']):
        ax.set_title(title);ax.set_xlabel('Training wall seconds');ax.grid(alpha=.2);ax.legend(fontsize=7.5);ax.set_xlim(0,90)
    axes[0].set_ylabel('Monitoring cross-entropy (lower is better)');fig.suptitle('Every declared trial / separate fixed monitoring windows')
    fig.text(.5,.015,'Monitor: 4 windows / 2,048 scored targets. Final ranking: 16 different windows / 8,192 scored targets.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ('png','svg'):fig.savefig(dest/f'loss_curves.{ext}')
    plt.close(fig)

if __name__=='__main__':main()
