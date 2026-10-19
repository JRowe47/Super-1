"""Frozen, inference-only norm controls and direction/context contrasts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','3')
import argparse,gc,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
import scoring as S

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';RAW=ROOT/'responses'
P=json.loads((ROOT/'protocol.json').read_text())
LO,HI=589824,655360;PATH='aaaaad'
PATHS=[''.join(p) for p in itertools.product('ad',repeat=6)]
SCAN_TEXT=[
 'The gardener filled a watering can and carried it to the greenhouse. Young plants stood in rows beneath the glass. She checked the soil before giving each pot a little water.',
 'A wooden bridge crosses the narrow valley. People use it to reach the village on the far side. After a storm, workers inspect the supports and clear fallen branches from the path.',
 'The student arranged the colored tiles into a square. Each row contained the same number of tiles. Moving one tile left a gap that made the pattern easier to understand.',
 'At the harbor, a crew unloaded boxes from a fishing boat. The ropes were wet and the deck was slippery. A worker carried the empty crates back to the warehouse.'
]
CONFIRM_TEXT=[
 'The baker covered the dough with a towel and left it in a warm place. An hour later it had grown larger. She shaped it into rolls and placed them on a tray.',
 'A guide pointed to a narrow trail above the lake. The group followed it slowly through the trees. Near the top, an opening between the rocks revealed a wide view of the valley.',
 'The telescope rested on a solid metal stand. Before looking through it, the observer adjusted the focus. A bright planet appeared beside several faint points of light.',
 'A delivery arrived at the office before lunch. Each package had a name and a room number printed on the label. The receptionist checked the list and placed the boxes on a cart.',
 'The child wound a string around the spool and tied a small knot. When the kite rose into the wind, the string became tight. Her brother helped keep it clear of the fence.',
 'Several birds gathered around the shallow dish. One splashed water over its feathers while the others waited nearby. When a cat appeared, the birds flew into the branches of a tree.',
 'The museum displayed a collection of old clocks. Some had long pendulums and others used small springs. A note beside each clock explained when it was made and how it worked.',
 'Before the performance, the actors checked their costumes and read through the opening scene. Behind the curtain, a stagehand moved a chair into position. The lights dimmed as the audience became quiet.'
]

def dump(name,obj):
    dest=OUT/name;tmp=dest.with_suffix(dest.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def prepare_inputs():
    excluded=set(json.loads((OUT/'prior_token_ids_excluded.json').read_text()))
    rng=np.random.default_rng(P['confirmation']['input_seed'])
    pool=rng.permutation(np.array([x for x in range(100,50000) if x not in excluded]))
    offset=0;sets={}
    for split,per in [('scan',8),('confirmation',16)]:
        rows=[]
        for n in [16,32,48]:
            for j in range(per):
                seq=pool[offset:offset+n];offset+=n
                rows.append(dict(id=len(rows),n=n,prefix=seq.tolist(),order=rng.permutation(n-1).tolist()))
        sets[split]=rows
    used=[{t for row in rows for t in row['prefix']} for rows in sets.values()]
    assert not used[0]&used[1] and not (used[0]|used[1])&excluded
    tok=S.AutoTokenizer.from_pretrained(str(S.CHECKPOINTS/'tokenizer'),local_files_only=True)
    prose={'scan':SCAN_TEXT,'confirmation':CONFIRM_TEXT}
    dump('probe_inputs.json',dict(sets=sets,prose=prose,prose_ids={k:[tok(t).input_ids for t in v] for k,v in prose.items()},
        token_ids_disjoint=True,prior_token_ids_excluded=len(excluded),description='Original diagnostic prose, not verified absent from model pretraining.'))

def prepare_curves():
    full={s:S.tensor(s,3) for s in [512,1000,8000]}
    windows={s:w.ravel()[LO:HI].copy() for s,w in full.items()}
    e=windows[1000].astype(np.float64)-windows[512];l=windows[8000].astype(np.float64)-windows[512]
    ec=np.stack([S.packet(e,p) for p in PATHS]);lc=np.stack([S.packet(l,p) for p in PATHS])
    early={p:S.ipacket(c,p) for p,c in zip(PATHS,ec)}
    late=S.ipacket(lc[PATHS.index(PATH)],PATH)
    norms=np.linalg.norm(ec,axis=1);r=float(norms[PATHS.index(PATH)])
    ratio=float(np.linalg.norm(late)/r)
    checks={}
    for label,delta,cs in [('early',e,ec),('late',l,lc)]:
        reconstructed=np.sum([S.ipacket(c,p) for p,c in zip(PATHS,cs)],axis=0)
        err=float(np.max(abs(reconstructed-delta)));assert err<1e-12
        checks[label+'_reconstruction_max_abs']=err
        checks[label+'_parseval_relative_error']=float(abs(np.sum(cs**2)-np.sum(delta**2))/np.sum(delta**2))
    dot=float(np.dot(early[PATH],late)/(r*np.linalg.norm(late)))
    matched={p:v*(r/float(np.linalg.norm(v))) for p,v in early.items()}
    matched_late=late/ratio
    np.savez_compressed(ROOT/'native_local_curves.npz',**{f'step{s}':v for s,v in windows.items()},
        paths=np.array(PATHS),early_coefficients=ec,late_coefficients=lc,
        selected_early=early[PATH],selected_late=late,selected_late_matched=matched_late,
        native_flat_indices=np.arange(LO,HI))
    meta=dict(common_norm=r,late_to_early_norm_ratio=ratio,selected_early_late_cosine=dot,
        selected_angle_degrees=float(np.degrees(np.arccos(dot))),selected_path=PATH,layer=3,head=6,
        start=LO,end=HI,coefficient_count_per_path=1024,native_support_per_path=65536,
        early_norms=dict(zip(PATHS,map(float,norms))),normalization_factors={p:float(r/n) for p,n in zip(PATHS,norms)},
        weight_only_max_norm_path=PATHS[int(norms.argmax())],**checks)
    dump('curve_metadata.json',meta)
    return full,matched,matched_late,meta

@torch.inference_mode()
def run_case(m,step,base,split,cid,v,lam,reference,meta):
    t=time.time();p=S.param(m,3)
    target=base.copy();target.ravel()[LO:HI]=(base.ravel()[LO:HI].astype(np.float64)-lam*v).astype(np.float32)
    actual=target.ravel()[LO:HI].astype(np.float64)-base.ravel()[LO:HI]
    intended=-lam*v
    assert np.array_equal(target.ravel()[:LO],base.ravel()[:LO]) and np.array_equal(target.ravel()[HI:],base.ravel()[HI:])
    p.copy_(torch.from_numpy(target))
    try:score,h,_=S.evaluate(m,split,reference)
    finally:p.copy_(torch.from_numpy(base))
    assert torch.equal(p,torch.from_numpy(base))
    row=dict(id=cid,step=step,split=split,lam=float(lam),**meta,**score,
        intended_update_norm=float(np.linalg.norm(intended)),actual_update_norm=float(np.linalg.norm(actual)),
        rounding_l2_error=float(np.linalg.norm(actual-intended)),rounding_max_abs_error=float(abs(actual-intended).max()),
        changed_native_weights=int(np.count_nonzero(actual)),changed_outside_window=False,restored_exactly=True,seconds=time.time()-t)
    np.savez_compressed(RAW/f'{split}_step{step}_{cid}.npz',**h)
    return row

@torch.inference_mode()
def baseline(m,step,split):
    score,h,teachers=S.evaluate(m,split)
    np.savez_compressed(RAW/f'{split}_step{step}_baseline.npz',**h)
    return dict(id='baseline',step=step,split=split,lam=0.,**score),teachers

def case_id(direction,path,lam):
    value=('m' if lam<0 else 'p')+f'{abs(lam):.6f}'.rstrip('0').rstrip('.').replace('.','d')
    return direction+'_'+path+'_'+value

@torch.inference_mode()
def main(checkpoints):
    begin=time.time();S.CHECKPOINTS=checkpoints
    torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol_hash=digest(ROOT/'protocol.json');prepare_inputs();full,early,late,meta=prepare_curves()
    m=S.model(1000);b,teachers=baseline(m,1000,'scan');scan=[b]
    print('SCAN baseline',b['repeat']['accuracy'],b['scrambled']['accuracy'],flush=True)
    for path in PATHS:
        cid=case_id('early',path,1)
        row=run_case(m,1000,full[1000],'scan',cid,early[path],1,teachers,dict(path=path,direction='early',role='all64_equal_norm'))
        scan.append(row);dump('scan.json',scan)
        print('SCAN',path,'NLL',round(row['repeat']['target_nll']-b['repeat']['target_nll'],6),'sec',round(row['seconds'],2),flush=True)
    winner=max(scan[1:],key=lambda r:r['repeat']['target_nll'])['path']
    selected=list(dict.fromkeys([PATH,'aaaaaa','dddddd',meta['weight_only_max_norm_path'],winner]))
    selection=dict(prior_selected=PATH,scan_winner=winner,paths=selected,confirmation_consulted=False,
        selection_rule=P['confirmation_selection'],protocol_sha256=protocol_hash,scan_sha256=digest(OUT/'scan.json'))
    dump('selection_locked.json',selection);selection_hash=digest(OUT/'selection_locked.json')
    print('LOCKED',json.dumps(selection),flush=True)
    restored,_,_=S.evaluate(m,'scan',teachers)
    err=max(abs(restored[k]['target_nll']-b[k]['target_nll']) for k in ['repeat','scrambled']);assert err==0
    scan_seconds=time.time()-begin
    del m,teachers;gc.collect()
    results=[];restores=[dict(split='scan',step=1000,maximum_nll_difference=err)]
    for step in [512,1000,8000]:
        m=S.model(step);b,teachers=baseline(m,step,'confirmation');results.append(b)
        print('CONFIRM baseline',step,b['repeat']['accuracy'],b['scrambled']['accuracy'],flush=True)
        w=m.embed_out.weight.double();mean=w.mean(0);gram=w.T@w/w.shape[0]-torch.outer(mean,mean)
        np.save(OUT/f'logit_gram_step{step}.npy',gram.numpy());del w,gram
        cases=[]
        for direction,v in [('early',early[PATH]),('late',late)]:
            for lam in [-1,-.5,.5,1]:cases.append((direction,PATH,lam,v,'direction_context_dose'))
            if step in [1000,8000]:cases.append((direction,PATH,meta['late_to_early_norm_ratio'],v,'natural_late_magnitude'))
        if step==1000:
            for path in selected:
                if path!=PATH:
                    for lam in [-1,1]:cases.append(('early',path,lam,early[path],'locked_norm_control'))
        for direction,path,lam,v,role in cases:
            cid=case_id(direction,path,lam)
            row=run_case(m,step,full[step],'confirmation',cid,v,lam,teachers,dict(direction=direction,path=path,role=role))
            results.append(row);dump('confirmation.json',results)
            print('CONFIRM',step,cid,'NLL',round(row['repeat']['target_nll']-b['repeat']['target_nll'],6),
                'scrambled',round(row['scrambled']['target_nll']-b['scrambled']['target_nll'],6),'sec',round(row['seconds'],2),flush=True)
        restored,_,_=S.evaluate(m,'confirmation',teachers)
        err=max(abs(restored[k]['target_nll']-b[k]['target_nll']) for k in ['repeat','scrambled']);assert err==0
        restores.append(dict(split='confirmation',step=step,maximum_nll_difference=err))
        del m,teachers;gc.collect()
    assert digest(ROOT/'protocol.json')==protocol_hash and digest(OUT/'selection_locked.json')==selection_hash
    variants=len(scan)-1+len(results)-3;assert variants<=100
    dump('environment.json',dict(total_seconds=time.time()-begin,scan_seconds=scan_seconds,variants=variants,training_updates=0,
        protocol_sha256=protocol_hash,selection_sha256=selection_hash,selection_changed=False,restore_checks=restores,
        torch=torch.__version__,threads=torch.get_num_threads(),dtype='Float64 transforms; float32 inference',cpu_only=True))
    print('COMPLETE',json.dumps(S.read('environment.json')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');args=ap.parse_args();main(args.checkpoints)
