"""Joint original-hierarchy mechanism pilot. No official test access."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parent
DATA=Path('/Volumes/SP AI 01_16/w560-function-weight-wt2-word-sota-20260828/data/wikitext-2')
PINS={'train.txt':'71220dd7735e8715884cf57d21fa6306ebc3249c0979f51c1fb22314e0842298','valid.txt':'1bd7ccbea6740d1f3837c268e41cd6c983e2e1d2673843b3d2c641f412c640b7'}
CONFIG=dict(seeds=[73,191,307],arms=['original','local','cross'],learning_rates=[.001,.003],width=64,scales=4,context=128,batch=4,updates=6000,selection_every=1000,selection_start=0,selection_targets=16384,confirmation_start=900000,confirmation_targets=32768)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2)

def load_model_module():
    spec=importlib.util.spec_from_file_location('language_regularized',ROOT/'regularized_rotation.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def construct(module,arm,initial):
    base=module.reference.reference.reference.PersistentMultiScaleLM(256,CONFIG['width'],CONFIG['scales'])
    base.load_state_dict(initial)
    return base if arm=='original' else module.RegularizedRotation(base,arm=='cross')

def stream_batch(data,starts,step):
    t=CONFIG['context']
    windows=torch.stack([data[start+step*t:start+(step+1)*t+1] for start in starts])
    assert windows.shape==(CONFIG['batch'],t+1)
    return windows[:,:-1],windows[:,1:]

def evaluate(model,data,start,targets):
    assert targets%(CONFIG['batch']*CONFIG['context'])==0
    steps=targets//(CONFIG['batch']*CONFIG['context'])
    starts=[start+i*steps*CONFIG['context'] for i in range(CONFIG['batch'])]
    losses=[]
    digest=hashlib.sha256()
    state=None
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            x,y=stream_batch(data,starts,step)
            logits,state=model(x,state,return_state=True)
            assert torch.isfinite(logits).all()
            losses.append(float(torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum')))
            digest.update(logits.contiguous().numpy().tobytes())
    model.train()
    return dict(ce=math.fsum(losses)/targets,targets=targets,logit_sha256=digest.hexdigest(),target_interval=[start+1,start+targets])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    module=load_model_module()
    if not args.run:
        torch.manual_seed(8193)
        initial=module.reference.reference.reference.PersistentMultiScaleLM(256,64,4).state_dict()
        data=torch.randint(256,(4097,))
        rows=[]
        for arm in CONFIG['arms']:
            model=construct(module,arm,initial)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
            state=None
            start=time.perf_counter()
            for step in range(4):
                x,y=stream_batch(data,[0,1024,2048,3072],step)
                optimizer.zero_grad()
                logits,state=model(x,state,return_state=True)
                loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                state=state.detach()
            first=evaluate(model,data,0,2048)
            second=evaluate(model,data,0,2048)
            assert first==second
            rows.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),four_updates_seconds=time.perf_counter()-start,duplicate_exact=True))
        save(ROOT/'REGULARIZED_LANGUAGE_PREFLIGHT.json',dict(rows=rows,task_data_reads=0,official_test_reads=0,config=CONFIG))
        print(json.dumps(rows))
        return
    out=ROOT/'regularized-language-run'
    out.mkdir(exist_ok=False)
    source_files=[ROOT/'regularized_language.py',ROOT/'regularized_rotation.py',ROOT/'packed_rotation.py',ROOT/'cross_scale_rotation_canary.py']+list((ROOT/'experiments').rglob('*.py'))
    seals={str(p):sha(p) for p in source_files}
    save(out/'PROTOCOL.json',dict(config=CONFIG,source_hashes=seals,data_hashes=PINS,scope='Original-hierarchy byte mechanism pilot, not matched FLOPs, word perplexity, globally pristine validation, novelty or SOTA. Confirmation target regions do not overlap selection. No official-test eligibility.',official_test_reads=0))
    def check():
        assert all(sha(Path(path))==pin for path,pin in seals.items())
        assert all(sha(DATA/name)==pin for name,pin in PINS.items())
    try:
        check()
        train=torch.tensor(list((DATA/'train.txt').read_bytes()),dtype=torch.long)
        valid=torch.tensor(list((DATA/'valid.txt').read_bytes()),dtype=torch.long)
        candidates=[]
        winners=[]
        streams={}
        for seed in CONFIG['seeds']:
            torch.manual_seed(seed)
            initial=module.reference.reference.reference.PersistentMultiScaleLM(256,64,4).state_dict()
            generator=torch.Generator().manual_seed(seed)
            starts=torch.randint(len(train)-CONFIG['updates']*CONFIG['context']-1,(4,),generator=generator).tolist()
            streams[str(seed)]=starts
            for arm in CONFIG['arms']:
                arm_trials=[]
                for lr in CONFIG['learning_rates']:
                    check()
                    model=construct(module,arm,initial)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,foreach=False)
                    state=None
                    best=None
                    for step in range(CONFIG['updates']):
                        x,y=stream_batch(train,starts,step)
                        optimizer.zero_grad()
                        logits,state=model(x,state,return_state=True)
                        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
                        loss.backward()
                        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                        optimizer.step()
                        state=state.detach()
                        if (step+1)%CONFIG['selection_every']==0:
                            metric=evaluate(model,valid,CONFIG['selection_start'],CONFIG['selection_targets'])
                            print(json.dumps(dict(seed=seed,arm=arm,lr=lr,step=step+1,selection=metric)),flush=True)
                            if best is None or metric['ce']<best['selection']['ce']:
                                checkpoint=out/f'{seed}-{arm}-{lr}-{step+1}.pt'
                                torch.save(model.state_dict(),checkpoint)
                                best=dict(seed=seed,arm=arm,lr=lr,step=step+1,selection=metric,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
                    best['completed_updates']=CONFIG['updates']
                    best['train_target_exposures']=CONFIG['updates']*CONFIG['batch']*CONFIG['context']
                    candidates.append(best)
                    arm_trials.append(best)
                winners.append(min(arm_trials,key=lambda r:(r['selection']['ce'],r['lr'],r['step'],r['checkpoint_sha256'])))
        save(out/'CANDIDATES.json',dict(trials=candidates,streams=streams))
        save(out/'FREEZE.json',dict(selected=winners))
        results=[]
        for row in winners:
            check()
            assert sha(Path(row['checkpoint']))==row['checkpoint_sha256']
            torch.manual_seed(row['seed'])
            initial=module.reference.reference.reference.PersistentMultiScaleLM(256,64,4).state_dict()
            model=construct(module,row['arm'],initial)
            saved=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
            assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in saved.values())
            model.load_state_dict(saved,strict=True)
            assert evaluate(model,valid,0,CONFIG['selection_targets'])==row['selection']
            first=evaluate(model,valid,CONFIG['confirmation_start'],CONFIG['confirmation_targets'])
            second=evaluate(model,valid,CONFIG['confirmation_start'],CONFIG['confirmation_targets'])
            assert first==second
            results.append(dict(candidate=row,first=first,second=second,duplicate_exact=True))
        paired=[]
        for seed in CONFIG['seeds']:
            ce={r['candidate']['arm']:r['first']['ce'] for r in results if r['candidate']['seed']==seed}
            paired.append(dict(seed=seed,ce=ce,passed=ce['cross']<ce['local'] and ce['cross']<ce['original']))
        check()
        save(out/'COMPLETION.json',dict(results=results,paired=paired,advance_gate_passed=all(p['passed'] for p in paired),official_test_reads=0,source_hashes=seals))
    except BaseException as error:
        save(out/'FAILURE.json',dict(type=type(error).__name__,message=str(error),official_test_reads=0))
        raise

if __name__=='__main__':
    main()
