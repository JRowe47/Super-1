"""Read-only receipt audit, separate from sealed trainer; no predictions."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'regularized-language-run'

def read(name):
    return json.loads((ROOT/name).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    protocol=read('PROTOCOL.json')
    for path,pin in protocol['source_hashes'].items():
        assert digest(Path(path))==pin, ('source changed',path)
    if (ROOT/'FAILURE.json').exists():
        return dict(status='failure',failure=read('FAILURE.json'))
    if not (ROOT/'COMPLETION.json').exists():
        return dict(status='pending',source_seals_intact=True,scope='Receipt status only, not process liveness')
    config=protocol['config']
    records=read('CANDIDATES.json')
    candidates=records['trials']
    freeze=read('FREEZE.json')['selected']
    terminal=read('COMPLETION.json')
    expected={(seed,arm,lr) for seed in config['seeds'] for arm in config['arms'] for lr in config['learning_rates']}
    slots=[(r['seed'],r['arm'],r['lr']) for r in candidates]
    assert len(slots)==len(expected) and set(slots)==expected
    assert set(records['streams'])==set(map(str,config['seeds']))
    for row in candidates:
        assert row['completed_updates']==config['updates']
        assert row['train_target_exposures']==config['updates']*config['context']*config['batch']
        assert row['step'] in range(config['selection_every'],config['updates']+1,config['selection_every'])
        path=Path(row['checkpoint']).resolve(strict=True)
        assert path.parent==ROOT.resolve() and digest(path)==row['checkpoint_sha256']
        assert row['selection']['targets']==config['selection_targets']
        assert math.isfinite(row['selection']['ce']) and row['selection']['ce']>0
    canonical=[]
    for seed in config['seeds']:
        for arm in config['arms']:
            choices=[r for r in candidates if r['seed']==seed and r['arm']==arm]
            canonical.append(min(choices,key=lambda r:(r['selection']['ce'],r['lr'],r['step'],r['checkpoint_sha256'])))
    assert freeze==canonical
    results=terminal['results']
    assert len(results)==len(freeze)
    assert [r['candidate'] for r in results]==freeze
    for row in results:
        assert row['first']==row['second'] and row['duplicate_exact'] is True
        assert row['first']['targets']==config['confirmation_targets']
        assert row['first']['target_interval']==[config['confirmation_start']+1,config['confirmation_start']+config['confirmation_targets']]
        assert len(row['first']['logit_sha256'])==64
        assert all(c in '0123456789abcdef' for c in row['first']['logit_sha256'])
        assert math.isfinite(row['first']['ce']) and row['first']['ce']>0
    paired=[]
    for seed in config['seeds']:
        ce={r['candidate']['arm']:r['first']['ce'] for r in results if r['candidate']['seed']==seed}
        paired.append(dict(seed=seed,ce=ce,passed=ce['cross']<ce['local'] and ce['cross']<ce['original']))
    assert terminal['paired']==paired
    assert terminal['advance_gate_passed']==all(p['passed'] for p in paired)
    assert terminal['official_test_reads']==0 and terminal['source_hashes']==protocol['source_hashes']
    return dict(status='terminal_receipt_audit_passed',paired=paired,advance_gate_passed=terminal['advance_gate_passed'],receipts={name:digest(ROOT/name) for name in ['PROTOCOL.json','CANDIDATES.json','FREEZE.json','COMPLETION.json']},limitations='No new native replay or independent provenance of every training update; audits source seals and terminal receipts, not scientific novelty or SOTA')

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
