"""Independent reload check for the two saved decoder continuations; no training."""
import argparse, hashlib, importlib.metadata, json, platform
from pathlib import Path
import numpy as np
import torch
import scoring as S
from pythia_train import lm_eval
from transforms import haar, inverse

ROOT = Path(__file__).resolve().parent

def main(checkpoints):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    S.CHECKPOINTS = checkpoints
    manifest = json.loads((ROOT/'checkpoint_manifest.json').read_text())
    with (checkpoints/'step143000/model.safetensors').open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == manifest['sha256']
    recorded = json.loads((ROOT/'results/pythia_evaluations.json').read_text())
    data = np.load(ROOT/'data/lm_token_blocks.npz')
    m = S.model(143000)
    _, _, teachers = S.evaluate(m, 'retention')
    checks = []
    for arm in ['native_adamw', 'curve_adamw']:
        params = dict(m.named_parameters())
        with np.load(ROOT/'models'/f'{arm}_final_decoder.npz') as z:
            assert len(z.files) == 72
            with torch.no_grad():
                for name in z.files:
                    params[name].copy_(torch.from_numpy(z[name]))
        reference = next(r for r in recorded if r['arm']==arm and r['step']==24)
        lm = lm_eval(m, data['validation'])
        ret, hidden, _ = S.evaluate(m, 'retention', teachers)
        lm_error = float(np.max(np.abs(np.array(lm['block_nll'])-reference['validation']['block_nll'])))
        ret_error = max(abs(ret[k][metric]-reference['retention'][k][metric])
            for k, metrics in [('repeat',['accuracy','target_nll']),('scrambled',['accuracy','target_nll']),('prose',['nll','kl_from_baseline','argmax_changed'])]
            for metric in metrics)
        with np.load(ROOT/'responses'/f'{arm}_24.npz') as z:
            hidden_error = max(float(np.max(np.abs(hidden[k]-z[k]))) for k in hidden)
        assert max(lm_error, ret_error, hidden_error) < 1e-6
        with torch.no_grad():
            before = m(torch.tensor(data['validation'][:1]),use_cache=False).logits
            max_roundtrip = 0.
            for name, p in params.items():
                if not name.startswith('gpt_neox.layers.'): continue
                rec = inverse(haar(p.detach().double().flatten())).reshape(p.shape).float()
                max_roundtrip = max(max_roundtrip,float((rec-p).abs().max()))
                p.copy_(rec)
            after = m(torch.tensor(data['validation'][:1]),use_cache=False).logits
            logit_error = float((after-before).abs().max())
        assert max_roundtrip < 1e-12 and logit_error < 1e-6
        checks.append(dict(arm=arm,validation_block_max_error=lm_error,
            retention_metric_max_error=ret_error,retention_hidden_max_error=hidden_error,
            saved_weight_curve_roundtrip_max_error=max_roundtrip,
            first_validation_block_roundtrip_logit_max_error=logit_error))
    result = dict(checks=checks,scope='Fresh model reload, 24 validation blocks and all retention probes; saved weight re-encoding checked on one validation block. No training or model selection.',
        python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ['torch','numpy','transformers','safetensors','pyarrow','matplotlib']})
    (ROOT/'results/saved_model_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints')
    main(p.parse_args().checkpoints)
