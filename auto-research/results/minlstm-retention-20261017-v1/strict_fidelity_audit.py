"""Audit the preserved verification under the ORIGINAL, unrelaxed tolerance.

This is an evidence audit, not a new numerical run or a repair of a failed gate.
The archived verify.py relaxed its FP32 optimizer-parameter threshold before
training, contrary to the repository's no-post-failure-relaxation instruction.
Keep that original source and every outcome; flag the affected control.
"""
import json,time
from experiment import ROOT,save,sha

def main():
    checks=json.loads((ROOT/'VERIFICATION.json').read_text())['checks'];rows=[]
    for row in checks:
        tol=2e-12 if row['dtype']=='torch.float64' else 2e-6
        fields=['output_max_abs','gradient_max_abs','input_gradient_max_abs','chunk_max_abs','optimizer_max_abs']
        failed=[key for key in fields if row[key]>=tol]
        rows.append(dict(arm=row['arm'],dtype=row['dtype'],original_tolerance=tol,
                         original_gate_pass=not failed,failed_fields=failed,
                         observed={key:row[key] for key in fields}))
    result=dict(status='ORIGINAL_GATE_FAILED_FOR_SUPPLEMENTAL_LSTM64_CONTROL',
       source_verification_sha256=sha(ROOT/'VERIFICATION.json'),rows=rows,
       matched_state_arms_pass=all(r['original_gate_pass'] for r in rows if r['arm']!='lstm64'),
       deviation='FP32 optimizer-parameter threshold was relaxed from2e-6 to5e-5 before training after observing2.03e-5 on LSTM64. The repo forbids post-failure relaxation. Its later PASS is not an original-gate pass.',
       consequence='Retain all measurements; LSTM64 is descriptive only and not a fidelity-qualified confirmation control. minGRU64, minLSTM64 and LSTM32 passed the original thresholds. No architecture promotion.',
       new_training=False,new_evaluation=False,audited_unix=time.time())
    save(ROOT/'STRICT_FIDELITY_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
