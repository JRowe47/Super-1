"""Original parity checks plus independent dense magnitude/cap checks."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib, json, time
from pathlib import Path
import torch
import core as C
import baseline_verify as B

ROOT = Path(__file__).resolve().parent


def main():
    begin = time.process_time()
    registration = json.loads((ROOT / 'REGISTRATION.json').read_text())
    for name, digest in registration['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    B.verify()
    result = json.loads((ROOT / 'VERIFICATION.json').read_text())
    result['extension'] = {}
    for dtype, tol in ((torch.float32, 2e-6), (torch.float64, 2e-12)):
        s = C.init(18719, dtype)
        capped = C.bounded(s)
        dense, cd = C.decode(s), C.decode(capped)
        errors = {}
        errors['cap_dense_equivalence'] = max(float((cd[k] - dense[k] * min(
            1., C.CAP * (1 - 4 * torch.finfo(dtype).eps) /
            float(torch.linalg.svdvals(dense[k])[0]))).abs().max()) for k in C.FK)
        capmax = max(float(torch.linalg.svdvals(cd[k])[0]) for k in C.FK)
        assert capmax <= C.CAP
        assert all(torch.equal(s[k], capped[k]) for k in C.WK)
        fixed = C.clone(capped)
        adapted = C.clone(capped)
        for i, k in enumerate(C.FK):
            adapted[k] = torch.roll(adapted[k], 1 + i) * .8
        adapted = C.bounded(adapted)
        gen = torch.Generator().manual_seed(52871)
        x = torch.randn(9, C.D, generator=gen, dtype=dtype)
        y = torch.randint(0, 2, (9, C.R), generator=gen).to(dtype)
        for arm, current, target in (
                ('frozen_match_adaptive', fixed, adapted),
                ('adaptive_match_frozen', adapted, fixed)):
            # Per-example dense equations, independently normalized in dense coordinates.
            wd, td = C.decode(current), C.decode(C.with_feedback(current, target))
            gs, gt = C.dense_reference(wd, x, y), C.dense_reference(td, x, y)
            new, bank, audit = C.policy_step(current, x, y, arm,
                                            reference=fixed, shadow=adapted)
            nd = C.decode(new)
            for k in ('W0', 'W1'):
                us, ut = gs[k] + C.WD_WB * wd[k], gt[k] + C.WD_WB * wd[k]
                direct = wd[k] - C.LR * us * (ut.norm() / us.norm())
                errors[arm + '_' + k] = float((nd[k] - direct).abs().max())
                assert audit[k]['relative_norm_error'] <= 2e-6
            errors[arm + '_W2'] = float((nd['W2'] - (
                wd['W2'] - C.LR * (gs['W2'] + C.WD_WB * wd['W2']))).abs().max())
            if arm == 'frozen_match_adaptive':
                assert all(torch.equal(new[k], fixed[k]) for k in C.FK)
                assert all(bank[k].data_ptr() != current[k].data_ptr() for k in C.FK)
                expected = C.bounded(C.apply(C.with_feedback(current, adapted),
                                              C.encode(gt), C.FK))
                errors['shadow_update'] = max(float((bank[k] - expected[k]).abs().max()) for k in C.FK)
        assert all(v <= tol for v in errors.values()), (dtype, errors)
        old = torch.autograd.grad
        def forbidden(*args, **kwargs):
            raise AssertionError('Autograd on local policy path')
        try:
            torch.autograd.grad = forbidden
            for arm in C.EXPERIMENT_ARMS[1:]:
                C.policy_step(adapted, x, y, arm, fixed, fixed)
        finally:
            torch.autograd.grad = old
        zero, _ = C.matched_drive(torch.ones(3, dtype=dtype), torch.zeros(3, dtype=dtype))
        assert bool((zero == 0).all())
        try:
            C.matched_drive(torch.zeros(3, dtype=dtype), torch.ones(3, dtype=dtype))
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Unmatchable direction did not fail')
        result['extension'][str(dtype)] = {'tolerance': tol, 'checks': errors,
                                          'maximum_capped_sigma': capmax}
    data = C.make_data(96977, 12, 32, 16)
    a, b = C.bounded(C.init(96977)), C.init(96977)
    for ix in data['warm_ix']:
        a = C.warm_step(a, data['base_x'][ix], data['base_y'][ix])
        b = C.update(b, data['base_x'][ix], data['base_y'][ix], 'backprop')
        assert all(torch.equal(a[k], b[k]) for k in C.WK)
    result.update(status='PASS', common_forward_warmup_bitwise=True,
                  matching_handles_zero=True, local_matching_without_autograd=True,
                  cpu_seconds=time.process_time() - begin)
    (ROOT / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verification': 'PASS', 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    main()
