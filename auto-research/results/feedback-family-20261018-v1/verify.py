"""Independent dense family switches, exact inactive state, retained regressions."""
import component_verify as V
from component_verify import C, torch, json, time, ROOT
import traceback


def verify_families(seed, dtype, tol):
    gen = torch.Generator().manual_seed(seed + 80000000)
    initial = C.bounded(C.init(seed, dtype))
    errors, capmax = {}, 0.
    for case, gain in (('bounded', 1.), ('projection_active', 1.5)):
        x = torch.randn(13, C.D, generator=gen, dtype=dtype)
        y = torch.randint(0, 2, (13, C.R), generator=gen).to(dtype)
        for policy, switches in C.FAMILY_SWITCHES.items():
            state = C.clone(initial)
            active = C.FAMILY_KEYS[policy]
            for k in active:
                state[k] *= gain
            snapshot = C.clone(state)
            dense = C.decode(state)
            gradients = C.dense_reference(dense, x, y)
            direct = {}
            for k, w in dense.items():
                if k in C.WK:
                    z = w - C.LR * (gradients[k] + C.WD_WB * w)
                elif k in active:
                    z = w - C.LR * gradients[k]
                    sigma = float(torch.linalg.svdvals(z)[0])
                    limit = 2. * (1 - 4 * torch.finfo(dtype).eps)
                    z = z * min(1., limit / sigma) if sigma > 0 else z
                else:
                    z = w
                direct[k] = z
            new = C.family_step(state, x, y, *switches)
            observed = C.decode(new)
            errors[case + '_' + policy] = max(float((observed[k] - direct[k]).abs().max()) for k in direct)
            assert errors[case + '_' + policy] <= tol, (seed, dtype, errors)
            frozen, _, _ = C.Q.policy_step(state, x, y, 'frozen', shadow={})
            assert all(torch.equal(new[k], frozen[k]) for k in C.WK)
            assert all(torch.equal(new[k], state[k]) for k in C.FK if k not in active)
            assert all(torch.equal(state[k], snapshot[k]) for k in state)
            assert all(not v.requires_grad and v.grad_fn is None for v in new.values())
            capmax = max(capmax, max(float(torch.linalg.svdvals(observed[k])[0]) for k in C.FK))
            assert capmax <= 2.
            if active:
                assert any(not torch.equal(new[k], state[k]) for k in active)
            if policy in ('frozen', 'learning'):
                ref, _, _ = C.Q.policy_step(state, x, y, policy, shadow={})
                assert all(torch.equal(new[k], ref[k]) for k in new)
    trajectories = {name: C.clone(initial) for name in C.FAMILY_SWITCHES}
    frozen, full = C.clone(initial), C.clone(initial)
    for _ in range(12):
        x = torch.randn(7, C.D, generator=gen, dtype=dtype)
        y = torch.randint(0, 2, (7, C.R), generator=gen).to(dtype)
        for name, switches in C.FAMILY_SWITCHES.items():
            trajectories[name] = C.family_step(trajectories[name], x, y, *switches)
            assert all(torch.equal(trajectories[name][k], initial[k])
                       for k in C.FK if k not in C.FAMILY_KEYS[name])
        frozen, _, _ = C.Q.policy_step(frozen, x, y, 'frozen', shadow={})
        full, _, _ = C.Q.policy_step(full, x, y, 'learning', shadow={})
        assert all(torch.equal(trajectories['frozen'][k], frozen[k]) and
                   torch.equal(trajectories['learning'][k], full[k]) for k in full)
    old = torch.autograd.grad
    def forbidden(*args, **kwargs):
        raise AssertionError('Autograd entered local family rule')
    try:
        torch.autograd.grad = forbidden
        for name in C.FAMILY_SWITCHES:
            C.policy_step(initial, x, y, name, shadow={})
    finally:
        torch.autograd.grad = old
    return {'seed': seed, 'dtype': str(dtype), 'tolerance': tol, 'errors': errors,
            'maximum_capped_sigma': capmax, 'inactive_matrices_bitwise_frozen_12steps': True,
            'same_state_forward_updates_bitwise': True, 'frozen_full_endpoints_and_12step_trajectories_bitwise': True,
            'input_unmodified_output_detached': True, 'all_local_paths_without_autograd': True,
            'active_family_nonzero_update_exercised': True}


def main():
    begin = time.process_time()
    V.main()
    out = json.loads((ROOT / 'VERIFICATION.json').read_text())
    out['feedback_families'] = [verify_families(seed, dtype, tol) for seed in (101927, 16827, 25827)
                                for dtype, tol in ((torch.float32, 2e-6), (torch.float64, 2e-12))]
    out.update(status='PASS', cpu_seconds=time.process_time() - begin)
    (ROOT / 'VERIFICATION.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'feedback_family_verification': 'PASS', 'cpu_seconds': out['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        path = ROOT / 'VERIFICATION.json'
        partial = json.loads(path.read_text()) if path.exists() else {}
        partial.update(status='FAILED', failure_traceback=traceback.format_exc(),
                       scope='Overall verification failed; nested successful checks are partial only.')
        path.write_text(json.dumps(partial, indent=2) + '\n')
        raise
