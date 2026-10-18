"""Independent readout checks plus all preserved rule, cap, family and FIFO checks."""
import family_verify as V
from family_verify import C, torch, json, time, ROOT
import traceback


def verify_readouts(seed, dtype, tol):
    gen = torch.Generator().manual_seed(seed + 90000000)
    initial = C.bounded(C.init(seed, dtype))
    errors, capmax = {}, 0.
    for case, gain in (('bounded', 1.), ('projection_active', 1.5)):
        x = torch.randn(13, C.D, generator=gen, dtype=dtype)
        y = torch.randint(0, 2, (13, C.R), generator=gen).to(dtype)
        for policy, switches in C.READOUT_SWITCHES.items():
            state = C.clone(initial)
            active = C.READOUT_KEYS[policy]
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
            new = C.readout_step(state, x, y, *switches)
            observed = C.decode(new)
            errors[case + '_' + policy] = max(float((observed[k] - direct[k]).abs().max()) for k in direct)
            assert errors[case + '_' + policy] <= tol, (seed, dtype, errors)
            frozen, _, _ = C.FAM.policy_step(state, x, y, 'frozen', shadow={})
            assert all(torch.equal(new[k], frozen[k]) for k in C.WK)
            assert all(torch.equal(new[k], state[k]) for k in C.FK if k not in active)
            assert all(torch.equal(state[k], snapshot[k]) for k in state)
            assert all(not v.requires_grad and v.grad_fn is None for v in new.values())
            capmax = max(capmax, max(float(torch.linalg.svdvals(observed[k])[0]) for k in C.FK))
            assert capmax <= 2.
            assert all(not torch.equal(new[k], state[k]) for k in active)
            if policy == 'frozen':
                assert all(torch.equal(new[k], frozen[k]) for k in new)
            if policy == 'both_u':
                ref, _, _ = C.FAM.policy_step(state, x, y, 'vu_only', shadow={})
                assert all(torch.equal(new[k], ref[k]) for k in (*C.WK, *C.B_KEYS, 'U0', 'U1'))
    trajectories = {name: C.clone(initial) for name in C.READOUT_SWITCHES}
    frozen, vu = C.clone(initial), C.clone(initial)
    for _ in range(12):
        x = torch.randn(7, C.D, generator=gen, dtype=dtype)
        y = torch.randint(0, 2, (7, C.R), generator=gen).to(dtype)
        for name, switches in C.READOUT_SWITCHES.items():
            trajectories[name] = C.readout_step(trajectories[name], x, y, *switches)
            assert all(torch.equal(trajectories[name][k], initial[k])
                       for k in C.FK if k not in C.READOUT_KEYS[name])
        frozen, _, _ = C.FAM.policy_step(frozen, x, y, 'frozen', shadow={})
        vu, _, _ = C.FAM.policy_step(vu, x, y, 'vu_only', shadow={})
        assert all(torch.equal(trajectories['frozen'][k], frozen[k]) for k in frozen)
        assert all(torch.equal(trajectories['both_u'][k], vu[k]) for k in (*C.WK, *C.B_KEYS, 'U0', 'U1'))
    assert all(not torch.equal(vu[k], initial[k]) for k in ('V1', 'V2'))
    # U1-only still updates W0 using its fixed U0 decoder.
    assert all(not torch.equal(trajectories['u1_only'][k], initial[k]) for k in C.WK)
    for args in ((1, False), (False, 1)):
        try:
            C.readout_step(initial, x, y, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonboolean switch accepted')
    old = torch.autograd.grad
    def forbidden(*args, **kwargs):
        raise AssertionError('Autograd entered local readout rule')
    try:
        torch.autograd.grad = forbidden
        for name in C.READOUT_SWITCHES:
            C.policy_step(initial, x, y, name, shadow={})
    finally:
        torch.autograd.grad = old
    return {'seed': seed, 'dtype': str(dtype), 'tolerance': tol, 'errors': errors,
            'maximum_capped_sigma': capmax, 'inactive_matrices_bitwise_frozen_12steps': True,
            'same_state_forward_updates_bitwise': True, 'frozen_endpoint_12step_trajectory_bitwise': True,
            'both_U_vs_preserved_VU_W_U_B_steps_and_12step_trajectory_bitwise': True,
            'V_fixed_in_both_U_and_changed_in_reference': True, 'U1_only_all_forward_layers_update': True,
            'input_unmodified_output_detached': True, 'all_local_paths_without_autograd': True,
            'active_readout_nonzero_update_exercised': True, 'nonboolean_switches_rejected': True}


def main():
    begin = time.process_time()
    V.main()
    out = json.loads((ROOT / 'VERIFICATION.json').read_text())
    out['feedback_readouts'] = [verify_readouts(seed, dtype, tol) for seed in (102937, 17837, 26837)
                               for dtype, tol in ((torch.float32, 2e-6), (torch.float64, 2e-12))]
    out.update(status='PASS', cpu_seconds=time.process_time() - begin)
    (ROOT / 'VERIFICATION.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'feedback_readout_verification': 'PASS', 'cpu_seconds': out['cpu_seconds']}), flush=True)


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
