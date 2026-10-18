# W560 Continuous Canonical MPS Optimizer

This package is the live Mac fallback for the recovered generation-7 W560
checkpoint. It replaces CPU-surrogate shortlist selection with one numerical
lane for both proposal search and promotion: MPS BF16 model forwards with CPU
FP64 functional normalization.

## Live run

- Runtime: `runtime`
- Config: `runtime/config/controller.json`
- Append-only ledger: `runtime/live/ledger`
- Continuous run: `runtime/live/run-20260822-attempt1`
- Initial generation: 7
- Initial checkpoint SHA-256:
  `04830cf33045b19dadb610873770b97e8d0cdb1446b5bccf96389f35b35b76fd`
- Initial canonical LOCAL_VERIFY CE: `6.222062796354294`
- Runner PID at launch: `99688`
- Execution session: `79530`
- Launch observed live at: `2026-08-22T05:42:40Z`

The runner is intentionally unbounded. Each job rotates through
`base_affines`, `left_factors`, `right_factors`, and `codecs`; saves every
provisionally accepted intermediate candidate; evaluates every saved candidate
twice on canonical LOCAL_VERIFY; appends at most one winning checkpoint/head;
and allocates the successor only after the prior decision is complete.

The first live cycle completed all 18 search pairs, saved and duplicated all 14
accepted candidates, rejected them because none passed the canonical threshold,
and automatically started the successor `left_factors` job. That proves the
continuous rejection-to-successor path against the recovered checkpoint.

No cloud access, backpropagation, Adam, scheduler, or training-resume capability
is present. LOCAL_VERIFY is reused optimization data, not held-out
generalization.

## Read-only status

```bash
cd "/Volumes/SP AI 01_16/hfw-w560-continuous-mps-cumulative-optimizer-20260822-v2-attempt1/runtime"
/usr/bin/python3 -B -m hfw_continuous.local_cli status \
  --config config/controller.json \
  --ledger live/ledger
```

Progress is append-only under `runtime/live/run-20260822-attempt1/attempts` and
authority records are under `runtime/live/ledger`.
