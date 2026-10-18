# Learned layer count and sampling positions for text

The learned quality-only depth selector chooses four executed layers on all three seeds. The library now also preserves explicit two- and three-layer profiles with measured quality and CPU latency. These profiles are opt-in experiments; none replaces the retained text reference.

The sampler learns interior curve coordinates per token. A separate three-logit categorical selector learns a workload-level preference for two, three or four executed layers by gradient descent on expected prediction loss. Training evaluates all depth branches on identical examples. At inference only the argmax branch executes. This is workload-level depth selection, not token-wise halting.

| Executed layers | Seed73 CE | Seed191 CE | Seed307 CE | Mean CE (nats/byte) | Mean CPU prefill ms |
|---|---:|---:|---:|---:|---:|
| 2 | 2.661193512 | 2.680108145 | 2.488886945 | 2.610062867 | 0.678875167 |
| 3 | 2.225181051 | 2.205833256 | 2.229000255 | 2.220004854 | 1.026083333 |
| 4 | 2.211858973 | 2.200646028 | 2.229459986 | 2.213988329 | 1.385396000 |

Three layers have 25.94% lower measured CPU prefill time, with 0.006016525 higher mean CE than four. Two layers lose 0.396074538 nats/byte. These are descriptive timings from one CPU thread, FP32, batch1, length64, twenty repeated forwards per model; no GPU, caching, generation throughput or end-to-end speed guarantee.

Learned final depth probabilities, in order [2,3,4]:

- Seed 73: [7.34883442987666e-08, 9.588367220203509e-07, 0.999998927116394]; selected depth 4, validation round 3.
- Seed 191: [6.892938131386472e-08, 2.326609774172539e-06, 0.999997615814209]; selected depth 4, validation round 2.
- Seed 307: [9.482778295932803e-08, 7.250583621498663e-06, 0.9999926090240479]; selected depth 4, validation round 3.

The validation-selected policy improves the old four-step reference by 0.001626646 nats/byte on average; paired95 change interval [-0.00675646966321578, 0.0035031786126227225]. All paired changes are negative, but the interval spans zero and the gain misses the fixed0.02-nat threshold. **No quality replacement is promoted.**

Fresh confirmation bytes: [1053774,1070414), 256 disjoint65-byte windows. These scores must not be compared directly with earlier runs scored on different regions. Each seed receives600 tuning minibatches for every depth; the backbone and all normalization parameters remain frozen. Depth2 has no interior coordinate and therefore no coordinate optimizer updates. Depth3 has one learned interior coordinate; depth4 has two. Lower depths use a prefix of the original frozen normalization blocks with curve coordinates spanning0..1; the backbone was trained at depth4, so reduced-depth distribution shift limits the interpretation.

Validation freezes all choices before confirmation. All15 endpoints replay exactly, all9 depth profiles reload exactly, and execution hooks confirm only the selected blocks run. Five negative loader checks reject wrong depth, wrong backbone and wrong source hashes. The canary passes four-step forward/optimizer parity and a three-step FP64 coordinate finite difference with2.13e-11 absolute error. Training preserved36 router checkpoints and12 depth-selector checkpoints, using47.49 CPU seconds /48.98 wall seconds.

The full frozen backbone remains stored for every profile. The library measures executed layers separately from stored parameters. A quality-only objective naturally favors more compute here; no arbitrary cost penalty was added to force a smaller-depth winner. Future workloads need their own training/evaluation evidence.

Files: [library](../../sampling_library/README.md), [profiles](../../sampling_library/depth_profiles.json), [protocol](../results/text-depth-library-20261018-v1/PROTOCOL.json), [confirmation](../results/text-depth-library-20261018-v1/execution/RESULT.json), [replay](../results/text-depth-library-20261018-v1/execution/VERIFICATION.json).

Reusable checks: `python3 -B -m sampling_library.test_library` and `python3 -B -m sampling_library.test_profiles`. The frozen study checker writes a one-time receipt, preserving its original result rather than overwriting it.
