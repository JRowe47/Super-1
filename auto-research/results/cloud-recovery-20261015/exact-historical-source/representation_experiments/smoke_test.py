"""CPU smoke tests for causality, shapes, budgets, and manifest validity."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from representation_experiments.atlas_v2_attention import AtlasV2Bank
from representation_experiments.operator_attention import (
    AtlasOperator,
    FourierLocalOperator,
    PositiveKernelOperator,
    WaveletRBFOperator,
)
from representation_experiments.predictive_atlas import (
    behavioral_fingerprint,
    hash_projection,
)
from representation_experiments.predictive_models import build_to_budget


def check_operator(module, hidden):
    module.eval()
    original = module(hidden, depth=0.3)
    changed = hidden.clone()
    changed[:, -1] += 100
    future_changed = module(changed, depth=0.3)
    assert original.shape == hidden.shape
    assert torch.isfinite(original).all()
    assert torch.allclose(original[:, :-1], future_changed[:, :-1], atol=2e-4, rtol=2e-4), \
        module.kind
    assert module.estimated_flops(hidden.shape[1]) > 0


def main():
    torch.manual_seed(7)
    hidden = torch.randn(2, 16, 32)
    operators = [
        PositiveKernelOperator(32, 8),
        FourierLocalOperator(32, 8, 8, 4, 4),
        WaveletRBFOperator(32, 8, 4, 3),
        AtlasOperator(32, 4, 2, 8, 8, 4, 3, 4),
    ]
    for operator in operators:
        check_operator(operator, hidden)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        check_operator(FourierLocalOperator(32, 8, 8, 4, 4), hidden)

    atlas_v2 = AtlasV2Bank(
        hidden_size=32, heads=4, charts=8, basis=2, topk_charts=2,
        value_dim=8, local_features=4, local_windows=[2, 4], latent=8,
        wavelet_scales=3, depth_basis=4)
    check_operator(atlas_v2, hidden)
    auxiliary = atlas_v2.auxiliary_losses()
    assert set(auxiliary) == {"chart_balance", "mix_balance"}
    assert all(torch.isfinite(value) for value in auxiliary.values())

    buckets, signs = hash_projection(64, 16, 11)
    first = torch.randn(2, 4, 64)
    second = torch.randn(2, 4, 64)
    fingerprint = behavioral_fingerprint(first, second, buckets, signs, 8, 16)
    assert fingerprint.shape == (2, 4, 38)

    base = {"minimum_width": 64, "maximum_width": 128}
    specs = {
        "transformer": {**base, "layers": 2, "heads": 8, "expansion": 2},
        "recurrent_logits": {**base, "layers": 2},
        "hierarchical_predictive": {
            **base, "atoms": 32, "segment_length": 4,
            "document_segments": 2, "future_horizons": [1, 2],
        },
    }
    for kind, spec in specs.items():
        budgeted = build_to_budget(kind, 256, 16, spec, 400_000)
        tokens = torch.randint(0, 256, (2, 16))
        output = budgeted.model(tokens)
        assert output.logits.shape == (2, 16, 256)
        assert budgeted.parameters <= 400_000
        assert budgeted.model.stored_state_elements(2, 16) > 0

    manifest_path = Path(__file__).parent / "configs" / "qwen06b_representation_studies.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["arms"]) == {
        "positive_kernel", "fourier_local", "wavelet_rbf", "atlas"}
    assert set(manifest["students"]) == {
        "transformer", "recurrent_logits", "hierarchical_predictive",
        "hierarchical_logit_control"}
    atlas_v2_path = Path(__file__).parent / "configs" / "qwen06b_atlas_v2.json"
    atlas_v2_manifest = json.loads(atlas_v2_path.read_text())
    assert atlas_v2_manifest["atlas_v2"]["groups"] == 4
    assert atlas_v2_manifest["milestones_tokens"] == [10_000_000, 50_000_000]
    print("representation_experiments smoke test: PASS")


if __name__ == "__main__":
    main()
