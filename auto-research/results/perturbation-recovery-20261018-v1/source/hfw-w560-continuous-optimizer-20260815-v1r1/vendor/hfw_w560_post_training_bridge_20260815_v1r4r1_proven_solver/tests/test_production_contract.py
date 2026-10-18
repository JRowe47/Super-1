from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import time
from unittest import mock
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.artifacts import (
    write_content_addressed_json,
)

from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.branch_search import (
    _functional_effective_rms,
    run_independent_branch,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.data_commitment import (
    load_committed_batches,
    resolve_public_calibration_layout,
    resolve_validation_corpus,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.manifest import (
    derive_group_manifest,
    file_sha256,
    load_protocol,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.parent_gate import (
    validate_immutable_parent,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.production_bridge import (
    LoadedW560Parent,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.production_runner import (
    execute_report_only_fixed_eval,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.production_phases import (
    _load_search_manifest, _validate_final_cuda_isolation,
    authority_finalize_core, run_group_worker_core,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.production_entrypoint import (
    _validate_runtime_root_argument,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.sealed_evaluation_bridge import (
    CanonicalProposalRegistry,
    create_production_search_capability,
    load_canonical_sealed_evaluation,
    load_canonical_sealed_evaluation_v2,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.static_gate import (
    validate_no_training_source_gate,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.runtime_guard import (
    NoTrainingRuntimeGuard,
)
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.power_sampler import (
    PeriodicGpuPowerSampler, integrate_power_samples_joules,
    power_sampler_from_protocol, query_gpu_power_draw_watts,
)


PACKAGE = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol(PACKAGE / "config/production_protocol.json")


@contextmanager
def raises(expected):
    try:
        yield
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def test_exact_meta_manifest_matches_canonical_binding() -> None:
    source = Path(PROTOCOL["frozen_source"]["local_review_root"])
    manifest = derive_group_manifest(PROTOCOL, source)
    assert manifest["total_parameters"] == 99_874_320
    assert manifest["group_counts"] == {
        "base_affines": 10_046_960,
        "left_factors": 1_703_520,
        "right_factors": 2_751_840,
        "codecs": 286_720,
    }
    assert manifest["frozen_count"] == 85_085_280
    assert manifest["canonical_weight_address_binding"]["manifest_sha256"] == (
        "fb1b77f23305995ed5546354274df60f59e2f5d8f91f8273095cf644387286d6"
    )


def test_parent_gate_accepts_exact_fixture_and_rejects_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"immutable completed checkpoint fixture")
    checkpoint_sha = file_sha256(checkpoint)
    review = tmp_path / "review.json"
    exact_model_state_digest = (
        "4143d8cb3729998bd0dc9ded907ba8756ba5a8f8b49335f8185b055f3aac3d0b"
    )
    exact_checkpoint_identity = "1" * 64
    review_body = {
        "schema": "review_stop_checkpoint_publication_receipt_v1",
        "step": 3052,
        "tokens": 25_001_984,
        "training_resume_permanently_forbidden": True,
        "checkpoint_identity_sha256": exact_checkpoint_identity,
        "checkpoint_publication": {
            "sha256": checkpoint_sha,
            "control": {"model_state_digest": exact_model_state_digest},
        },
    }
    review.write_text(json.dumps(review_body), encoding="utf-8")
    info = checkpoint.stat()
    protocol = {
        "parent": {
            "checkpoint_path": str(checkpoint),
            "review_receipt_path": str(review),
            "size_bytes": info.st_size,
            "inode": info.st_ino,
            "minimum_link_count": 1,
            "checkpoint_sha256": checkpoint_sha,
            "review_receipt_sha256": file_sha256(review),
            "review_schema": review_body["schema"],
            "completed_step": 3052,
            "completed_tokens": 25_001_984,
            "checkpoint_identity_sha256": exact_checkpoint_identity,
            "model_state_digest": exact_model_state_digest,
        }
    }
    assert validate_immutable_parent(protocol)["status"] == "PASS_IMMUTABLE_COMPLETED_PARENT"
    assert len(protocol["parent"]["model_state_digest"]) == 64
    protocol["parent"]["model_state_digest"] = exact_model_state_digest[:-1]
    with raises(RuntimeError):
        validate_immutable_parent(protocol)
    protocol["parent"]["model_state_digest"] = exact_model_state_digest
    checkpoint.write_bytes(b"mutated")
    with raises(RuntimeError):
        validate_immutable_parent(protocol)


def test_fresh_data_commitment_is_pairwise_disjoint_and_replayable(tmp_path: Path) -> None:
    data_root = tmp_path / "source"
    data_path = data_root / "data" / "val.bin"
    data_path.parent.mkdir(parents=True)
    np.arange(250_000, dtype=np.uint32).tofile(data_path)
    protocol = {
        "data": {
            "relative_path": "data/val.bin",
            "sha256": file_sha256(data_path),
            "dtype": "uint32",
            "context": 32,
            "batch_size": 2,
            "calibration_batches": 12,
            "fresh_fixed_eval_batches": 16,
            "prior_public_seed": 33001,
            "calibration_seed": 73001,
        }
    }
    commitment = resolve_public_calibration_layout(protocol, data_root)
    intervals = []
    assert "fresh_fixed_eval" not in commitment
    assert commitment["fresh_fixed_eval_offsets_present"] is False
    assert commitment["fixed_eval_seed_present"] is False
    for section, records in (
        ("prior_public_eval_exclusion", commitment["prior_public_eval_exclusion"]),
        ("calibration", commitment["calibration"]),
    ):
        for record in records:
            intervals.extend((section, tuple(value)) for value in record["intervals"])
    for index, (first_section, first) in enumerate(intervals):
        for second_section, second in intervals[index + 1:]:
            if first_section == second_section == "prior_public_eval_exclusion":
                continue
            assert not (first[0] < second[1] and second[0] < first[1])
    replay = list(load_committed_batches(protocol, data_root, commitment, "calibration"))
    assert len(replay) == 12 and replay[0].shape == (2, 33)


def test_source_and_runtime_roots_are_distinct_and_swap_alias_drift_fail(tmp_path: Path) -> None:
    source_root = tmp_path / "source_only"
    runtime_root = tmp_path / "runtime_only"
    source_root.mkdir()
    validation = runtime_root / "data" / "val.bin"
    validation.parent.mkdir(parents=True)
    np.arange(100_000, dtype=np.uint32).tofile(validation)
    protocol = {
        "runtime": {"runtime_root": str(runtime_root)},
        "data": {
            "relative_path": "data/val.bin",
            "sha256": file_sha256(validation),
            "size_bytes": validation.stat().st_size,
            "inode": validation.stat().st_ino,
            "minimum_link_count": 1,
            "dtype": "uint32",
            "context": 16,
            "batch_size": 2,
            "calibration_batches": 2,
            "fresh_fixed_eval_batches": 2,
            "prior_public_seed": 33001,
            "calibration_seed": 73001,
        },
    }
    commitment = resolve_public_calibration_layout(protocol, runtime_root)
    assert commitment["runtime_root"] == str(runtime_root)
    assert commitment["data_relative_path"] == "data/val.bin"
    assert not (source_root / "data").exists()
    with raises(FileNotFoundError):
        resolve_public_calibration_layout(protocol, source_root)
    alias = tmp_path / "runtime_alias"
    alias.symlink_to(runtime_root, target_is_directory=True)
    with raises(RuntimeError):
        _validate_runtime_root_argument(protocol, alias)
    with raises(RuntimeError):
        load_committed_batches(protocol, alias, commitment, "calibration").__next__()
    validation.write_bytes(validation.read_bytes()[:-4] + b"\x00\x00\x00\x00")
    with raises(RuntimeError):
        resolve_validation_corpus(protocol, runtime_root)


def test_induced_effective_rms_matches_materialized_definition() -> None:
    generator = torch.Generator().manual_seed(7)
    design = torch.randn(5, 3, generator=generator)
    left = torch.randn(3, 4, 2, generator=generator)
    right = torch.randn(3, 2, 6, generator=generator)
    materialized = torch.einsum("dc,cir,cro->dio", design, left, right)
    actual = _functional_effective_rms(design, left, right)
    assert torch.allclose(actual, materialized.square().mean().sqrt(), rtol=1e-5, atol=1e-6)


class _TinyFunctionalAffine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.control_left = torch.nn.Parameter(torch.randn(2, 5, 2, generator=generator))
        self.control_right = torch.nn.Parameter(torch.randn(2, 2, 4, generator=generator))
        self.register_buffer("design", torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]))


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _TinyFunctionalAffine()


class _LinearSyntheticObjective:
    def __init__(self) -> None:
        model = _TinyModel().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.parent = SimpleNamespace(model=model, groups={"left_factors": ("block.control_left",)})
        self.base_parameters = {name: value.detach() for name, value in model.named_parameters()}
        self.buffers = {name: value.detach() for name, value in model.named_buffers()}
        self._preferred = torch.linspace(-1, 1, self.base_parameters["block.control_left"].numel())

    def flat_group(self, group: str) -> torch.Tensor:
        assert group == "left_factors"
        return self.base_parameters["block.control_left"].reshape(-1).detach()

    def loss(self, group: str, flat: torch.Tensor, batch_index: int) -> float:
        assert not flat.requires_grad
        return float(-(flat.float() * self._preferred).mean() + batch_index * 1e-6)

    def loss_vector(self, group: str, flat: torch.Tensor, batch_index: int):
        value = self.loss(group, flat, batch_index)
        return (value, value)


def test_independent_branch_is_forward_only_append_only_and_parent_immutable(tmp_path: Path) -> None:
    objective = _LinearSyntheticObjective()
    before = objective.flat_group("left_factors").clone()
    protocol = json.loads(json.dumps(PROTOCOL))
    protocol["search"]["independent_branches"] = ["left_factors"]
    canonical, canonical_v2 = load_canonical_sealed_evaluation_v2(protocol, PACKAGE)
    capability = canonical.create_eval_commitment(
        authority_root=tmp_path / "authority",
        public_root=tmp_path / "public",
        validation_sha256="b" * 64,
        validation_token_count=100_000,
        evaluation_offsets=[50_000],
        sample_length_tokens=513,
        salt=bytes(range(32)),
    )
    calibration_sha = canonical.create_calibration_manifest(
        capability,
        validation_sha256="b" * 64,
        validation_token_count=100_000,
        calibration_offsets=[10_000, 20_000],
        sample_length_tokens=513,
    )
    aliases = protocol["canonical_weight_address_binding"]["internal_to_canonical_group"]
    allowed = tuple(aliases[name] for name in protocol["search"]["independent_branches"])
    registry = CanonicalProposalRegistry(
        canonical_v2, capability, protocol["parent"]["checkpoint_sha256"],
        calibration_sha, "left_factors", aliases, allowed,
    )
    result = run_independent_branch(
        objective, protocol, "left_factors", tmp_path, registry, materialize=True
    )
    assert result.proposals == 18
    assert result.accepts == 18
    assert result.receipt["forward_calls"] == 180
    assert result.receipt["optimizer_steps"] == 0
    assert result.receipt["autograd_calls"] == 0
    assert result.receipt["fixed_eval_opened"] is False
    assert torch.equal(before, objective.flat_group("left_factors"))
    delta_files = list((tmp_path / "left_factors").glob("accepted_*_delta.f32le"))
    assert len([path for path in delta_files if "incremental" not in path.name]) == 18
    assert len([path for path in delta_files if "incremental" in path.name]) == 18
    chain = canonical.validate_proposal_chain(capability)
    assert chain["event_count"] == 18
    assert chain["accepted"] == 18 and chain["rejected"] == 0
    child = torch.load(
        tmp_path / "left_factors" / "materialized_child_18.pt",
        map_location="cpu", weights_only=True,
    )
    reloaded = _TinyModel()
    reloaded.load_state_dict(child["model"], strict=True)
    assert "block.design" in child["model"]
    child_one_sha = file_sha256(tmp_path / "left_factors" / "materialized_child_17.pt")
    assert child["immutable_parent_checkpoint_sha256"] == protocol["parent"]["checkpoint_sha256"]
    assert child["immediate_parent_child_sha256"] == child_one_sha
    with raises(FileExistsError):
        run_independent_branch(objective, protocol, "left_factors", tmp_path, registry, materialize=True)


class _FourGroupToyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
        self.left = torch.nn.Parameter(torch.tensor([0.1, 0.3]))
        self.right = torch.nn.Parameter(torch.tensor([-0.4, 0.2]))
        self.codec = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
        self.register_buffer("design", torch.tensor([1.0]))

    def forward(self, input_ids: torch.Tensor):
        logits = torch.cat((self.base, self.left, self.right, self.codec))
        logits = logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


class _FourGroupSearchObjective:
    def __init__(self, model: _FourGroupToyLM, groups):
        self.parent = SimpleNamespace(model=model, groups=groups)
        self.base_parameters = {name: value.detach() for name, value in model.named_parameters()}
        self.buffers = {name: value.detach() for name, value in model.named_buffers()}
        self.preferred = {
            group: torch.linspace(-1.0, 1.0, self.base_parameters[names[0]].numel())
            for group, names in groups.items()
        }

    def flat_group(self, group: str):
        return torch.cat([self.base_parameters[name].reshape(-1) for name in self.parent.groups[group]])

    def loss(self, group: str, flat: torch.Tensor, batch_index: int):
        return float(-(flat.float() * self.preferred[group]).mean() + batch_index * 1e-6)

    def loss_vector(self, group: str, flat: torch.Tensor, batch_index: int):
        value = self.loss(group, flat, batch_index)
        return (value, value)


def test_production_v2_flow_exact_four_then_parent_plus_four_final_eval(tmp_path: Path) -> None:
    protocol = json.loads(json.dumps(PROTOCOL))
    source = tmp_path / "data_source"
    validation = source / "data" / "val.bin"
    validation.parent.mkdir(parents=True)
    (np.arange(40_000, dtype=np.uint32) % 8).tofile(validation)
    protocol["runtime"].update({
        "runtime_root": str(source),
        "validation_path": str(validation),
    })
    protocol["data"].update({
        "relative_path": "data/val.bin",
        "sha256": file_sha256(validation),
        "size_bytes": validation.stat().st_size,
        "inode": validation.stat().st_ino,
        "context": 8,
        "batch_size": 2,
        "calibration_batches": 2,
        "fresh_fixed_eval_batches": 2,
    })
    calibration = resolve_public_calibration_layout(protocol, source)
    authority_root = tmp_path / "private_authority"
    public_root = tmp_path / "public_search"
    canonical, canonical_v2, authority, capability, calibration_sha = (
        create_production_search_capability(
            protocol, PACKAGE,
            authority_root=authority_root,
            public_root=public_root,
            calibration_layout=calibration,
            salt=bytes(range(32)),
        )
    )
    public_payload = json.dumps(capability.public_payload())
    for forbidden in protocol["canonical_sealed_evaluation_binding"]["search_capability_must_exclude"]:
        assert forbidden not in public_payload
    groups = {
        "base_affines": ("base",),
        "left_factors": ("left",),
        "right_factors": ("right",),
        "codecs": ("codec",),
    }
    model = _FourGroupToyLM().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    objective = _FourGroupSearchObjective(model, groups)
    output = tmp_path / "search_output"
    output.mkdir()
    aliases = protocol["canonical_weight_address_binding"]["internal_to_canonical_group"]
    allowed = tuple(aliases[group] for group in protocol["search"]["independent_branches"])
    finals = []
    for group in protocol["search"]["independent_branches"]:
        registry = CanonicalProposalRegistry(
            canonical_v2, capability, protocol["parent"]["checkpoint_sha256"],
            calibration_sha, group, aliases, allowed,
        )
        run_independent_branch(objective, protocol, group, output, registry, materialize=True)
        finals.append(registry.branch_final())
    chain = canonical.validate_proposal_chain(capability)
    assert chain["event_count"] == 72
    authorization_v2 = canonical_v2.seal_reveal_authorize_four_branch_final_eval(
        authority,
        parent_checkpoint_sha256=protocol["parent"]["checkpoint_sha256"],
        calibration_manifest_sha256=calibration_sha,
        required_groups=allowed,
        branch_finals=finals,
        terminal_status="LOCAL_PROXY_ALL_FOUR_COMPLETE",
        validation_path=validation,
    )
    assert authorization_v2["status"] == "AUTHORIZED"
    reveal = canonical.read_public_reveal(capability)
    loaded = LoadedW560Parent(
        model=model,
        groups=groups,
        parent_gate={
            "status": "PASS_IMMUTABLE_COMPLETED_PARENT",
            "checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
        },
        group_manifest={},
        checkpoint_sha256=protocol["parent"]["checkpoint_sha256"],
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    fixed = execute_report_only_fixed_eval(
        protocol, loaded,
        authorization={
            "status": "PASS_FIXED_EVAL_MAY_NOW_OPEN",
            "all_four_branches_complete": True,
            "evaluation_offsets": reveal["evaluation_offsets"],
            "sample_length_tokens": reveal["sample_length_tokens"],
            "bound_branch_finals": {final["group"]: final for final in finals},
        },
        validation_path=validation,
        search_root=output,
        device=torch.device("cpu"),
    )
    assert fixed["status"] == "PASS_FIXED_EVAL_REPORTED_NO_SELECTION"
    assert fixed["arms_evaluated"] == ["immutable_parent", *protocol["search"]["independent_branches"]]
    assert len(fixed["results"]) == 5
    assert set(fixed["paired_branch_vs_parent"]) == set(protocol["search"]["independent_branches"])
    finals_by_group = {final["group"]: final for final in finals}
    for group in protocol["search"]["independent_branches"]:
        canonical_group = aliases[group]
        assert fixed["results"][group]["evaluated_exact_state_artifact_sha256"] == (
            finals_by_group[canonical_group]["final_checkpoint_sha256"]
        )
        assert fixed["results"][group]["evaluation_source"] == (
            "same_fd_loaded_terminal_bound_materialized_child"
        )
    assert fixed["branch_selected_or_combined"] is False
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    registry_root = tmp_path / "final_registry"
    artifact = write_content_addressed_json(registry_root, ".result.json", fixed)
    assert Path(artifact["path"]).is_file()
    with raises(FileExistsError):
        write_content_addressed_json(registry_root, ".result.json", fixed)
    value = torch.tensor(2.0, requires_grad=True)
    (value.square()).backward()
    assert value.grad.item() == 4.0


def test_authority_prepare_phase_exits_before_search_reads_capability(tmp_path: Path) -> None:
    protocol = json.loads(json.dumps(PROTOCOL))
    source = tmp_path / "authority_data"
    validation = source / "data" / "val.bin"
    validation.parent.mkdir(parents=True)
    (np.arange(50_000, dtype=np.uint32) % 8).tofile(validation)
    protocol["runtime"].update({
        "runtime_root": str(source),
        "validation_path": str(validation),
    })
    protocol["data"].update({
        "relative_path": "data/val.bin",
        "sha256": file_sha256(validation),
        "size_bytes": validation.stat().st_size,
        "inode": validation.stat().st_ino,
        "context": 8,
        "batch_size": 2,
        "calibration_batches": 2,
        "fresh_fixed_eval_batches": 2,
    })
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    authority = tmp_path / "authority_process_only"
    public = tmp_path / "search_process_only"
    script = """
import json, sys
from pathlib import Path
from funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.production_phases import authority_prepare_core
p=json.loads(Path(sys.argv[1]).read_text())
r=authority_prepare_core(p, Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), authority_root=Path(sys.argv[5]), public_root=Path(sys.argv[6]))
print(json.dumps(r, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(protocol_path), str(PACKAGE), str(PACKAGE), str(source), str(authority), str(public)],
        cwd=PACKAGE.parents[1], check=True, capture_output=True, text=True,
        env={**dict(__import__("os").environ), "PYTHONPYCACHEPREFIX": "/tmp/hfw_authority_phase_pyc"},
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    public_artifact = result["public_search_manifest"]
    manifest = _load_search_manifest(
        Path(public_artifact["path"]), public_artifact["sha256"], protocol, PACKAGE
    )
    assert int(manifest["authority_process_pid"]) != __import__("os").getpid()
    visible = json.dumps(manifest, sort_keys=True)
    for forbidden in ("authority_root", "generation_record_sha256", "private_entropy", "evaluation_offsets", "salt_hex"):
        assert forbidden not in visible
    private_files = list(authority.rglob("*.generation.json")) + list(authority.rglob("*.reveal.json"))
    assert private_files and all(path.stat().st_mode & 0o777 == 0o400 for path in private_files)


def test_static_gate_has_no_training_calls(tmp_path: Path) -> None:
    del tmp_path
    gate = validate_no_training_source_gate(PACKAGE)
    assert gate["status"] == "PASS_NO_AUTOGRAD_NO_OPTIMIZER_SOURCE"


def test_runtime_guard_blocks_optimizer_construction_and_restores_runtime() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with NoTrainingRuntimeGuard() as guard:
        with raises(RuntimeError):
            torch.optim.SGD([parameter], lr=0.1)
    assert guard.receipt()["optimizer_construction_attempts"] == 1
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    assert optimizer is not None


def test_power_draw_query_uses_exact_physical_gpu_cli_decimal_and_timeout() -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="2, 65.35\n", stderr="")
    with mock.patch(
        "funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.power_sampler.subprocess.run",
        return_value=completed,
    ) as invoked:
        assert query_gpu_power_draw_watts(2, 7.5) == 65.35
    assert invoked.call_args.args[0] == [
        "nvidia-smi", "-i", "2", "--query-gpu=index,power.draw",
        "--format=csv,noheader,nounits",
    ]
    assert invoked.call_args.kwargs["timeout"] == 7.5


def test_power_draw_query_fails_on_rc2_timeout_malformed_and_index_drift() -> None:
    target = "funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.power_sampler.subprocess.run"
    with mock.patch(target, side_effect=subprocess.CalledProcessError(2, ["nvidia-smi"])):
        with raises(subprocess.CalledProcessError):
            query_gpu_power_draw_watts(0, 1.0)
    with mock.patch(target, side_effect=subprocess.TimeoutExpired(["nvidia-smi"], 1.0)):
        with raises(subprocess.TimeoutExpired):
            query_gpu_power_draw_watts(0, 1.0)
    for stdout in ("0, N/A\n", "1, 65.35\n", "0, nan\n", "0, -1.0\n", "0, 1.0\n1, 2.0\n"):
        completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
        with mock.patch(target, return_value=completed):
            with raises(RuntimeError):
                query_gpu_power_draw_watts(0, 1.0)


def test_power_samples_integrate_monotonic_trapezoids_in_joules() -> None:
    samples = [
        {"monotonic_seconds": 10.0, "power_watts": 10.0},
        {"monotonic_seconds": 12.0, "power_watts": 20.0},
        {"monotonic_seconds": 15.0, "power_watts": 10.0},
    ]
    assert integrate_power_samples_joules(samples) == 75.0
    with raises(RuntimeError):
        integrate_power_samples_joules([samples[0], dict(samples[0])])
    with raises(RuntimeError):
        integrate_power_samples_joules(samples[:1])
    with raises(RuntimeError):
        PeriodicGpuPowerSampler(
            0, 1.0, 1, query_timeout_seconds=1.0,
            query_power=lambda index: 1.0,
        )


def test_periodic_power_sampler_receipt_and_body_failure_cleanup() -> None:
    clock_values = iter((10.0, 12.0))
    sampler = PeriodicGpuPowerSampler(
        3, 60.0, 2, query_timeout_seconds=1.0,
        query_power=lambda index: 10.0 if index == 3 else -1.0,
        monotonic_clock=lambda: next(clock_values),
    )
    with sampler:
        pass
    receipt = sampler.receipt()
    assert receipt["sample_count"] == 2
    assert receipt["integrated_energy_joules"] == 20.0
    assert receipt["physical_cuda_index"] == 3
    assert receipt["power_units"] == "watts" and receipt["energy_units"] == "joules"
    assert receipt["background_thread_stopped"] is True and not sampler.thread_alive

    failure_clock = iter((20.0, 21.0))
    failed = PeriodicGpuPowerSampler(
        1, 60.0, 2, query_timeout_seconds=1.0,
        query_power=lambda index: 5.0,
        monotonic_clock=lambda: next(failure_clock),
    )
    with raises(ValueError):
        with failed:
            raise ValueError("worker failed")
    assert not failed.thread_alive
    assert failed.receipt()["sample_count"] == 2


def test_periodic_power_sampler_propagates_background_and_dual_failure_without_leak() -> None:
    calls = {"count": 0}
    def query(index: int) -> float:
        calls["count"] += 1
        if calls["count"] == 1:
            return 5.0
        raise RuntimeError("query failed")
    sampler = PeriodicGpuPowerSampler(
        0, 0.001, 2, query_timeout_seconds=0.1,
        query_power=query, monotonic_clock=time.monotonic,
    )
    with raises(RuntimeError):
        with sampler:
            time.sleep(0.01)
    assert not sampler.thread_alive

    calls["count"] = 0
    dual = PeriodicGpuPowerSampler(
        0, 0.001, 2, query_timeout_seconds=0.1,
        query_power=query, monotonic_clock=time.monotonic,
    )
    try:
        with dual:
            time.sleep(0.01)
            raise ValueError("measured body failed")
    except RuntimeError as error:
        assert "measured body and GPU power sampler both failed" in str(error)
        assert "ValueError" in str(error) and "query failed" in str(error)
    else:
        raise AssertionError("dual failure did not propagate")
    assert not dual.thread_alive


def test_worker_and_final_paths_use_protocol_bound_power_sampler() -> None:
    compute = PROTOCOL["compute"]
    assert float(compute["power_draw_sampling_interval_seconds"]) > 0
    assert float(compute["power_draw_query_timeout_seconds"]) > 0
    assert int(compute["power_draw_minimum_samples"]) >= 2
    sampler = power_sampler_from_protocol(PROTOCOL, 2)
    assert sampler.physical_index == 2
    assert sampler.sampling_interval_seconds == float(
        compute["power_draw_sampling_interval_seconds"]
    )
    worker_source = inspect.getsource(run_group_worker_core)
    final_source = inspect.getsource(authority_finalize_core)
    for source in (worker_source, final_source):
        assert "with power_sampler_from_protocol" in source
        assert "gpu_power_draw_energy" in source
        assert "total_energy_consumption" not in source
    mapping_source = inspect.getsource(_validate_final_cuda_isolation)
    assert "final_evaluation_physical_cuda_index" in mapping_source
    assert "CUDA_VISIBLE_DEVICES" in mapping_source


def test_final_cuda_mapping_rejects_remap_and_requires_one_isolated_device() -> None:
    target = "funcweights.hfw_w560_post_training_bridge_20260815_v1r4r1_proven_solver.production_phases.torch.cuda"
    with mock.patch.dict(__import__("os").environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
        with mock.patch(target) as cuda:
            cuda.is_available.return_value = True
            cuda.device_count.return_value = 1
            with raises(RuntimeError):
                _validate_final_cuda_isolation(PROTOCOL)
    with mock.patch.dict(__import__("os").environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
        with mock.patch(target) as cuda:
            cuda.is_available.return_value = True
            cuda.device_count.return_value = 2
            with raises(RuntimeError):
                _validate_final_cuda_isolation(PROTOCOL)
            cuda.device_count.return_value = 1
            assert _validate_final_cuda_isolation(PROTOCOL) == 0


def test_canonical_commit_reveal_gives_search_only_opaque_capability(tmp_path: Path) -> None:
    canonical = load_canonical_sealed_evaluation(PROTOCOL, PACKAGE)
    authority = tmp_path / "authority"
    public = tmp_path / "public"
    capability = canonical.create_eval_commitment(
        authority_root=authority,
        public_root=public,
        validation_sha256="a" * 64,
        validation_token_count=100_000,
        evaluation_offsets=[10_000, 20_000],
        sample_length_tokens=513,
        salt=bytes(range(32)),
    )
    payload = capability.public_payload()
    encoded = json.dumps(payload)
    for forbidden in PROTOCOL["canonical_sealed_evaluation_binding"]["search_capability_must_exclude"]:
        assert forbidden not in encoded
    private = list((authority / "sealed_reveals").glob("*.reveal.json"))
    assert len(private) == 1
    assert private[0].stat().st_mode & 0o777 == 0o400
