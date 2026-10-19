#!/usr/bin/env python3
"""Check saved-score arithmetic only, using the Python standard library.

No experiment module imports, network, checkpoint loads, model forward passes,
optimization, bootstrap reruns, or writes. These checks cannot validate whether
historical model outputs are correct. Run from any directory with Python 3.10+.
"""
import gzip
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "snapshots"
COUNTS = Counter()
FILES = set()


def read(stage, name):
    path = ROOT / stage / "results" / name
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    FILES.add(str(path.relative_to(ROOT)))
    return json.loads(data)


def mean(values):
    values = list(values)
    assert values, "Empty averaging group"
    return math.fsum(values) / len(values)


def close(a, b, label, *, atol=1e-9, rtol=1e-10):
    assert math.isfinite(a) and math.isfinite(b), label + ": non-finite"
    assert math.isclose(a, b, abs_tol=atol, rel_tol=rtol), (
        f"{label}: calculated {a:.17g}, saved {b:.17g}"
    )
    COUNTS[label.split(":")[0]] += 1


def walk_scores(obj):
    """Reaggregate raw positions, sequences, blocks, and prose passages."""
    if isinstance(obj, list):
        for value in obj:
            walk_scores(value)
    elif isinstance(obj, dict):
        if "position_nll" in obj and "target_nll" in obj:
            close(mean(obj["position_nll"]), obj["target_nll"], "position_nll", atol=2e-6, rtol=5e-7)
        if "position_correct" in obj and "accuracy" in obj:
            close(mean(obj["position_correct"]), obj["accuracy"], "position_accuracy", atol=6e-8)
        if "block_nll" in obj and "nll" in obj:
            close(mean(obj["block_nll"]), obj["nll"], "validation_mean")
        if "sequences" in obj:
            for key in ("target_nll", "accuracy", "first_nll", "advantage", "content_attention", "duplicate_attention"):
                if key in obj and all(key in s for s in obj["sequences"]):
                    close(mean(s[key] for s in obj["sequences"]), obj[key], "sequence_mean:" + key)
        if "passages" in obj and "tokens" in obj:
            rows = obj["passages"]
            assert sum(r["tokens"] for r in rows) == obj["tokens"], "prose token total"
            COUNTS["prose_token_total"] += 1
            for key in ("nll", "kl_from_baseline", "argmax_changed"):
                if key in obj and all(key in r for r in rows):
                    value = math.fsum(r[key] * r["tokens"] for r in rows) / obj["tokens"]
                    close(value, obj[key], "prose_weighted_mean:" + key)
        for value in obj.values():
            walk_scores(value)


def measure(row, short_only=False):
    ret = row["retention"]
    result = {"validation_nll": mean(row["validation"]["block_nll"]), "prose_nll": ret["prose"]["nll"]}
    for kind in ("repeat", "scrambled"):
        rows = ret[kind]["sequences"]
        selected = [s for s in rows if s["n"] <= 48] if short_only else rows
        result[kind] = {key: mean(s[key] for s in selected) for key in ("accuracy", "target_nll")}
        if all("last_query_teacher_kl" in s for s in selected):
            result[kind]["last_query_teacher_kl"] = mean(s["last_query_teacher_kl"] for s in selected)
    return result


def check_estimates(saved, a, b, path=""):
    """Check point contrasts; confidence intervals remain historical."""
    for key, value in saved.items():
        here = path + "/" + key
        if isinstance(value, dict) and "estimate" in value:
            scale = 100 if key == "accuracy" else 1
            close(scale * (a[key] - b[key]), value["estimate"], "contrast_estimate:" + here)
            assert len(value["ci95"]) == 2 and value["ci95"][0] <= value["ci95"][1], here
        elif isinstance(value, dict):
            check_estimates(value, a[key], b[key], here)


def average_measure(a, b):
    return {k: average_measure(v, b[k]) if isinstance(v, dict) else (v + b[k]) / 2 for k, v in a.items()}


def check_curve_stage(stage):
    rows = read(stage, "evaluations.json")
    walk_scores(rows)
    measures = {(r["arm"], r["step"]): measure(r, True) for r in rows}
    if stage == "curve_mask_transport":
        for step in (1, 12, 24):
            measures[("random_mean", step)] = average_measure(measures[("random0", step)], measures[("random1", step)])
    effects = read(stage, "effects.json")
    base = measures[("baseline", 0)]
    for row in effects:
        m = measures[(row["arm"], row["step"])]
        close(m["validation_nll"], row["validation_nll"], "effect_validation_mean")
        for kind in ("repeat", "scrambled"):
            for key, value in m[kind].items():
                close(value, row[kind][key], "effect_short_sequence_mean:" + key)
        check_estimates(row["changes"], m, base)
    for row in read(stage, "contrasts.json"):
        step = row.get("step", 24)
        check_estimates(row["contrast"], measures[(row["a"], step)], measures[(row["b"], step)])
    return rows


def main():
    # The raw score files preserve outcomes rather than requiring their original
    # checkpoint/response-array dependencies.
    raw = {
        "pythia70m_component_causality": ["discovery.json", "confirmation.json", "later_context.json"],
        "pythia70m_matched_components": ["scan.json", "confirmation.json"],
        "curve_training_compatibility": ["pythia_evaluations.json", "stable_sgd_evaluations.json"],
    }
    for stage, names in raw.items():
        for name in names:
            walk_scores(read(stage, name))
    check_curve_stage("curve_retention_depth")
    check_curve_stage("curve_mask_transport")

    stage = "curve_mask_transport"
    train = read(stage, "training.json")
    reference = read(stage, "reference_step_norms.json")
    max_error = max(abs(r["step_norm"] - reference[r["step"] - 1]) for r in train)
    saved = read(stage, "analysis_checks.json")
    close(max_error, saved["matching_step_norm_max_error"], "mask_step_norm_max_error", atol=1e-18)
    masks = read(stage, "mask_band_counts.json")
    assert all(len(set(row["enabled"].values())) == 1 for row in masks)
    COUNTS["equal_per_band_mask_counts"] = len(masks)
    frozen = read(stage, "mask_checks.json")
    assert len(frozen) == 12 and all(r["frozen_coefficient_max_change"] == 0 for r in frozen)
    COUNTS["recorded_frozen_master_checks"] = len(frozen)
    bands = read(stage, "haar_update_bands.json")
    for arm in ("full", "targeted", "random0", "random1", "opposite"):
        selected = [r for r in bands if r["arm"] == arm and r["step"] == 24]
        assert sum(r["count"] for r in selected) == 18914304
        COUNTS["full_coordinate_count"] += 1
    toys = read(stage, "transport_evaluations.json")
    assert len(toys) == 80
    close(max(r["max_drift_from_12_pulse"] for r in toys), saved["max_toy_grid_drift"], "toy_grid_drift", atol=1e-18)
    for item in read(stage, "transport_summary.json"):
        values = [r for r in toys if r["arm"] == item["arm"] and r["grid"] == item["grid"]]
        assert len(values) == 5
        for key in ("expected_mse", "empirical_mse", "sign_accuracy", "auxiliary_energy", "max_drift_from_12_pulse"):
            for statistic, fn in (("mean", mean), ("min", min), ("max", max)):
                close(fn(r[key] for r in values), item[key][statistic], "toy_saved_summary:" + key, atol=1e-40, rtol=1e-10)
    print(json.dumps({
        "status": "PASS",
        "scope": "Saved-score arithmetic and consistency only; no model execution or fresh experimental validation.",
        "files_read": len(FILES), "numeric_checks": sum(COUNTS.values()), "checks_by_category": dict(COUNTS),
        "not_checked": ["historical forward-pass correctness", "bootstrap confidence-interval recomputation", "excluded checkpoint/coefficient/hidden-state arrays", "all reported findings across all ten stages"],
    }, indent=2))


if __name__ == "__main__":
    main()
