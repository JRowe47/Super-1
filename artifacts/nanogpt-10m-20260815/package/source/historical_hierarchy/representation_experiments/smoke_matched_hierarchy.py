"""Dependency-light architecture and budget validation for the matched study."""
from __future__ import annotations

import argparse
import json

import torch

from representation_experiments.common import parameter_count
from representation_experiments.predictive_models import build_student


def resolve_budget(kind, model_config, study, target_parameters):
    widths = list(range(
        int(model_config["minimum_width"]),
        int(model_config["maximum_width"]) + 1,
        int(model_config["width_step"])))
    if kind == "transformer":
        widths = [width for width in widths
                  if width % int(model_config["heads"]) == 0]
    selected = None
    for width in widths:
        with torch.device("meta"):
            model = build_student(
                kind, int(study["vocab_size"]), width,
                int(study["model_max_context"]), model_config)
        parameters = parameter_count(model)
        if parameters <= int(target_parameters):
            selected = (width, parameters, model)
        else:
            break
    if selected is None:
        raise RuntimeError(f"{kind} has no model under the storage budget")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        study = json.load(stream)
    rows = {}
    utilization = []
    for arm, arm_config in study["arms"].items():
        kind = arm_config["architecture"]
        model_config = study["models"][kind]
        target_parameters = int(arm_config.get(
            "parameter_budget", study["student_parameter_budget"]))
        width, parameters, model = resolve_budget(
            kind, model_config, study, target_parameters)
        utilization.append(parameters / target_parameters)
        rows[arm] = {
            "architecture": kind,
            "width": width,
            "parameters": parameters,
            "budget_utilization": utilization[-1],
            "state_bytes_bf16_at_512": 2 * model.stored_state_elements(1, 512),
            "state_bytes_bf16_at_2048": 2 * model.stored_state_elements(1, 2048),
            "estimated_macs_at_512": model.estimated_macs_per_token(512),
        }
    maximum_spread = float(
        study["promotion_gates"]["maximum_budget_utilization_spread"])
    if max(utilization) - min(utilization) > maximum_spread:
        raise RuntimeError("parameter-budget utilization is not matched")

    tiny_vocab = 257
    tiny_width = 64
    for arm, arm_config in study["arms"].items():
        kind = arm_config["architecture"]
        model_config = study["models"][kind]
        horizons = [int(value) for value in model_config.get("timescales", [])]
        tiny_context = max([80, *(value + 2 for value in horizons)])
        model = build_student(
            kind, tiny_vocab, tiny_width, max(96, tiny_context), model_config)
        inputs = torch.randint(0, tiny_vocab, (2, tiny_context))
        output = model(inputs)
        if output.logits.shape != (2, tiny_context, tiny_vocab):
            raise RuntimeError(f"unexpected output shape for {arm}")
        output.logits.float().square().mean().backward()
    print(json.dumps({"status": "PASS", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
