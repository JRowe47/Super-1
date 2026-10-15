import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def load_text(path):
    lines = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) > 4:
            lines.extend((parts[2], parts[4]))
    return "\n".join(lines)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


class ScalarProjectionLM(nn.Module):
    def __init__(self, vocab, width, cfg):
        super().__init__()
        projections = cfg["scalar_projections"]
        self.embedding = nn.Embedding(vocab, width)
        self.content = nn.Parameter(torch.randn(projections, width) * 0.03)
        self.decay_logit = nn.Parameter(torch.linspace(-2.0, 3.0, projections))
        self.input_gain = nn.Parameter(torch.ones(projections))
        self.mixer = nn.Sequential(
            nn.Linear(projections, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.output = nn.Linear(width, vocab)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        state = torch.zeros(tokens.shape[0], self.content.shape[0], device=tokens.device)
        outputs = []
        decay = torch.sigmoid(self.decay_logit)
        for index in range(tokens.shape[1]):
            update = torch.tanh(embedded[:, index] @ self.content.T) * self.input_gain
            state = decay * state + (1.0 - decay) * update
            outputs.append(self.output(self.mixer(state)))
        return torch.stack(outputs, dim=1)


class MultiScaleCore(nn.Module):
    def __init__(self, vocab, width, cfg):
        super().__init__()
        self.scales = cfg["state_scales"]
        self.width = width
        self.embedding = nn.Embedding(vocab, width)
        self.scale = nn.Parameter(torch.randn(self.scales, width) * 0.02)
        self.input_update = nn.Linear(width, width)
        self.state_update = nn.Linear(width, width, bias=False)
        self.input_gate = nn.Linear(width, width)
        self.state_gate = nn.Linear(width, width, bias=False)
        self.coupling = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def scan(self, tokens):
        embedded = self.embedding(tokens)
        state = torch.zeros(tokens.shape[0], self.scales, self.width, device=tokens.device)
        outputs = []
        for index in range(tokens.shape[1]):
            current = embedded[:, index]
            global_state = state.mean(dim=1)
            update = torch.tanh(
                self.input_update(current)[:, None, :]
                + self.state_update(state)
                + self.coupling(global_state)[:, None, :]
                + self.scale[None, :, :]
            )
            gate = torch.sigmoid(
                self.input_gate(current)[:, None, :]
                + self.state_gate(state)
                - self.scale[None, :, :]
            )
            state = gate * state + (1.0 - gate) * update
            outputs.append(self.norm(state.mean(dim=1)))
        return embedded, outputs


class MultiScaleLM(nn.Module):
    def __init__(self, vocab, width, cfg):
        super().__init__()
        self.core = MultiScaleCore(vocab, width, cfg)
        self.output = nn.Linear(width, vocab)

    def forward(self, tokens):
        _, states = self.core.scan(tokens)
        return torch.stack([self.output(state) for state in states], dim=1)


class AssociativeMemoryLM(nn.Module):
    def __init__(self, vocab, width, cfg):
        super().__init__()
        self.core = MultiScaleCore(vocab, width, cfg)
        self.memory_width = max(8, width // 4)
        self.key = nn.Linear(width, self.memory_width)
        self.value = nn.Linear(width, self.memory_width)
        self.query = nn.Linear(width, self.memory_width)
        self.memory_decay = nn.Linear(width, 1)
        self.readout = nn.Sequential(
            nn.Linear(width + self.memory_width, width), nn.GELU(), nn.Linear(width, vocab)
        )

    def forward(self, tokens):
        embedded, states = self.core.scan(tokens)
        memory = torch.zeros(
            tokens.shape[0], self.memory_width, self.memory_width, device=tokens.device
        )
        outputs = []
        for index, state in enumerate(states):
            key = torch.tanh(self.key(embedded[:, index]))
            value = torch.tanh(self.value(state))
            query = torch.tanh(self.query(state))
            decay = torch.sigmoid(self.memory_decay(state))[:, :, None]
            memory = decay * memory + torch.einsum("bi,bj->bij", key, value)
            read = torch.einsum("bij,bi->bj", memory, query) / math.sqrt(self.memory_width)
            outputs.append(self.readout(torch.cat((state, read), dim=-1)))
        return torch.stack(outputs, dim=1)


class FunctionValuedLinear(nn.Module):
    def __init__(self, input_width, output_width, condition_width, cfg):
        super().__init__()
        bases = cfg["function_bases"]
        rank = cfg["function_rank"]
        self.base = nn.Linear(input_width, output_width)
        self.coefficients = nn.Linear(condition_width, bases)
        self.left = nn.Parameter(torch.randn(bases, output_width, rank) * 0.02)
        self.right = nn.Parameter(torch.randn(bases, rank, input_width) * 0.02)

    def forward(self, values, condition):
        coefficients = torch.tanh(self.coefficients(condition))
        compressed = torch.einsum("bi,kri->bkr", values, self.right)
        expanded = torch.einsum("bkr,kor->bko", compressed, self.left)
        return self.base(values) + (coefficients[:, :, None] * expanded).sum(dim=1)


class FunctionWeightLM(nn.Module):
    def __init__(self, vocab, width, cfg):
        super().__init__()
        self.scales = cfg["state_scales"]
        self.width = width
        self.embedding = nn.Embedding(vocab, width)
        self.scale = nn.Parameter(torch.randn(self.scales, width) * 0.02)
        condition_width = width * 2
        self.input_function = FunctionValuedLinear(width, width, condition_width, cfg)
        self.state_function = FunctionValuedLinear(width, width, condition_width, cfg)
        self.gate_function = FunctionValuedLinear(width, width, condition_width, cfg)
        self.output_function = FunctionValuedLinear(width, vocab, width, cfg)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        state = torch.zeros(tokens.shape[0], self.scales, self.width, device=tokens.device)
        outputs = []
        for index in range(tokens.shape[1]):
            current = embedded[:, index]
            repeated = current[:, None, :].expand(-1, self.scales, -1)
            condition = torch.cat((repeated, self.scale[None, :, :].expand_as(repeated)), dim=-1)
            flat_condition = condition.reshape(-1, condition.shape[-1])
            flat_input = repeated.reshape(-1, self.width)
            flat_state = state.reshape(-1, self.width)
            update = torch.tanh(
                self.input_function(flat_input, flat_condition)
                + self.state_function(flat_state, flat_condition)
            ).reshape_as(state)
            gate = torch.sigmoid(
                self.gate_function(flat_state, flat_condition)
            ).reshape_as(state)
            state = gate * state + (1.0 - gate) * update
            aggregate = self.norm(state.mean(dim=1))
            outputs.append(self.output_function(aggregate, aggregate))
        return torch.stack(outputs, dim=1)


class TransformerLM(nn.Module):
    def __init__(self, vocab, width, cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Embedding(cfg["context"], width)
        feedforward = width * 2
        block = nn.TransformerEncoderLayer(
            width,
            cfg["transformer_heads"],
            feedforward,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.network = nn.TransformerEncoder(block, cfg["transformer_layers"])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.embedding(tokens) + self.position(positions)
        return self.output(self.norm(self.network(hidden, mask=mask)))


def fit_model(factory, vocab, cfg, width_step):
    target = cfg["parameter_target"]
    candidates = []
    for width in range(width_step, 385, width_step):
        try:
            model = factory(vocab, width, cfg)
            candidates.append((abs(count_parameters(model) - target), width, model))
        except Exception:
            continue
    _, width, model = min(candidates, key=lambda item: item[0])
    return width, model


def fixed_windows(source, cfg, seed):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(
        0,
        len(source) - cfg["context"] - 1,
        (cfg["evaluation_windows"],),
        generator=generator,
    )
    windows = torch.stack([source[index : index + cfg["context"] + 1] for index in starts])
    return windows[:, :-1], windows[:, 1:]


def training_starts(source, cfg, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        len(source) - cfg["context"] - 1,
        (cfg["training_steps"], cfg["batch_size"]),
        generator=generator,
    )


def batch_from_starts(source, starts, context, device):
    windows = torch.stack([source[index : index + context + 1] for index in starts])
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def evaluate(model, examples, cfg, device):
    model.eval()
    x, y = examples
    total_loss = 0.0
    correct = 0
    tokens = 0
    with torch.no_grad():
        for start in range(0, len(x), cfg["batch_size"]):
            inputs = x[start : start + cfg["batch_size"]].to(device)
            targets = y[start : start + cfg["batch_size"]].to(device)
            logits = model(inputs)
            total_loss += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            ).item()
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            tokens += targets.numel()
    model.train()
    return total_loss / tokens, correct / tokens


def train_one(model, train_ids, starts, control, test, cfg, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    best_control = float("inf")
    best_state = None
    curve = []
    started = time.time()
    for step in range(1, cfg["training_steps"] + 1):
        x, y = batch_from_starts(train_ids, starts[step - 1], cfg["context"], device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % cfg["evaluation_interval"] == 0:
            control_ce, control_accuracy = evaluate(model, control, cfg, device)
            curve.append({"step": step, "control_ce": control_ce, "control_accuracy": control_accuracy})
            if control_ce < best_control:
                best_control = control_ce
                best_state = copy.deepcopy(model).cpu().state_dict()
                model = model.to(device)
    model.load_state_dict(best_state)
    model = model.to(device)
    test_ce, test_accuracy = evaluate(model, test, cfg, device)
    return {
        "parameters": count_parameters(model),
        "best_control_ce": best_control,
        "test_ce": test_ce,
        "test_bpc": test_ce / math.log(2),
        "test_accuracy": test_accuracy,
        "elapsed_seconds": time.time() - started,
        "curve": curve,
        "checkpoint": {name: value.cpu() for name, value in model.state_dict().items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    corpus = Path(args.corpus)
    text = load_text(corpus)
    chars = sorted(set(text))
    encoding = {char: index for index, char in enumerate(chars)}
    ids = torch.tensor([encoding[char] for char in text])
    train_end = int(len(ids) * 0.8)
    control_end = int(len(ids) * 0.9)
    train_ids = ids[:train_end]
    control_ids = ids[train_end:control_end]
    test_ids = ids[control_end:]
    factories = {
        "scalar_projection": (ScalarProjectionLM, 8),
        "multiscale_state": (MultiScaleLM, 8),
        "associative_memory": (AssociativeMemoryLM, 8),
        "function_valued_weights": (FunctionWeightLM, 8),
        "transformer": (TransformerLM, cfg["transformer_heads"]),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in cfg["seeds"]:
        starts = training_starts(train_ids, cfg, seed + 30000)
        control = fixed_windows(control_ids, cfg, seed + 10000)
        test = fixed_windows(test_ids, cfg, seed + 20000)
        models = {}
        for name, (factory, step) in factories.items():
            torch.manual_seed(seed)
            width, model = fit_model(factory, len(chars), cfg, step)
            result = train_one(model, train_ids, starts, control, test, cfg, "cuda")
            checkpoint = result.pop("checkpoint")
            torch.save(checkpoint, out / f"checkpoint-{name}-seed-{seed}.pt")
            result["width"] = width
            models[name] = result
            print(json.dumps({"seed": seed, "model": name, "width": width, "parameters": result["parameters"], "test_ce": result["test_ce"]}), flush=True)
        runs.append({"seed": seed, "models": models})
    aggregate = {}
    for name in factories:
        records = [run["models"][name] for run in runs]
        aggregate[name] = {
            "mean_parameters": sum(record["parameters"] for record in records) / len(records),
            "mean_test_ce": sum(record["test_ce"] for record in records) / len(records),
            "mean_test_accuracy": sum(record["test_accuracy"] for record in records) / len(records),
            "mean_seconds": sum(record["elapsed_seconds"] for record in records) / len(records),
        }
    result = {
        "status": "PASS_FUNCTION_VALUED_WEIGHT_LANGUAGE_COMPARISON",
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "corpus_characters": len(text),
        "vocab": len(chars),
        "config": cfg,
        "training_contract": {
            "causal_loss_at_every_position": True,
            "same_training_windows_per_seed": True,
            "same_control_and_test_windows_per_seed": True,
            "test_used_after_control_checkpoint_selection": True
        },
        "runs": runs,
        "aggregate": aggregate,
    }
    (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
