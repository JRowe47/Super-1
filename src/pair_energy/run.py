"""Full-batch concept geometry and equal-rehearsal interference experiment.

Synthetic input generation is independent of every student architecture.
All hyperparameters and budgets are frozen in PROTOCOL.json before training.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("dense", "distance", "constrained_distance")


def digest_tensor(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def datasets(seed, c):
    """Eight concepts, context-dependent surface bit, unseen nuisance combinations."""
    rng = torch.Generator().manual_seed(seed + 10000)
    styles = torch.tensor(list(itertools.product((-1., 1.), repeat=4)))
    parity = (styles > 0).sum(1) % 2
    train_styles = torch.where(parity == 0)[0]
    odd = torch.where(parity == 1)[0]
    style_sets = (train_styles, odd[:len(odd)//2], odd[len(odd)//2:])
    # Columns are orthonormal before tanh; no use of student coordinates.
    mixing = torch.linalg.qr(torch.randn(c["input_dim"], 8, generator=rng)).Q
    splits, evidence = [], []
    for count, allowed in zip((c["train_per_class"], c["validation_per_class"], c["test_per_class"]), style_sets):
        xs, ys = [], []
        for label in range(c["classes"]):
            bits = torch.tensor([(label >> shift) & 1 for shift in (2, 1, 0)])
            context = torch.arange(count) % 2
            semantic = bits.expand(count, -1).clone()
            semantic[:, 2] ^= context
            style_ids = allowed[torch.arange(count) // 2 % len(allowed)]
            latent = torch.cat((semantic.float()*2-1, (context.float()*2-1)[:, None], styles[style_ids]), dim=1)
            x = torch.tanh(latent @ mixing.T) + c["noise_std"]*torch.randn(count, c["input_dim"], generator=rng)
            xs.append(x)
            ys.append(torch.full((count,), label, dtype=torch.long))
        data = (torch.cat(xs), torch.cat(ys))
        splits.append(data)
        evidence.append({"examples": len(data[1]), "style_ids": allowed.tolist(),
                         "inputs_sha256": digest_tensor(data[0]), "labels_sha256": digest_tensor(data[1])})
    assert all(set(evidence[i]["style_ids"]).isdisjoint(evidence[j]["style_ids"]) for i in range(3) for j in range(i))
    return splits, evidence


class Dense(nn.Module):
    def __init__(self, c, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(c["input_dim"], width), nn.Tanh(), nn.Linear(width, c["classes"]))

    def forward(self, x):
        return self.network(x), None


class Distance(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(c["input_dim"], c["hidden_width"]), nn.Tanh(),
                                     nn.Linear(c["hidden_width"], c["embedding_dim"]))
        self.prototypes = nn.Parameter(torch.randn(c["classes"], c["embedding_dim"]))
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(c["initial_temperature"]))))

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        distances = (z[:, None, :]-prototypes[None, :, :]).square().sum(-1)
        return -F.softplus(self.raw_temperature)*distances, z


def count(model):
    return sum(p.numel() for p in model.parameters())


def make_model(arm, seed, c):
    torch.manual_seed(seed)
    geometry = Distance(c)
    if arm != "dense":
        return geometry
    target = count(geometry)
    width = min(range(1, target+1), key=lambda w: abs((c["input_dim"]+1)*w+(w+1)*c["classes"]-target))
    torch.manual_seed(seed)
    return Dense(c, width)


def subset(data, mask):
    return data[0][mask], data[1][mask]


class Constraints:
    def __init__(self, labels, c):
        self.c = c
        self.i, self.j = torch.triu_indices(len(labels), len(labels), offset=1)
        same = labels[self.i] == labels[self.j]
        self.same = same
        self.dual = torch.zeros(len(self.i))
        assert len(self.i) == len(labels)*(len(labels)-1)//2

    def violations(self, z):
        d2 = (z[self.i]-z[self.j]).square().sum(-1)
        return torch.where(self.same, d2-self.c["close_radius"]**2,
                           self.c["separate_radius"]**2-d2)

    def penalty(self, violations):
        terms = self.dual * violations.relu() + .5*self.c["quadratic_penalty"]*violations.relu().square()
        # Give compatible and conflicting pairs equal total importance.
        groups = [terms[mask].mean() for mask in (self.same, ~self.same) if mask.any()]
        return torch.stack(groups).mean()

    def update(self, violations):
        with torch.no_grad():
            self.dual.add_(self.c["dual_learning_rate"]*violations.detach()).clamp_(min=0)

    def measure(self, z):
        with torch.no_grad():
            v = self.violations(z)
            result = {"pairs": len(v), "positive_pairs": int(self.same.sum()),
                      "negative_pairs": int((~self.same).sum()),
                      "max_squared_distance_violation": float(v.relu().max()),
                      "mean_violation": float(v.relu().mean()),
                      "satisfied_fraction": float((v <= self.c["constraint_tolerance"]).float().mean()),
                      "max_embedding_norm": float(z.norm(dim=-1).max())}
            for name, mask in (("compatible", self.same), ("conflicting", ~self.same)):
                result[name+"_satisfied_fraction"] = float((v[mask] <= self.c["constraint_tolerance"]).float().mean()) if mask.any() else None
            return result


def metrics(model, data, old_classes=None):
    with torch.no_grad():
        logits, _ = model(data[0])
        labels = data[1]
        pred = logits.argmax(-1)
        result = {"ce": float(F.cross_entropy(logits, labels)), "accuracy": float((pred==labels).float().mean()),
                  "conflicting_partner_error": float((pred==(labels^1)).float().mean())}
        if old_classes is not None:
            for name, mask in (("old", labels < old_classes), ("new", labels >= old_classes)):
                result[name+"_accuracy"] = float((pred[mask]==labels[mask]).float().mean()) if mask.any() else None
        return result


def fit(arm, seed, lr, c, train, validation, steps, initial=None):
    model = make_model(arm, seed, c)
    if initial is not None:
        model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    relations = Constraints(train[1], c) if arm != "dense" else None
    first = metrics(model, validation)
    best_ce, best_step, best = first["ce"], 0, copy.deepcopy(model.state_dict())
    history = [{"step": 0, "validation_ce": first["ce"], "validation_accuracy": first["accuracy"]}]
    start = time.perf_counter()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        logits, z = model(train[0])
        loss = F.cross_entropy(logits, train[1])
        if arm == "constrained_distance":
            violation = relations.violations(z)
            loss = loss + relations.penalty(violation)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite objective")
        loss.backward()
        optimizer.step()
        if arm == "constrained_distance":
            relations.update(violation)
        if step % c["validation_interval"] == 0 or step == steps:
            measured = metrics(model, validation)
            row = {"step": step, "validation_ce": measured["ce"], "validation_accuracy": measured["accuracy"]}
            if relations is not None:
                with torch.no_grad():
                    row["train_geometry"] = relations.measure(model(train[0])[1])
            history.append(row)
            if measured["ce"] < best_ce:
                best_ce, best_step, best = measured["ce"], step, copy.deepcopy(model.state_dict())
    final = copy.deepcopy(model.state_dict())
    elapsed = time.perf_counter()-start
    model.load_state_dict(best)
    record = {"arm": arm, "seed": seed, "learning_rate": lr, "parameters": count(model),
              "best_step": best_step, "validation": metrics(model, validation),
              "train": metrics(model, train), "training_seconds": elapsed, "history": history,
              "training_examples_per_update": len(train[1]), "steps": steps}
    if relations is not None:
        with torch.no_grad():
            record["train_geometry"] = relations.measure(model(train[0])[1])
            record["validation_geometry"] = Constraints(validation[1], c).measure(model(validation[0])[1])
            record["final_dual_max"] = float(relations.dual.max())
    return model, best, final, record


def save_checkpoint(out, name, state):
    path = out/(name+".pt")
    torch.save(state, path)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tune(arm, seed, c, train, validation, steps, out, name):
    candidates = []
    for lr in c["learning_rates"]:
        model, best, final, record = fit(arm, seed, lr, c, train, validation, steps)
        candidates.append((model, best, final, record))
        print(json.dumps({"stage": name, "seed": seed, "arm": arm, "lr": lr,
                          "validation_accuracy": record["validation"]["accuracy"],
                          "validation_ce": record["validation"]["ce"]}), flush=True)
    winner = min(candidates, key=lambda item: item[3]["validation"]["ce"])
    winner[3]["candidate_validation_results"] = [
        {"learning_rate": item[3]["learning_rate"], "validation": item[3]["validation"],
         "best_step": item[3]["best_step"], "training_seconds": item[3]["training_seconds"]} for item in candidates]
    winner[3]["checkpoint"] = save_checkpoint(out, f"{name}-{seed}-{arm}", winner[1])
    return winner


def make_report(results, out, c):
    def avg(rows, fn):
        return sum(fn(r) for r in rows)/len(rows)
    lines = ["# Full-batch disparity constraints experiment", "",
             "Eight synthetic concepts have context-dependent surface features. Noise and nuisance combinations vary. Training, validation, and test use disjoint nuisance combinations. This is a synthetic classification/interference test, not a language-model benchmark.", "",
             "Every update uses every example in the current training set. The constrained model also uses every unordered training pair, with per-pair nonnegative multipliers, a quadratic violation penalty, and unit-norm embeddings. Different labels repel and same labels attract. Class labels define compatibility in this task; it is not a claim that all distinct real-world concepts should repel.", "",
             "All arms receive the same two learning-rate candidates and optimizer-step budgets. Validation CE selects the learning rate and checkpoint; test is evaluated only afterward. Parameters are approximately matched; FLOPs and wall time are not. Constraint constants are fixed in advance, not tuned on the test set.", "",
             "## Joint learning", "", "| Model | Parameters | Test accuracy | Test CE | Conflicting-partner error | Training constraints satisfied | Worst squared-distance violation |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows = [r for r in results["joint"] if r["arm"]==arm]
        geometry = f"{100*avg(rows, lambda r:r['train_geometry']['satisfied_fraction']):.2f}%" if arm != "dense" else "N/A"
        worst = f"{max(r['train_geometry']['max_squared_distance_violation'] for r in rows):.6f}" if arm != "dense" else "N/A"
        lines.append(f"| {arm} | {rows[0]['parameters']} | {100*avg(rows, lambda r:r['test']['accuracy']):.2f}% | {avg(rows, lambda r:r['test']['ce']):.4f} | {100*avg(rows, lambda r:r['test']['conflicting_partner_error']):.2f}% | {geometry} | {worst} |")
    lines += ["", "## New-concept interference", "",
              "Pretrain on six classes, then add two classes. All arms use identical rehearsal examples; validation includes all classes during adaptation. The fixed-final-step columns prevent checkpoint selection from hiding eventual forgetting. Forgetting is old-class accuracy before adaptation minus afterward; negative values indicate improvement.", "",
              "| Rehearsal per old class | Model | Old accuracy before | Old accuracy selected | New accuracy selected | Forgetting selected (pp) | Old accuracy final step | New accuracy final step |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rehearsal in c["rehearsal_per_old_class"]:
        for arm in ARMS:
            rows = [r for r in results["adaptation"] if r["arm"]==arm and r["rehearsal_per_old_class"]==rehearsal]
            values = [avg(rows, f) for f in (
                lambda r:r["before_test"]["old_accuracy"], lambda r:r["test"]["old_accuracy"],
                lambda r:r["test"]["new_accuracy"], lambda r:r["forgetting"],
                lambda r:r["final_test"]["old_accuracy"], lambda r:r["final_test"]["new_accuracy"])]
            lines.append(f"| {rehearsal} | {arm} | "+" | ".join(f"{100*v:.2f}" for v in values)+" |")
    lines += ["", "All values are means over three seeds except the worst violation. Full per-seed results, constraint coverage, held-out geometry, learning curves, checkpoint hashes, and data hashes are in RESULT.json and PROTOCOL.json. Data-generation mappings also vary by seed. No statistical significance or additional independent information capacity is claimed."]
    (out/"REPORT.md").write_text('\n'.join(lines)+'\n')
    os.environ.setdefault("MPLCONFIGDIR", str(out/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    names = ["Dense", "Distance", "Constrained\ndistance"]
    rows_by_arm = [[r for r in results["joint"] if r["arm"]==arm] for arm in ARMS]
    axes[0].bar(names, [100*avg(rows, lambda r:r["test"]["accuracy"]) for rows in rows_by_arm])
    axes[0].set(title="Joint learning: unseen combinations", ylabel="Test accuracy (%)", ylim=(0,100))
    for axis, rehearsal in zip(axes[1:], c["rehearsal_per_old_class"]):
        for arm, label in zip(ARMS, names):
            rows = [r for r in results["adaptation"] if r["arm"]==arm and r["rehearsal_per_old_class"]==rehearsal]
            axis.scatter(100*avg(rows, lambda r:r["test"]["new_accuracy"]),
                         100*avg(rows, lambda r:r["test"]["old_accuracy"]), label=label, s=80)
        axis.set(title=f"Rehearsal: {rehearsal} examples / old class", xlabel="New-class test accuracy (%)",
                 ylabel="Retained old-class accuracy (%)", xlim=(-3,103), ylim=(-3,103))
        axis.legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/"comparison.png", dpi=170)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    assert c["classes"] == 8 and c["input_dim"] >= 8 and c["embedding_dim"] >= 3
    assert c["train_per_class"] % 16 == 0 and c["validation_per_class"] % 8 == 0 and c["test_per_class"] % 8 == 0
    assert 0 < c["close_radius"] < c["separate_radius"] <= math.sqrt(4/3)
    torch.set_num_threads(c["threads"])
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    prepared = {seed: datasets(seed, c) for seed in c["seeds"]}
    protocol = {"config": c, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "torch": torch.__version__, "device": "cpu", "arms": ARMS,
                "data": {seed:value[1] for seed,value in prepared.items()},
                "geometry_feasibility": "Eight unit cube vertices embedded in 6D have minimum squared separation 4/3; within-class collapse gives zero positive distance. Abstract margins are feasible, but encoder optimization is not guaranteed.",
                "full_batch": True, "all_unordered_pairs": True,
                "rehearsal_selection": "first examples of each old class from training only, same IDs across arms",
                "selection": "minimum validation CE; adaptation reuses selected pretraining learning rate",
                "test_access": "only after each model and checkpoint choice is frozen"}
    (out/"PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
    (out/"run_source.py").write_text(Path(__file__).read_text())
    results = {"joint": [], "pretrain": [], "adaptation": []}
    for seed in c["seeds"]:
        (train, valid, test), _ = prepared[seed]
        for arm in ARMS:
            model, best, final, record = tune(arm, seed, c, train, valid, c["joint_steps"], out, "joint")
            record["test"] = metrics(model, test)
            if arm != "dense":
                with torch.no_grad():
                    record["test_geometry"] = Constraints(test[1], c).measure(model(test[0])[1])
            results["joint"].append(record)
            old_train = subset(train, train[1] < c["old_classes"])
            old_valid = subset(valid, valid[1] < c["old_classes"])
            old_model, old_best, _, old_record = tune(arm, seed, c, old_train, old_valid, c["pretrain_steps"], out, "pretrain")
            before = metrics(old_model, test, c["old_classes"])
            old_record["test"] = before
            results["pretrain"].append(old_record)
            for rehearsal in c["rehearsal_per_old_class"]:
                indices = [torch.where(train[1]>=c["old_classes"])[0]]
                for label in range(c["old_classes"]):
                    indices.append(torch.where(train[1]==label)[0][:rehearsal])
                chosen = torch.cat(indices)
                adaptation_train = subset(train, chosen)
                adapted, adapted_best, adapted_final, adapted_record = fit(
                    arm, seed, old_record["learning_rate"], c, adaptation_train, valid, c["adapt_steps"], old_best)
                adapted_record["test"] = metrics(adapted, test, c["old_classes"])
                adapted_record["before_test"] = before
                adapted_record["forgetting"] = before["old_accuracy"]-adapted_record["test"]["old_accuracy"]
                adapted_record["rehearsal_per_old_class"] = rehearsal
                adapted_record["training_indices"] = chosen.tolist()
                adapted_record["checkpoint"] = save_checkpoint(out, f"adapt-{seed}-{arm}-rehearsal{rehearsal}", adapted_best)
                adapted.load_state_dict(adapted_final)
                adapted_record["final_test"] = metrics(adapted, test, c["old_classes"])
                adapted_record["final_checkpoint"] = save_checkpoint(out, f"adapt-final-{seed}-{arm}-rehearsal{rehearsal}", adapted_final)
                results["adaptation"].append(adapted_record)
                print(json.dumps({"stage": "adapt", "seed": seed, "arm": arm, "rehearsal": rehearsal,
                                  "old_accuracy": adapted_record["test"]["old_accuracy"],
                                  "new_accuracy": adapted_record["test"]["new_accuracy"]}), flush=True)
            (out/"RESULT.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    make_report(results, out, c)
    print(str(out/"REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
