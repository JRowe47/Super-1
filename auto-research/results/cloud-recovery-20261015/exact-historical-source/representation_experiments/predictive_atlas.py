"""Behavioral fingerprints and online atlas fitting.

Atoms are defined only through teacher output distributions and their response
to input interventions.  No teacher neuron, layer coordinate, or hidden-state
axis is stored in the atlas.
"""
from __future__ import annotations

import math

import torch


def hash_projection(vocabulary: int, dimensions: int, seed: int,
                    device: torch.device | None = None):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    buckets = torch.randint(dimensions, (vocabulary,), generator=generator)
    signs = torch.randint(0, 2, (vocabulary,), generator=generator).float() * 2 - 1
    if device is not None:
        buckets = buckets.to(device)
        signs = signs.to(device)
    return buckets, signs


def predictive_sketch(logits: torch.Tensor, buckets: torch.Tensor,
                      signs: torch.Tensor, topk: int, dimensions: int) -> torch.Tensor:
    """CountSketch of the teacher distribution plus confidence statistics."""
    log_probability = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(log_probability, min(topk, logits.shape[-1]), dim=-1)
    probability = values.exp()
    target_shape = (*indices.shape[:-1], dimensions)
    sketch = torch.zeros(target_shape, device=logits.device, dtype=torch.float32)
    bucket = buckets.index_select(0, indices.reshape(-1)).view_as(indices)
    sign = signs.index_select(0, indices.reshape(-1)).view_as(indices)
    sketch.scatter_add_(-1, bucket, probability * sign)
    captured_mass = probability.sum(-1, keepdim=True)
    entropy_topk = -(probability * values).sum(-1, keepdim=True)
    margin = (values[..., :1] - values[..., 1:2]
              if values.shape[-1] > 1 else torch.zeros_like(values[..., :1]))
    return torch.cat((sketch, captured_mass, entropy_topk, margin), dim=-1)


def behavioral_fingerprint(original_logits: torch.Tensor,
                           intervened_logits: torch.Tensor,
                           buckets: torch.Tensor, signs: torch.Tensor,
                           topk: int, dimensions: int) -> torch.Tensor:
    original = predictive_sketch(
        original_logits, buckets, signs, topk, dimensions)
    intervened = predictive_sketch(
        intervened_logits, buckets, signs, topk, dimensions)
    return torch.cat((original, original - intervened), dim=-1)


def corrupt_context(input_ids: torch.Tensor, replacement_id: int,
                    span_fraction: float, seed: int) -> torch.Tensor:
    result = input_ids.clone()
    batch, sequence = result.shape
    span = max(1, int(round(sequence * span_fraction)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    starts = torch.randint(max(1, sequence - span + 1), (batch,), generator=generator)
    for index, start in enumerate(starts.tolist()):
        result[index, start:start + span] = int(replacement_id)
    return result


class OnlineKMeans:
    def __init__(self, centers: torch.Tensor, counts: torch.Tensor | None = None):
        self.centers = centers.float()
        self.counts = (counts.float() if counts is not None
                       else torch.zeros(centers.shape[0], device=centers.device))

    @torch.no_grad()
    def update(self, samples: torch.Tensor, rate_floor: float = 0.01):
        samples = samples.float()
        assignment = nearest_atoms(samples, self.centers)
        for atom in assignment.unique().tolist():
            selected = samples[assignment == atom]
            new_count = float(selected.shape[0])
            old_count = float(self.counts[atom])
            rate = max(rate_floor, new_count / max(old_count + new_count, 1.0))
            self.centers[atom].lerp_(selected.mean(0), rate)
            self.counts[atom] += new_count
        return assignment


@torch.no_grad()
def nearest_atoms(samples: torch.Tensor, centers: torch.Tensor,
                  center_chunk: int = 1024) -> torch.Tensor:
    samples = samples.float()
    best_distance = torch.full(
        (samples.shape[0],), float("inf"), device=samples.device)
    best_index = torch.zeros(samples.shape[0], dtype=torch.long, device=samples.device)
    for start in range(0, centers.shape[0], center_chunk):
        chunk = centers[start:start + center_chunk].to(samples.device).float()
        distance = torch.cdist(samples, chunk).square()
        value, index = distance.min(dim=-1)
        replace = value < best_distance
        best_distance[replace] = value[replace]
        best_index[replace] = index[replace] + start
    return best_index


def atlas_diagnostics(centers: torch.Tensor, counts: torch.Tensor) -> dict:
    probability = counts.float() / counts.sum().clamp_min(1)
    active = counts > 0
    entropy = -(probability[active] * probability[active].log()).sum()
    perplexity = entropy.exp()
    normalized = torch.nn.functional.normalize(centers.float(), dim=-1)
    gram = normalized @ normalized.t()
    gram.fill_diagonal_(0)
    return {
        "atoms": int(centers.shape[0]),
        "dimensions": int(centers.shape[1]),
        "active_atoms": int(active.sum()),
        "assignment_perplexity": float(perplexity),
        "largest_atom_fraction": float(probability.max()),
        "mean_nearest_cosine": float(gram.max(dim=-1).values.mean()),
    }

