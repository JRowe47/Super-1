"""Parameter-free experiment adapter for the pinned upstream nanoGPT GPT.

The model implementation itself is vendored byte-for-byte at
``third_party/nanogpt/model.py`` from commit
``3adf61e154c3fe3fca428ad6bc3818b27a3b8291``.  This adapter deliberately
owns no tensors outside ``self.gpt``.  Its only job is to expose the common
student result shape and explicit theoretical KV accounting used by the
comparison harness.  The forward below is the full-logit branch of upstream
``GPT.forward`` with its post-final-LayerNorm hidden tensor retained for the
already-frozen representation objectives.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from third_party.nanogpt.model import GPT, GPTConfig


NANOGPT_UPSTREAM_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANOGPT_UPSTREAM_MODEL_SHA256 = (
    "7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c")
NANOGPT_UPSTREAM_LICENSE_SHA256 = (
    "a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe")


class NanoGPTStudent(nn.Module):
    """Faithful nanoGPT GPT plus a tensor-free harness interface."""

    kind = "nanogpt"
    state_schema = "theoretical_autoregressive_kv_v1"
    upstream_commit = NANOGPT_UPSTREAM_COMMIT
    upstream_model_sha256 = NANOGPT_UPSTREAM_MODEL_SHA256
    upstream_license_sha256 = NANOGPT_UPSTREAM_LICENSE_SHA256

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 layers: int, heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        if width % heads:
            raise ValueError("nanoGPT width must divide evenly into heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("nanoGPT dropout must be in [0, 1)")
        self.vocab_size = int(vocab_size)
        self.width = int(width)
        self.max_context = int(max_context)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.bias = bool(bias)
        self.gpt = GPT(GPTConfig(
            block_size=self.max_context,
            vocab_size=self.vocab_size,
            n_layer=self.layers,
            n_head=self.heads,
            n_embd=self.width,
            dropout=self.dropout,
            bias=self.bias,
        ))

    @property
    def embedding(self) -> nn.Embedding:
        """Expose the tied upstream token embedding to the shared initializer."""

        return self.gpt.transformer.wte

    @property
    def final_norm(self) -> nn.Module:
        return self.gpt.transformer.ln_f

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the exact upstream tied language-model head."""

        return self.gpt.lm_head(hidden)

    def forward(self, input_ids: torch.Tensor):
        if input_ids.ndim != 2:
            raise ValueError("nanoGPT input_ids must have shape [batch, time]")
        if input_ids.shape[1] > self.max_context:
            raise ValueError(
                "nanoGPT input length exceeds the frozen block size")
        length = int(input_ids.shape[1])
        positions = torch.arange(
            0, length, dtype=torch.long, device=input_ids.device)
        token_embeddings = self.gpt.transformer.wte(input_ids)
        position_embeddings = self.gpt.transformer.wpe(positions)
        hidden = self.gpt.transformer.drop(
            token_embeddings + position_embeddings)
        for block in self.gpt.transformer.h:
            hidden = block(hidden)
        hidden = self.gpt.transformer.ln_f(hidden)
        logits = self.gpt.lm_head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=hidden)

    def output_head_macs_per_token(self) -> int:
        return self.width * self.vocab_size

    def stored_state_elements(self, batch_size: int = 1,
                              context: int = 512) -> int:
        """Theoretical standard autoregressive KV entries, not runtime state."""

        bounded_context = min(int(context), self.max_context)
        return 2 * int(batch_size) * bounded_context * self.width * self.layers

    def state_accounting(self, batch_size: int = 1, context: int = 512,
                         runtime_state=None) -> dict:
        if runtime_state is not None:
            raise RuntimeError(
                "upstream nanoGPT has no reusable incremental KV runtime "
                "state implementation to audit")
        bounded_context = min(int(context), self.max_context)
        elements = self.stored_state_elements(batch_size, bounded_context)
        per_sequence_elements = (
            2 * bounded_context * self.width * self.layers)
        return {
            "state_schema": self.state_schema,
            "architecture": self.kind,
            "upstream_commit": self.upstream_commit,
            "batch_size": int(batch_size),
            "context": bounded_context,
            "layers": self.layers,
            "heads": self.heads,
            "width": self.width,
            "incremental_cache_implemented": False,
            "actual_runtime_measured": False,
            "theoretical_kv_elements": elements,
            "theoretical_kv_elements_per_sequence": per_sequence_elements,
            "theoretical_nominal_bf16_bytes": elements * 2,
            "theoretical_nominal_bf16_bytes_per_sequence": (
                per_sequence_elements * 2),
            "theoretical_fp32_bytes": elements * 4,
            "theoretical_fp32_bytes_per_sequence": (
                per_sequence_elements * 4),
        }

    def estimated_macs_per_token(self, context: int = 512) -> int:
        """Dense inference MACs matching nanoGPT's GPT-2 block topology."""

        bounded_context = min(int(context), self.max_context)
        projections_and_mlp = 12 * self.width * self.width
        attention = 2 * bounded_context * self.width
        return int(
            self.output_head_macs_per_token()
            + self.layers * (projections_and_mlp + attention))
