"""Models for the decisive word-level multiscale comparison.

The recurrent transition is deliberately not reimplemented as a new module.
``PersistentMultiScaleLM`` owns an instance of the exact ``MultiScaleCore``
used by both completed small-scale experiments and only exposes its recurrent
state at chunk boundaries.  The dense Transformer likewise reuses the archived
RoPE Transformer implementation; only its tied, bias-free vocabulary head is
replaced by the untied, biased ``nn.Linear`` convention used in the original
small comparison.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F


_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
AUTHORITATIVE_MULTISCALE_PATH = (
    _EXPERIMENTS_ROOT / "function_weight_wikitext2_benchmark" / "model_core.py"
)
AUTHORITATIVE_TRANSFORMER_PATH = (
    _EXPERIMENTS_ROOT / "function_weight_wt2_original_transformer_port" / "model.py"
)
PACKED2_RECURRENCE_BACKEND = "packed2_mean_then_couple_postnorm_v1"


def _load_authoritative_module(name: str, path: Path) -> ModuleType:
    """Import one authoritative source file without copying it into this bundle."""

    resolved = path.resolve(strict=True)
    specification = importlib.util.spec_from_file_location(name, resolved)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to load authoritative model source: {resolved}")
    module = importlib.util.module_from_spec(specification)
    # Match Python's normal import semantics.  TorchDynamo resolves inherited
    # methods through the defining module name while tracing; executing a
    # module without registering that name leaves eager execution functional
    # but makes the same class impossible for the compiler to import.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


AUTHORITATIVE_MULTISCALE_MODULE = _load_authoritative_module(
    "_wt2_decisive_authoritative_multiscale",
    AUTHORITATIVE_MULTISCALE_PATH,
)
AUTHORITATIVE_TRANSFORMER_MODULE = _load_authoritative_module(
    "_wt2_decisive_authoritative_transformer",
    AUTHORITATIVE_TRANSFORMER_PATH,
)


def expected_multiscale_parameter_count(
    vocab_size: int,
    width: int,
    state_scales: int,
) -> int:
    """Closed-form storage for the exact core plus its untied biased head."""

    return (
        5 * width * width
        + (state_scales + 4) * width
        + 2 * vocab_size * width
        + vocab_size
    )


def expected_dense_rope_transformer_parameter_count(
    vocab_size: int,
    d_model: int,
    n_layers: int,
) -> int:
    """Closed-form storage for dense RoPE blocks and an untied biased head."""

    embeddings_and_head = 2 * vocab_size * d_model + vocab_size
    blocks = n_layers * (12 * d_model * d_model + 4 * d_model)
    final_norm = 2 * d_model
    return embeddings_and_head + blocks + final_norm


class PersistentMultiScaleLM(nn.Module):
    """The original learned-gate multiscale LM with caller-owned state.

    Supplying no state exactly reproduces the original zero-state forward path.
    Supplying the returned state to a later call continues the same recurrence,
    which permits contiguous truncated-BPTT without changing the transition.
    """

    def __init__(
        self,
        vocab_size: int,
        width: int,
        state_scales: int = 4,
        execution_backend: str = PACKED2_RECURRENCE_BACKEND,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or width <= 0 or state_scales <= 0:
            raise ValueError("Vocabulary size, width, and state scales must be positive")
        self.vocab_size = int(vocab_size)
        self.width = int(width)
        self.state_scales = int(state_scales)
        if execution_backend != PACKED2_RECURRENCE_BACKEND:
            raise ValueError("Unsupported original-hierarchy execution backend")
        self.execution_backend = execution_backend
        self.core = AUTHORITATIVE_MULTISCALE_MODULE.MultiScaleCore(
            self.vocab_size,
            self.width,
            {"state_scales": self.state_scales},
        )
        # The original MultiScaleLM uses a separate nn.Linear(width, vocab),
        # including bias.  It is intentionally not tied to the embedding.
        self.output = nn.Linear(self.width, self.vocab_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        reference = self.core.embedding.weight
        return torch.zeros(
            batch_size,
            self.state_scales,
            self.width,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def _validate_state(self, state: torch.Tensor, embedded: torch.Tensor) -> None:
        expected = (embedded.shape[0], self.state_scales, self.width)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"Multiscale state shape {tuple(state.shape)} does not match {expected}"
            )
        if state.device != embedded.device:
            raise ValueError("Multiscale state and token embeddings must share a device")
        if state.dtype != embedded.dtype:
            raise ValueError("Multiscale state and token embeddings must share a dtype")

    @staticmethod
    def detach_state(state: torch.Tensor) -> torch.Tensor:
        """Detach a carried state at a truncated-BPTT boundary."""

        return state.detach()

    def _scan_hidden_tensor(
        self,
        embedded: torch.Tensor,
        state: torch.Tensor,
        *,
        collect_hidden: bool = True,
        collect_retention: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Evaluate the original recurrence with launch-efficient projections.

        The transition is unchanged.  This implementation only applies linear
        identities that are exact over real arithmetic:

        * token-only input/update projections are evaluated for the whole
          chunk before entering the recurrence;
        * the update and gate state projections share one packed GEMM.

        The state trajectory remains sequential because the learned gate and
        tanh update depend nonlinearly on the preceding state.  LayerNorm is
        not part of that dependency, so it is applied once to the stacked
        trajectory after the scan instead of launching once per token.

        Packing creates temporary differentiable tensors from the
        authoritative Parameters on every call.  It adds no Parameters or
        buffers, preserves the original state dict, and routes autograd back
        to the exact original tensors.  The coupling remains a separate
        projection of the mean state, preserving the authoritative operation
        count instead of redundantly projecting every scale.
        """

        input_weight = torch.cat(
            (self.core.input_update.weight, self.core.input_gate.weight),
            dim=0,
        )
        input_bias = torch.cat(
            (self.core.input_update.bias, self.core.input_gate.bias),
            dim=0,
        )
        input_terms = F.linear(embedded, input_weight, input_bias)
        input_update_terms, input_gate_terms = input_terms.split(
            self.width,
            dim=-1,
        )

        recurrent_weight = torch.cat(
            (
                self.core.state_update.weight,
                self.core.state_gate.weight,
            ),
            dim=0,
        )
        scale = self.core.scale[None, :, :]
        mean_states: list[torch.Tensor] = []
        retention_sums = (
            torch.zeros(
                self.state_scales,
                device=embedded.device,
                dtype=torch.float32,
            )
            if collect_retention
            else None
        )
        for index in range(embedded.shape[1]):
            recurrent_terms = F.linear(state, recurrent_weight)
            state_update_term, state_gate_term = (
                recurrent_terms.split(self.width, dim=-1)
            )
            coupled_global_state = self.core.coupling(
                state.mean(dim=1, keepdim=True)
            )
            update = torch.tanh(
                input_update_terms[:, index, None, :]
                + state_update_term
                + coupled_global_state
                + scale
            )
            gate = torch.sigmoid(
                input_gate_terms[:, index, None, :]
                + state_gate_term
                - scale
            )
            if retention_sums is not None:
                retention_sums = retention_sums + gate.float().sum(dim=(0, 2))
            state = gate * state + (1.0 - gate) * update
            if collect_hidden:
                mean_states.append(state.mean(dim=1))

        if not collect_hidden:
            return None, state, retention_sums
        trajectory = torch.stack(mean_states, dim=1)
        return self.core.norm(trajectory), state, retention_sums

    def advance_state(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Advance only the original recurrent state for a detached boundary pass."""

        if tokens.ndim != 2:
            raise ValueError("Tokens must have shape [batch, time]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("Tokens require nonempty batch and time dimensions")
        embedded = self.core.embedding(tokens)
        self._validate_state(state, embedded)
        _, next_state, _ = self._scan_hidden_tensor(
            embedded,
            state,
            collect_hidden=False,
        )
        return next_state

    def scan_with_state(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("Tokens must have shape [batch, time]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("Tokens require nonempty batch and time dimensions")

        embedded = self.core.embedding(tokens)
        if state is None:
            state = self.initial_state(
                embedded.shape[0],
                device=embedded.device,
                dtype=embedded.dtype,
            )
        self._validate_state(state, embedded)

        hidden, state, _ = self._scan_hidden_tensor(embedded, state)
        assert hidden is not None
        return list(hidden.unbind(dim=1)), state

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("Tokens must have shape [batch, time]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("Tokens require nonempty batch and time dimensions")
        embedded = self.core.embedding(tokens)
        if state is None:
            state = self.initial_state(
                embedded.shape[0],
                device=embedded.device,
                dtype=embedded.dtype,
            )
        self._validate_state(state, embedded)
        hidden_values, next_state, _ = self._scan_hidden_tensor(embedded, state)
        assert hidden_values is not None
        # The vocabulary projection has no recurrent dependency.  Stack the
        # hidden trajectory first so CUDA can execute one large GEMM instead
        # of one small GEMM per recurrent step.  This preserves the exact
        # MultiScaleCore transition and the ordinary untied Linear head.
        logits = self.output(hidden_values)
        if return_state:
            return logits, next_state
        return logits

    def forward_with_retention(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the same FP32 path while summing every learned retention gate."""

        if tokens.ndim != 2:
            raise ValueError("Tokens must have shape [batch, time]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("Tokens require nonempty batch and time dimensions")
        embedded = self.core.embedding(tokens)
        self._validate_state(state, embedded)
        hidden, next_state, retention_sums = self._scan_hidden_tensor(
            embedded,
            state,
            collect_retention=True,
        )
        assert hidden is not None and retention_sums is not None
        return self.output(hidden), next_state, retention_sums


class DenseRoPETransformer(AUTHORITATIVE_TRANSFORMER_MODULE.GPT):
    """Archived dense RoPE Transformer with the small-test output convention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        context_length: int,
    ) -> None:
        if min(vocab_size, d_model, n_heads, n_layers, context_length) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("Transformer width must divide evenly into heads")
        if (d_model // n_heads) % 2:
            raise ValueError("RoPE requires an even attention-head width")
        super().__init__(
            vocab=int(vocab_size),
            d=int(d_model),
            nh=int(n_heads),
            L=int(n_layers),
            ctx=int(context_length),
            mode="full",
            r=0,
        )
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.context_length = int(context_length)
        # Reassignment removes the archived tied, bias-free head.  PyTorch's
        # ordinary Linear initialization and bias match the original small
        # Transformer output convention.
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("Tokens must have shape [batch, time]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("Tokens require nonempty batch and time dimensions")
        if tokens.shape[1] > self.context_length:
            raise ValueError("Token sequence exceeds the configured Transformer context")
        return super().forward(tokens)


__all__ = [
    "AUTHORITATIVE_MULTISCALE_MODULE",
    "AUTHORITATIVE_MULTISCALE_PATH",
    "AUTHORITATIVE_TRANSFORMER_MODULE",
    "AUTHORITATIVE_TRANSFORMER_PATH",
    "DenseRoPETransformer",
    "PACKED2_RECURRENCE_BACKEND",
    "PersistentMultiScaleLM",
    "expected_dense_rope_transformer_parameter_count",
    "expected_multiscale_parameter_count",
]
