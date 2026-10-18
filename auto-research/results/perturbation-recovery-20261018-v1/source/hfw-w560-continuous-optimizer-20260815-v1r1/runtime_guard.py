from __future__ import annotations

from typing import Mapping

import torch


class NoTrainingRuntimeGuard:
    def __init__(self) -> None:
        self.counters = {
            "backward_attempts": 0,
            "autograd_grad_attempts": 0,
            "optimizer_construction_attempts": 0,
            "optimizer_step_attempts": 0,
        }
        self._originals = []

    def _block(self, counter: str):
        def blocked(*args, **kwargs):
            del args, kwargs
            self.counters[counter] += 1
            raise RuntimeError(f"production no-training runtime gate blocked {counter}")
        return blocked

    def _patch(self, owner, name: str, replacement) -> None:
        if hasattr(owner, name):
            self._originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

    def __enter__(self):
        self._patch(torch.Tensor, "backward", self._block("backward_attempts"))
        self._patch(torch.autograd, "backward", self._block("backward_attempts"))
        self._patch(torch.autograd, "grad", self._block("autograd_grad_attempts"))
        optimizer_base = torch.optim.Optimizer
        self._patch(optimizer_base, "__init__", self._block("optimizer_construction_attempts"))
        pending = [optimizer_base]
        seen = set()
        while pending:
            optimizer = pending.pop()
            if optimizer in seen:
                continue
            seen.add(optimizer)
            pending.extend(optimizer.__subclasses__())
            if "step" in optimizer.__dict__:
                self._patch(optimizer, "step", self._block("optimizer_step_attempts"))
        return self

    def __exit__(self, exception_type, exception, traceback):
        del exception_type, exception, traceback
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        return False

    def receipt(self) -> Mapping[str, object]:
        return {
            "status": "PASS_NO_TRAINING_RUNTIME_GUARD" if not any(self.counters.values()) else "BLOCK",
            **self.counters,
        }
