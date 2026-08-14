"""
intervention.py -- read/write access to the residual stream via forward hooks.

Design decisions (per the agreed protocol):

  * ONE layer convention. Both reading and writing use a forward hook on the
    SAME decoder-layer module, operating on that block's OUTPUT hidden state
    (resid_post). This avoids the off-by-one that appears if you mix
    output_hidden_states (embeddings = index 0) with hooks (block output).

  * Read and write are SEPARATE, context-managed objects with guaranteed
    cleanup (handles removed on __exit__), so a capture can never accidentally
    write, and a stale write-hook can never contaminate a control run.

  * Interventions target the FINAL token position only by default.

  * Perturbation magnitude is defined RELATIVE to the captured activation norm,
    because residual norms (especially Gemma's) are large and an absolute unit
    perturbation would be negligible.

A decoder layer's forward returns either a bare tensor or a tuple whose first
element is the hidden state. We handle both and preserve the container type.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch

# An op maps the selected hidden-state slice [batch, hidden] -> [batch, hidden].
HiddenOp = Callable[[torch.Tensor], torch.Tensor]


def _split_output(output):
    """Return (hidden_state_tensor, rebuild_fn) for a layer's forward output."""
    if isinstance(output, tuple):
        rest = output[1:]
        return output[0], (lambda new_hs: (new_hs, *rest))
    return output, (lambda new_hs: new_hs)


class ReadHook:
    """Capture (clone of) a block's output hidden state. Read-only."""

    def __init__(self, layer_module: torch.nn.Module):
        self.layer_module = layer_module
        self.handle = None
        self.activation: torch.Tensor | None = None  # [batch, seq, hidden]

    def _hook(self, module, inputs, output):
        hs, _ = _split_output(output)
        self.activation = hs.detach().clone()
        return None  # read-only: never alter the forward output

    def __enter__(self) -> "ReadHook":
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def last_token(self) -> torch.Tensor:
        assert self.activation is not None, "no activation captured yet"
        return self.activation[:, -1, :].clone()  # [batch, hidden]


class WriteHook:
    """Apply an op to one position of a block's output hidden state.

    op receives the slice at `position` (shape [batch, hidden]) and returns its
    replacement. Only that position is modified; all others pass through
    unchanged. Records the pre/post norm at the modified position for telemetry.
    """

    def __init__(self, layer_module: torch.nn.Module, op: HiddenOp, position: int = -1):
        self.layer_module = layer_module
        self.op = op
        self.position = position
        self.handle = None
        self.norm_before: float | None = None
        self.norm_after: float | None = None
        self.fired = False

    def _hook(self, module, inputs, output):
        hs, rebuild = _split_output(output)
        # hs: [batch, seq, hidden]
        sl = hs[:, self.position, :]
        self.norm_before = float(sl.norm(dim=-1).mean().item())
        new_sl = self.op(sl)
        self.norm_after = float(new_sl.norm(dim=-1).mean().item())
        hs = hs.clone()
        hs[:, self.position, :] = new_sl.to(hs.dtype)
        self.fired = True
        return rebuild(hs)

    def __enter__(self) -> "WriteHook":
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


# ---------------------------------------------------------------------------
# Ops / delta builders
# ---------------------------------------------------------------------------

def op_identity() -> HiddenOp:
    """alpha=0 negative control: exercises the write path but changes nothing."""
    return lambda h: h


def op_zero() -> HiddenOp:
    """Ablate the activation at the target position."""
    return lambda h: torch.zeros_like(h)


def random_unit_direction(hidden_size: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """A fixed unit vector in R^hidden. Reused across the alpha sweep so the
    sweep is a clean dose-response along ONE direction."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(hidden_size, generator=gen, dtype=torch.float32)
    d = d / d.norm()
    return d.to(device=device, dtype=dtype)


def op_add_scaled_direction(direction: torch.Tensor, alpha: float) -> HiddenOp:
    """h -> h + alpha * ||h|| * direction  (norm-relative random perturbation)."""
    def _op(h: torch.Tensor) -> torch.Tensor:
        norm = h.norm(dim=-1, keepdim=True)  # [batch, 1]
        return h + alpha * norm * direction.view(1, -1)
    return _op


def op_add_self(alpha: float) -> HiddenOp:
    """h -> (1 + alpha) * h. Deliberately strong STRUCTURED intervention:
    amplifies the model's own activation direction, guaranteed to move logits
    if the write path is live. Distinguishes 'plumbing works' from 'this random
    delta happened to be inert'."""
    return lambda h: (1.0 + alpha) * h


def op_replace(vector: torch.Tensor) -> HiddenOp:
    """h -> vector (broadcast over batch). Used to inject another prompt's
    captured activation at the same layer/position."""
    def _op(h: torch.Tensor) -> torch.Tensor:
        return vector.view(1, -1).to(h.dtype).expand_as(h).clone()
    return _op


@contextmanager
def no_hook():
    """Trivial context so control runs read symmetric with intervention runs."""
    yield None
