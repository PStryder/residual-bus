"""
carto_lib.py -- Phase 2A capture engine + metrics.

Reuses the Phase 1 primitives (model.py, intervention.py) unchanged. Adds:

  * full-stack capture: resid_post at EVERY decoder layer, EVERY token position,
    in a single forward pass, via ReadHooks on all layers.
  * a combined write+capture runner: apply ONE WriteHook at (layer, position)
    and simultaneously capture the full post-intervention activation stack, with
    explicit hook ordering so the target layer's captured state is POST-write.
  * logit metrics (KL, entropy, top-k overlap, answer contrast).
  * downstream propagation metrics (delta norm/ratio/cosine, causal invariants,
    alignment to a natural A->B trajectory, distance-to-B improvement).

LAYER CONVENTION (identical to Phase 1):
  H[L] = output hidden state of decoder block L = resid_post of layer L
       = HuggingFace hidden_states[L+1]  (hidden_states[0] would be embeddings).
  A WriteHook at layer L modifies resid_post[L]; blocks L+1.. then reprocess it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import model as M
import intervention as IV


# ---------------------------------------------------------------------------
# capture engine
# ---------------------------------------------------------------------------

@dataclass
class CaptureResult:
    logits: torch.Tensor          # [vocab] final-position next-token logits (gpu)
    H: torch.Tensor               # [num_layers, seq, hidden] resid_post stack (gpu)
    norm_before: float | None     # write-site norm pre-op (mean over batch)
    norm_after: float | None      # write-site norm post-op


@torch.no_grad()
def run_and_capture_all(
    lm: M.LoadedModel,
    input_ids: torch.Tensor,
    write=None,   # None or (target_layer:int, position:int, op:IV.HiddenOp)
) -> CaptureResult:
    """Single forward pass. Captures resid_post at all layers/positions.

    If `write` is given, a WriteHook is applied at (target_layer, position) and
    the target layer's ReadHook is registered AFTER the WriteHook so the captured
    layer-L state reflects the POST-write output. Downstream layers naturally
    reflect the modified stream. No KV cache.
    """
    layers = lm.layers
    n = len(layers)
    reads = {L: IV.ReadHook(layers[L]) for L in range(n)}
    whook = None
    entered = []
    try:
        if write is not None:
            tL, pos, op = write
            whook = IV.WriteHook(layers[tL], op, position=pos)
            whook.__enter__(); entered.append(whook)   # register FIRST on target module
        for L in range(n):
            reads[L].__enter__(); entered.append(reads[L])  # target read registers after write
        out = lm.model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].detach()
        H = torch.stack([reads[L].activation[0].detach() for L in range(n)], dim=0)  # [n,seq,hidden]
    finally:
        for h in reversed(entered):
            h.__exit__()
    return CaptureResult(
        logits=logits,
        H=H,
        norm_before=(whook.norm_before if whook else None),
        norm_after=(whook.norm_after if whook else None),
    )


def random_direction(hidden: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Deterministic unit vector, seeded per (site) so each cell gets its own
    generic direction (a single global direction can be pathologically aligned
    or orthogonal at particular sites)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    d = torch.randn(hidden, generator=gen, dtype=torch.float32)
    d = d / d.norm()
    return d.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# logit metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) in nats, P from p_logits."""
    lp = torch.log_softmax(p_logits.double(), dim=-1)
    lq = torch.log_softmax(q_logits.double(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum().item())


@torch.no_grad()
def entropy(logits: torch.Tensor) -> float:
    lp = torch.log_softmax(logits.double(), dim=-1)
    return float(-(lp.exp() * lp).sum().item())


@torch.no_grad()
def topk_overlap(a_logits: torch.Tensor, b_logits: torch.Tensor, k: int = 10) -> float:
    ta = set(torch.topk(a_logits, k).indices.tolist())
    tb = set(torch.topk(b_logits, k).indices.tolist())
    return len(ta & tb) / k


@torch.no_grad()
def logit_metrics(control: torch.Tensor, other: torch.Tensor, tokenizer, k: int = 10) -> dict:
    top_c = int(torch.argmax(control).item())
    top_o = int(torch.argmax(other).item())
    diff = (control.double() - other.double()).abs()
    return {
        "kl_from_control": kl_div(control, other),
        "max_abs_logit_diff": float(diff.max().item()),
        "l2_logit_diff": float((control.double() - other.double()).norm().item()),
        "top1_id": top_o,
        "top1_tok": tokenizer.decode([top_o]).strip(),
        "top1_changed": top_o != top_c,
        "topk_overlap": topk_overlap(control, other, k),
        "entropy": entropy(other),
    }


@torch.no_grad()
def answer_contrast(logits: torch.Tensor, tA: int, tB: int) -> dict:
    """C = logit(tB) - logit(tA). Also log-probs / probs of each answer token."""
    d = logits.double()
    lp = torch.log_softmax(d, dim=-1)
    return {
        "C": float((d[tB] - d[tA]).item()),
        "logit_A": float(d[tA].item()),
        "logit_B": float(d[tB].item()),
        "logp_A": float(lp[tA].item()),
        "logp_B": float(lp[tB].item()),
        "p_A": float(lp[tA].exp().item()),
        "p_B": float(lp[tB].exp().item()),
    }


# ---------------------------------------------------------------------------
# downstream propagation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def causal_invariants(H_base: torch.Tensor, H_prime: torch.Tensor, tL: int, pos: int) -> dict:
    """Given an intervention at (tL, pos), check that:
      * every layer strictly below tL is unchanged everywhere;
      * every token position strictly before `pos` is unchanged at every layer
        (causal attention -> earlier tokens cannot see a later-token edit).
    Returns bitwise-zero booleans + the max abs deviation observed.
    """
    delta = (H_prime - H_base)
    lower = delta[:tL] if tL > 0 else delta[:0]
    earlier = delta[:, :pos] if pos > 0 else delta[:, :0]
    lower_max = float(lower.abs().max().item()) if lower.numel() else 0.0
    earlier_max = float(earlier.abs().max().item()) if earlier.numel() else 0.0
    return {
        "lower_layers_zero": lower_max == 0.0,
        "earlier_positions_zero": earlier_max == 0.0,
        "lower_layers_max_abs": lower_max,
        "earlier_positions_max_abs": earlier_max,
    }


@torch.no_grad()
def propagation_perturb(H_base: torch.Tensor, H_prime: torch.Tensor, tL: int, pos: int) -> dict:
    """Generic (non-semantic) propagation summary for a perturbation cell.

    n=num_layers. Focus positions: the intervened `pos` and the final position.
    All 'downstream' aggregates are over layers L >= tL.
    """
    n, seq, hid = H_base.shape
    last = n - 1
    fin = seq - 1
    delta = H_prime - H_base                         # [n,seq,hid]
    dnorm = delta.norm(dim=-1)                        # [n,seq]
    bnorm = H_base.norm(dim=-1).clamp_min(1e-12)      # [n,seq]
    rel = dnorm / bnorm                               # [n,seq]

    # cosine(H_prime, H_base) at final position over downstream layers (rotation)
    down = slice(tL, n)
    cos_fin = torch.nn.functional.cosine_similarity(
        H_prime[down, fin], H_base[down, fin], dim=-1)  # [n-tL]

    # spread: fraction of positions strictly after `pos` that are affected
    # (rel-delta > 1e-4) at the last layer.
    after = rel[last, pos + 1:] if pos + 1 < seq else rel[last, 0:0]
    spread_frac = float((after > 1e-4).float().mean().item()) if after.numel() else 0.0

    return {
        "rel_delta_site_last_layer": float(rel[last, pos].item()),
        "rel_delta_finalpos_last_layer": float(rel[last, fin].item()),
        "min_cos_finalpos_downstream": float(cos_fin.min().item()),
        "mean_cos_finalpos_downstream": float(cos_fin.mean().item()),
        "spread_frac_after_site_last_layer": spread_frac,
    }


@torch.no_grad()
def propagation_semantic(H_A: torch.Tensor, H_B: torch.Tensor, H_prime: torch.Tensor,
                         tL: int, pos: int) -> dict:
    """Does patching (tL,pos) move A's downstream computation along the natural
    A->B trajectory?  D_natural = H_B - H_A ; D_induced = H_prime - H_A.

    Aggregates over downstream layers (L > tL) at the FINAL position (where the
    answer is read) and also at the intervened `pos`.  Distance-to-B improvement
    is normalized:  (||H_A - H_B|| - ||H_prime - H_B||) / ||H_A - H_B||  (in [-inf,1];
    positive => moved closer to B).
    """
    n, seq, hid = H_A.shape
    fin = seq - 1
    D_nat = H_B - H_A
    D_ind = H_prime - H_A

    def at(layer_slice, p):
        dn = D_nat[layer_slice, p]      # [k,hid]
        di = D_ind[layer_slice, p]
        cos = torch.nn.functional.cosine_similarity(di, dn, dim=-1)          # [k]
        ratio = di.norm(dim=-1) / dn.norm(dim=-1).clamp_min(1e-12)           # [k]
        dist_A = (H_A[layer_slice, p] - H_B[layer_slice, p]).norm(dim=-1)    # [k]
        dist_P = (H_prime[layer_slice, p] - H_B[layer_slice, p]).norm(dim=-1)
        improve = (dist_A - dist_P) / dist_A.clamp_min(1e-12)                # [k]
        return cos, ratio, improve

    down = slice(tL + 1, n)  # strictly downstream layers
    if tL + 1 >= n:
        # patch at last layer: no downstream layers
        empty = torch.zeros(1)
        return {
            "n_downstream_layers": 0,
            "mean_cos_finalpos": 0.0, "mean_ratio_finalpos": 0.0, "mean_distimprove_finalpos": 0.0,
            "mean_cos_site": 0.0, "mean_ratio_site": 0.0, "mean_distimprove_site": 0.0,
        }

    cos_f, ratio_f, imp_f = at(down, fin)
    cos_s, ratio_s, imp_s = at(down, pos)
    return {
        "n_downstream_layers": int(n - (tL + 1)),
        "mean_cos_finalpos": float(cos_f.mean().item()),
        "mean_ratio_finalpos": float(ratio_f.mean().item()),
        "mean_distimprove_finalpos": float(imp_f.mean().item()),
        "mean_cos_site": float(cos_s.mean().item()),
        "mean_ratio_site": float(ratio_s.mean().item()),
        "mean_distimprove_site": float(imp_s.mean().item()),
    }


@torch.no_grad()
def propagation_trace_semantic(H_A: torch.Tensor, H_B: torch.Tensor, H_prime: torch.Tensor,
                               pos: int) -> dict:
    """Full per-layer propagation trace at the final position (for top sites).
    Returns python lists keyed by layer index."""
    n, seq, hid = H_A.shape
    fin = seq - 1
    D_nat = H_B - H_A
    D_ind = H_prime - H_A
    cos = torch.nn.functional.cosine_similarity(D_ind[:, fin], D_nat[:, fin], dim=-1)  # [n]
    ratio = D_ind[:, fin].norm(dim=-1) / D_nat[:, fin].norm(dim=-1).clamp_min(1e-12)
    dist_A = (H_A[:, fin] - H_B[:, fin]).norm(dim=-1)
    dist_P = (H_prime[:, fin] - H_B[:, fin]).norm(dim=-1)
    improve = (dist_A - dist_P) / dist_A.clamp_min(1e-12)
    return {
        "layer": list(range(n)),
        "cos_finalpos": cos.tolist(),
        "ratio_finalpos": ratio.tolist(),
        "distimprove_finalpos": improve.tolist(),
    }
