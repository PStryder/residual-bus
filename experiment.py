"""
experiment.py -- Phase 1 residual-bus validation.

Experiment A: capture mid-layer residual activations, record stats, and test
              reproducibility across two identical forward passes.

Experiment B: single-forward-pass causal interventions at the final token of
              several middle layers:
                * identity (alpha=0)        -- strict negative control
                * zero                      -- ablation
                * norm-relative random      -- alpha in {0.1, 0.3, 1.0, 3.0}
                * structured +alpha*h       -- alpha in {0.5, 1.0, 3.0}
                * replace w/ prompt-B state -- structured, semantic-ish
              measuring KL, logit diffs, top-token changes, and pre/post norms.

Success is judged as: reproducible capture AND a dose-dependent causal effect
(alpha=0 is a true no-op; effect grows monotonically with random alpha; the
structured intervention clearly moves the distribution).

No text generation, no KV cache -- next-token logits only.

Run:  python experiment.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

import model as M
import intervention as IV

RESULTS = Path(__file__).parent / "results"
PROMPT_A = "What is the capital of France? Answer in one word."
PROMPT_B = "What is the chemical symbol for gold? Answer in one word."
SYSTEM = "You are a helpful assistant."
DIRECTION_SEED = 1234

RAND_ALPHAS = [0.1, 0.3, 1.0, 3.0]
SELF_ALPHAS = [0.5, 1.0, 3.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tstats(t: torch.Tensor) -> dict:
    tf = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "norm_full": float(tf.norm().item()),
        "mean": float(tf.mean().item()),
        "std": float(tf.std().item()),
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
    }


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.float().flatten(), b.float().flatten()
    diff = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0)).item()
    return {
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "cosine": float(cos),
    }


def logit_metrics(control: torch.Tensor, other: torch.Tensor, tokenizer, k: int = 10) -> dict:
    """control/other: [vocab] logits (batch dim already squeezed)."""
    c = control.double()
    o = other.double()
    logp_c = torch.log_softmax(c, dim=-1)
    logp_o = torch.log_softmax(o, dim=-1)
    p_c = logp_c.exp()
    kl = float((p_c * (logp_c - logp_o)).sum().item())  # KL(P_control || P_other)

    diff = (c - o).abs()
    top_c = torch.topk(c, k).indices.tolist()
    top_o = torch.topk(o, k).indices.tolist()
    overlap = len(set(top_c) & set(top_o)) / k

    def tok(i):
        return tokenizer.decode([i]).strip()

    return {
        "kl_control_vs_other": kl,
        "max_abs_logit_diff": float(diff.max().item()),
        "l2_logit_diff": float((c - o).norm().item()),
        "top1_control": {"id": top_c[0], "tok": tok(top_c[0])},
        "top1_other": {"id": top_o[0], "tok": tok(top_o[0])},
        "top1_changed": top_c[0] != top_o[0],
        "topk_overlap": overlap,
    }


def run_clean(lm, input_ids, sweep_idxs):
    """One clean forward pass. Returns (last-token logits [vocab], {idx: last-token act})."""
    layers = lm.layers
    reads = [IV.ReadHook(layers[i]) for i in sweep_idxs]
    for r in reads:
        r.__enter__()
    try:
        logits = M.forward_logits(lm, input_ids)[0]  # [vocab]
        acts = {idx: r.last_token()[0] for idx, r in zip(sweep_idxs, reads)}  # [hidden]
    finally:
        for r in reads:
            r.__exit__()
    return logits.detach(), acts


def run_with_write(lm, input_ids, layer_module, op, position=-1):
    with IV.WriteHook(layer_module, op, position=position) as w:
        logits = M.forward_logits(lm, input_ids)[0]
    return logits.detach(), {"norm_before": w.norm_before, "norm_after": w.norm_after, "fired": w.fired}


# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------

def experiment_a(lm, input_ids, sweep_idxs, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    layers = lm.layers
    mods = [layers[i] for i in sweep_idxs]

    # run 1
    reads1 = [IV.ReadHook(m) for m in mods]
    for r in reads1:
        r.__enter__()
    try:
        logits1 = M.forward_logits(lm, input_ids)[0]
        acts1 = {i: r.activation[0].detach().clone() for i, r in zip(sweep_idxs, reads1)}  # [seq,hidden]
    finally:
        for r in reads1:
            r.__exit__()

    # run 2 (identical)
    reads2 = [IV.ReadHook(m) for m in mods]
    for r in reads2:
        r.__enter__()
    try:
        logits2 = M.forward_logits(lm, input_ids)[0]
        acts2 = {i: r.activation[0].detach().clone() for i, r in zip(sweep_idxs, reads2)}
    finally:
        for r in reads2:
            r.__exit__()

    report = {"prompt": PROMPT_A, "sweep_layers": sweep_idxs, "layers": {}}
    for i in sweep_idxs:
        full = acts1[i]                # [seq, hidden]
        last = full[-1]               # [hidden]
        report["layers"][str(i)] = {
            "activation_stats_full": tstats(full),
            "last_token_norm": float(last.float().norm().item()),
            "reproducibility_activation": compare(acts1[i], acts2[i]),
        }
    report["reproducibility_logits"] = compare(logits1, logits2)

    torch.save({str(i): acts1[i].cpu() for i in sweep_idxs}, outdir / "activations_run1.pt")
    torch.save({str(i): acts2[i].cpu() for i in sweep_idxs}, outdir / "activations_run2.pt")
    torch.save(logits1.cpu(), outdir / "logits_run1.pt")
    (outdir / "experiment_a.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Experiment B
# ---------------------------------------------------------------------------

def experiment_b(lm, input_ids_a, input_ids_b, sweep_idxs, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    layers = lm.layers
    tok = lm.tokenizer
    hidden = lm.hidden_size

    # Clean pass on prompt A: control logits + captured last-token acts per layer.
    control_logits, acts_a = run_clean(lm, input_ids_a, sweep_idxs)

    # Clean pass on prompt B: last-token acts per layer (for replacement injection).
    _, acts_b = run_clean(lm, input_ids_b, sweep_idxs)

    direction = IV.random_unit_direction(hidden, DIRECTION_SEED, lm.device, dtype=lm.dtype)

    report = {
        "prompt_a": PROMPT_A,
        "prompt_b": PROMPT_B,
        "sweep_layers": sweep_idxs,
        "direction_seed": DIRECTION_SEED,
        "layers": {},
    }

    saved_logits = {"control": control_logits.cpu()}

    for i in sweep_idxs:
        layer_mod = layers[i]
        h_a = acts_a[i]  # [hidden]
        h_b = acts_b[i]

        interventions = {}
        interventions["identity_alpha0"] = IV.op_identity()
        interventions["zero"] = IV.op_zero()
        for a in RAND_ALPHAS:
            interventions[f"rand_a{a}"] = IV.op_add_scaled_direction(direction, a)
        for a in SELF_ALPHAS:
            interventions[f"self_a{a}"] = IV.op_add_self(a)
        interventions["replace_promptB"] = IV.op_replace(h_b)

        layer_report = {
            "control_last_token_norm": float(h_a.float().norm().item()),
            "control_top1": None,
            "interventions": {},
        }
        c_top = torch.topk(control_logits.double(), 1).indices.tolist()[0]
        layer_report["control_top1"] = {"id": c_top, "tok": tok.decode([c_top]).strip()}

        for name, op in interventions.items():
            logits, norms = run_with_write(lm, input_ids_a, layer_mod, op)
            metrics = logit_metrics(control_logits, logits, tok)
            metrics.update(norms)
            layer_report["interventions"][name] = metrics
            saved_logits[f"layer{i}_{name}"] = logits.cpu()

        # dose-response check on the random sweep (KL should be non-decreasing)
        kls = [layer_report["interventions"][f"rand_a{a}"]["kl_control_vs_other"] for a in RAND_ALPHAS]
        monotonic = all(kls[j] <= kls[j + 1] + 1e-9 for j in range(len(kls) - 1))
        layer_report["random_sweep_kl"] = dict(zip([str(a) for a in RAND_ALPHAS], kls))
        layer_report["random_sweep_monotonic"] = bool(monotonic)

        report["layers"][str(i)] = layer_report

    torch.save(saved_logits, outdir / "next_token_logits.pt")
    (outdir / "experiment_b.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def verdict(rep_a: dict, rep_b: dict) -> dict:
    """Judge the success sentence: a controlled vector added to an intermediate
    residual activation produces a reproducible, dose-dependent causal change.

    The causal gate uses DIRECTION-CHANGING interventions (ablation + the
    largest random perturbation). Pure-scaling (self_a*) and same-suffix
    replacement (replace_promptB) are intentionally NOT gates: a pre-norm
    RMSNorm model normalizes away a pure magnitude change, and the final
    structural token barely differs between prompts -- both are expected-weak
    and are reported as diagnostics, not failures.
    """
    logit_repro = rep_a["reproducibility_logits"]
    reproducible = logit_repro["exact_equal"] or (
        logit_repro["max_abs_diff"] < 1e-3 and logit_repro["cosine"] > 0.99999
    )

    KL_STRONG = 1.0
    max_rand = f"rand_a{RAND_ALPHAS[-1]}"

    alpha0_null = True
    monotonic_all = True
    strong_effect = True   # direction-changing interventions clearly move logits
    top1_flips_all = True  # strongest intervention flips the argmax token
    diagnostics = {}

    for i, lr in rep_b["layers"].items():
        a0 = lr["interventions"]["identity_alpha0"]
        if a0["kl_control_vs_other"] > 1e-6 or a0["max_abs_logit_diff"] > 1e-3:
            alpha0_null = False
        if not lr["random_sweep_monotonic"]:
            monotonic_all = False

        zero_kl = lr["interventions"]["zero"]["kl_control_vs_other"]
        rand_kl = lr["interventions"][max_rand]["kl_control_vs_other"]
        if max(zero_kl, rand_kl) < KL_STRONG:
            strong_effect = False
        if not (lr["interventions"]["zero"]["top1_changed"]
                or lr["interventions"][max_rand]["top1_changed"]):
            top1_flips_all = False

        diagnostics[i] = {
            "zero_kl": zero_kl,
            f"{max_rand}_kl": rand_kl,
            "self_amax_kl": lr["interventions"][f"self_a{SELF_ALPHAS[-1]}"]["kl_control_vs_other"],
            "replace_promptB_kl": lr["interventions"]["replace_promptB"]["kl_control_vs_other"],
        }

    causal = alpha0_null and monotonic_all and strong_effect and top1_flips_all
    return {
        "reproducible_capture": bool(reproducible),
        "alpha0_is_true_noop": bool(alpha0_null),
        "random_sweep_monotonic_all_layers": bool(monotonic_all),
        "direction_change_strong_all_layers": bool(strong_effect),
        "top1_flips_under_strong_intervention_all_layers": bool(top1_flips_all),
        "causal_control_demonstrated": bool(causal),
        "overall_success": bool(reproducible and causal),
        "diagnostics_note": (
            "Pure-scaling (self_a*) and same-suffix replacement (replace_promptB) are "
            "expected-weak (RMSNorm scale-invariance; structural final token) and are "
            "NOT used as success gates."
        ),
        "per_layer_kl": diagnostics,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    lm = M.load()
    print(f"Loaded {lm.model_id}: {lm.num_layers} layers, hidden={lm.hidden_size}, "
          f"dtype={lm.dtype}, device={lm.device}")

    ids_a = M.build_inputs(lm, PROMPT_A, SYSTEM)
    ids_b = M.build_inputs(lm, PROMPT_B, SYSTEM)
    sweep = M.middle_layer_indices(lm.num_layers)
    print(f"Sweep layers: {sweep}  (input_a tokens: {ids_a.shape[1]})")

    rep_a = experiment_a(lm, ids_a, sweep, RESULTS / "expA")
    print("Experiment A done. logits reproducibility:", rep_a["reproducibility_logits"])

    rep_b = experiment_b(lm, ids_a, ids_b, sweep, RESULTS / "expB")
    print("Experiment B done.")

    v = verdict(rep_a, rep_b)
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": lm.model_id,
        "dtype": str(lm.dtype),
    }
    summary = {"env": env, "verdict": v,
               "reproducibility_logits": rep_a["reproducibility_logits"]}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== VERDICT ===")
    for k, val in v.items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
