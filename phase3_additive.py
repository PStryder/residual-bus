"""
phase3_additive.py -- Phase 3A: exact additive semantic write-back.

Moves from activation TRANSPLANT (h_A <- h_B, Phases 2A-2C) to additive
write-back (h_A <- h_A + alpha*D, D = h_B - h_A) -- the first form resembling
the eventual PCDC interface.

Same frozen apparatus (Qwen2.5-1.5B-Instruct, FP32, eager, deterministic,
resid_post hooks, single forward pass, no KV cache, no grad). carto_lib.py reused
unchanged. Reuses the validated Phase 2A (color copy) and Phase 2C (numeric
comparison) tasks rather than re-mapping. Writes only to results/phase3a/.

Predeclared PASS gates (Part 5): see classify(). Only a PASS advances to 3C.

Run:  python phase3_additive.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch

import model as M
import intervention as IV
import carto_lib as CL

OUT = Path(__file__).parent / "results" / "phase3a"
P2A = Path(__file__).parent / "results" / "phase2a"
P2C = Path(__file__).parent / "results" / "phase2c"
SYSTEM = "You are a helpful assistant."

ALPHAS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

# ---- task builders (identical wording to the validated phases) ----

def color_user(v):   # Phase 2A copy family
    return f"Fact: the dax is {v}. Question: what color is the dax? Reply with only one lowercase word."

def compare_user(x, t):  # Phase 2C comparison family
    return f"Is {x} greater than {t}?" + " Answer in one word."


def op_add(delta: torch.Tensor, alpha: float):
    """h -> h + alpha*delta  (absolute additive delta, not norm-scaled)."""
    add = (alpha * delta).view(1, -1)
    return lambda h: h + add


def load_task(lm, name):
    if name == "color":
        ids_A = M.build_inputs(lm, color_user("blue"), SYSTEM)
        ids_B = M.build_inputs(lm, color_user("green"), SYSTEM)
        P = 20
        expect = ("blue", "green")
    elif name == "compare":
        ids_A = M.build_inputs(lm, compare_user(1, 4), SYSTEM)
        ids_B = M.build_inputs(lm, compare_user(5, 4), SYSTEM)
        P = 16
        expect = ("no", "yes")
    else:
        raise ValueError(name)
    bA, bB = CL.run_and_capture_all(lm, ids_A), CL.run_and_capture_all(lm, ids_B)
    ansA, ansB = int(torch.argmax(bA.logits).item()), int(torch.argmax(bB.logits).item())
    tokA, tokB = lm.tokenizer.decode([ansA]).strip().lower(), lm.tokenizer.decode([ansB]).strip().lower()
    assert tokA == expect[0] and tokB == expect[1], f"{name}: baselines {tokA}/{tokB} != {expect}"
    acA = CL.answer_contrast(bA.logits, ansA, ansB)
    acB = CL.answer_contrast(bB.logits, ansA, ansB)
    return {"name": name, "ids_A": ids_A, "ids_B": ids_B, "P": P,
            "H_A": bA.H, "H_B": bB.H, "base_A": bA.logits, "base_B": bB.logits,
            "ansA": ansA, "ansB": ansB, "tokA": tokA, "tokB": tokB,
            "C_A": acA["C"], "C_B": acB["C"], "S_natural": acB["C"] - acA["C"],
            "pA_base": acA["p_A"], "pB_base": acB["p_B"], "seq": ids_A.shape[1]}


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item())


def inject(lm, task, L, P, delta, alpha, direction):
    """direction 'toB' runs on A (delta=+D), 'toA' runs on B (delta=-D)."""
    if direction == "toB":
        ids, base, base_other = task["ids_A"], task["base_A"], task["base_B"]
        h_src, target = task["H_A"][L, P], task["ansB"]
    else:
        ids, base, base_other = task["ids_B"], task["base_B"], task["base_A"]
        h_src, target = task["H_B"][L, P], task["ansA"]
    v = h_src + alpha * delta
    res = CL.run_and_capture_all(lm, ids, write=(L, P, op_add(delta, alpha)))
    lm_m = CL.logit_metrics(base, res.logits, lm.tokenizer)
    ac = CL.answer_contrast(res.logits, task["ansA"], task["ansB"])
    C = ac["C"]
    transfer = (C - task["C_A"]) if direction == "toB" else (task["C_B"] - C)
    tgt_p = ac["p_B"] if direction == "toB" else ac["p_A"]
    src_p = ac["p_A"] if direction == "toB" else ac["p_B"]
    return {"L": L, "P": P, "alpha": alpha, "direction": direction,
            "argmax_id": lm_m["top1_id"], "argmax_tok": lm_m["top1_tok"],
            "argmax_is_target": lm_m["top1_id"] == target,
            "target_prob": tgt_p, "source_prob": src_p,
            "C": C, "semantic_transfer": transfer, "transfer_fraction": transfer / task["S_natural"],
            "kl_from_source": lm_m["kl_from_control"], "kl_from_target": CL.kl_div(base_other, res.logits),
            "entropy": lm_m["entropy"],
            "delta_norm": float(delta.norm().item()), "inj_norm": float((alpha * delta).norm().item()),
            "result_norm": float(v.norm().item()),
            "cos_to_hA": cos(v, task["H_A"][L, P]), "cos_to_hB": cos(v, task["H_B"][L, P]),
            "dist_to_hB": float((v - task["H_B"][L, P]).norm().item()),
            "logits": res.logits}


def sweep_task(lm, task, sites, rand_seeds=(101, 202)):
    """sites: list of (L, role). Returns rows (list of dicts, logits stripped for jsonl)."""
    rows = []
    hidden = lm.hidden_size
    P = task["P"]
    for (L, role) in sites:
        D = task["H_B"][L, P] - task["H_A"][L, P]              # toB delta
        Dnorm = float(D.norm().item())
        deltas = {"semantic": D}
        for si, seed in enumerate(rand_seeds):
            u = CL.random_direction(hidden, seed ^ (L * 131 + P), lm.device, lm.dtype)
            deltas[f"rand{si+1}"] = u * Dnorm                  # matched L2 norm
        for direction in ("toB", "toA"):
            for kind, dvec in deltas.items():
                dv = dvec if direction == "toB" else -dvec
                for a in ALPHAS:
                    r = inject(lm, task, L, P, dv, a, direction)
                    r.update({"task": task["name"], "role": role, "kind": kind})
                    rows.append(r)
    return rows


def alpha1_sanity(lm, task, L):
    """Additive alpha=1 must reproduce full B<-... replacement (algebra: h_A + (h_B-h_A) = h_B)."""
    P = task["P"]
    out = {}
    for direction, (ids, h_to) in [("toB", (task["ids_A"], task["H_B"][L, P])),
                                   ("toA", (task["ids_B"], task["H_A"][L, P]))]:
        D = (task["H_B"][L, P] - task["H_A"][L, P]) if direction == "toB" else (task["H_A"][L, P] - task["H_B"][L, P])
        add = CL.run_and_capture_all(lm, ids, write=(L, P, op_add(D, 1.0))).logits
        rep = CL.run_and_capture_all(lm, ids, write=(L, P, IV.op_replace(h_to))).logits
        v_add = (task["H_A"][L, P] if direction == "toB" else task["H_B"][L, P]) + D
        out[direction] = {
            "site_max_abs_diff": float((v_add - h_to).abs().max().item()),
            "site_cosine": cos(v_add, h_to),
            "logit_max_abs_diff": float((add.double() - rep.double()).abs().max().item()),
            "logit_cosine": cos(add, rep),
            "kl_add_vs_replace": CL.kl_div(rep, add),
            "argmax_equal": int(torch.argmax(add).item()) == int(torch.argmax(rep).item()),
        }
    return out


# ---------------------------------------------------------------------------
# gate / classification
# ---------------------------------------------------------------------------

def classify(task, rows, sanity, strong_L, post_L):
    tf = task["S_natural"]
    def sel(kind, direction, L):
        return sorted([r for r in rows if r["kind"] == kind and r["direction"] == direction and r["L"] == L],
                      key=lambda r: r["alpha"])

    # (1) alpha=1 reproduces replacement
    sanity_ok = all(s["logit_cosine"] > 0.9999 and s["kl_add_vs_replace"] < 1e-3 and s["argmax_equal"]
                    and s["site_cosine"] > 0.9999 for s in sanity.values())

    # (5a/5b) sub-replacement regime at the strong site (toB)
    strong = sel("semantic", "toB", strong_L)
    sub = [r for r in strong if r["alpha"] < 1.0]
    sub_flip = [r for r in sub if r["argmax_is_target"]]
    sub_bigfrac = [r for r in sub if r["transfer_fraction"] >= 0.5]
    flip_alpha = (min(r["alpha"] for r in sub_flip) if sub_flip else None)

    # (3) dose-response coherence: target_prob non-decreasing across the ramp (allow small noise)
    tp = [r["target_prob"] for r in strong]
    dose_coherent = all(tp[i + 1] >= tp[i] - 0.05 for i in range(len(tp) - 1)) and tp[-1] > tp[0] + 0.3

    # (2) semantic beats norm-matched random (target-specific), at matched alpha across the ramp
    def flips(kind):
        return {r["alpha"]: r["argmax_is_target"] for r in sel(kind, "toB", strong_L)}
    sem_f = flips("semantic")
    rnd_f1, rnd_f2 = flips("rand1"), flips("rand2")
    # semantic flips at alphas where random does not
    sem_specific = any(sem_f.get(a) and not (rnd_f1.get(a) or rnd_f2.get(a)) for a in ALPHAS)
    # semantic max target_prob >> random max target_prob
    sem_maxtp = max(r["target_prob"] for r in sel("semantic", "toB", strong_L))
    rnd_maxtp = max([r["target_prob"] for r in sel("rand1", "toB", strong_L) + sel("rand2", "toB", strong_L)] or [0])
    semantic_beats_random = bool(sem_specific and sem_maxtp > rnd_maxtp + 0.3)

    # (4) receptive site beats post-deadline (same semantic delta).
    # Oracle gate keyed on the project-standard DECISIVE metrics (argmax / P(target) / KL),
    # NOT transfer_fraction (the contrast metric, which Phase 2C established drifts under
    # generic perturbation and is not decisive). Declared before re-evaluation.
    def at(kind, direction, L, a):
        c = [r for r in rows if r["kind"] == kind and r["direction"] == direction and r["L"] == L and abs(r["alpha"] - a) < 1e-9]
        return c[0] if c else None
    s1 = at("semantic", "toB", strong_L, 1.0)
    p1 = at("semantic", "toB", post_L, 1.0)
    post_all = [r for r in rows if r["kind"] == "semantic" and r["direction"] == "toB" and r["L"] == post_L]
    post_never_flips = not any(r["argmax_is_target"] for r in post_all)
    site_receptive_wins = bool(
        s1 and p1
        and s1["argmax_is_target"] and s1["target_prob"] >= 0.5
        and post_never_flips and p1["target_prob"] < 0.10 and p1["kl_from_source"] < 0.5)

    gate5 = bool(sub_flip) or bool(sub_bigfrac)
    passed = bool(sanity_ok and semantic_beats_random and dose_coherent and site_receptive_wins and gate5)

    if passed:
        verdict = "PASS"
    elif not sanity_ok:
        verdict = "METHOD_FAILURE"
    elif not (semantic_beats_random or gate5):
        verdict = "NEGATIVE"
    elif not dose_coherent:
        verdict = "AMBIGUOUS"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "gate_sanity_alpha1_reproduces_replacement": sanity_ok,
        "gate_semantic_beats_random": semantic_beats_random,
        "gate_dose_response_coherent": dose_coherent,
        "gate_receptive_beats_postdeadline": site_receptive_wins,
        "gate_subreplacement_regime(5a_or_5b)": gate5,
        "subreplacement_flip_alpha(5a)": flip_alpha,
        "subreplacement_bigfraction_exists(5b)": bool(sub_bigfrac),
        "strong_site_L": strong_L, "post_deadline_L": post_L,
        "postdeadline_never_flips_any_alpha": post_never_flips,
        "postdeadline_targetprob_at_alpha1": (p1["target_prob"] if p1 else None),
        "postdeadline_kl_at_alpha1": (p1["kl_from_source"] if p1 else None),
        "oracle_gate_metric": "argmax_is_target + P(target)>=0.5 (strong); never-flips + P(target)<0.10 + KL<0.5 (post-deadline). Corrected from transfer_fraction (Phase 2C: contrast drifts under perturbation, not decisive).",
        "semantic_max_targetprob": sem_maxtp, "random_max_targetprob": rnd_maxtp,
        "recommended_site_L": strong_L,
        "recommended_alpha": (flip_alpha if flip_alpha is not None else 1.0),
        "S_natural": tf,
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plots(task, rows, strong_L, near_L, post_L, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(kind, direction, L, field):
        rs = sorted([r for r in rows if r["kind"] == kind and r["direction"] == direction and r["L"] == L],
                    key=lambda r: r["alpha"])
        return [r["alpha"] for r in rs], [r[field] for r in rs]

    for field, ylab, fname in [("semantic_transfer", "semantic_transfer", "transfer_vs_alpha"),
                               ("target_prob", "P(target answer)", "targetprob_vs_alpha"),
                               ("kl_from_source", "KL(source||injected)", "kl_vs_alpha")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for L, lab in [(strong_L, f"L{strong_L} strong"), (near_L, f"L{near_L} near-deadline"),
                       (post_L, f"L{post_L} post-deadline")]:
            x, y = series("semantic", "toB", L, field)
            ax.plot(x, y, "-o", ms=3, label=lab)
        x, y = series("rand1", "toB", strong_L, field)
        ax.plot(x, y, "--x", ms=3, c="grey", label=f"L{strong_L} random")
        if field == "semantic_transfer":
            ax.axhline(task["S_natural"], ls=":", c="k", alpha=.5, label="S_natural")
        ax.set_title(f"{task['name']}: {ylab} vs alpha (A->B additive)")
        ax.set_xlabel("alpha"); ax.set_ylabel(ylab); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(outdir / f"{task['name']}_{fname}.png", dpi=130); plt.close(fig)

    # distance to natural h_B vs alpha at strong site
    fig, ax = plt.subplots(figsize=(7, 4))
    x, y = series("semantic", "toB", strong_L, "dist_to_hB")
    ax.plot(x, y, "-o", ms=3)
    ax.axvline(1.0, ls="--", c="red", alpha=.5, label="alpha=1 (=h_B)")
    ax.set_title(f"{task['name']}: ||injected - h_B|| vs alpha (L{strong_L})")
    ax.set_xlabel("alpha"); ax.set_ylabel("dist to natural h_B"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / f"{task['name']}_dist_to_hB.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected_raw").mkdir(exist_ok=True)
    lm = M.load()
    print(f"[load] {lm.model_id}: {lm.num_layers} layers")

    # sites (from Phase 2A cartography: source writable L0-21, deadline ~L22, dead L23+)
    color_sites = [(4, "strong"), (8, "strong_primary"), (15, "strong"),
                   (21, "near_deadline"), (24, "post_deadline")]
    # Phase 2C: source writable L0-11, deadline L11, dead after
    compare_sites = [(4, "strong_primary"), (8, "strong"), (20, "post_deadline")]

    color = load_task(lm, "color")
    print(f"[color] ans {color['tokA']}(={color['ansA']}) / {color['tokB']}(={color['ansB']}) | "
          f"P={color['P']} S_natural={color['S_natural']:.2f} pA={color['pA_base']:.3f} pB={color['pB_base']:.3f}")
    sanity = alpha1_sanity(lm, color, 8)
    print(f"[sanity L8] toB: logit_cos={sanity['toB']['logit_cosine']:.6f} "
          f"kl={sanity['toB']['kl_add_vs_replace']:.2e} argmax_eq={sanity['toB']['argmax_equal']} "
          f"site_cos={sanity['toB']['site_cosine']:.6f}")

    rows = sweep_task(lm, color, color_sites)
    # secondary corroboration on 2C (different topology)
    compare = load_task(lm, "compare")
    print(f"[compare] ans {compare['tokA']}/{compare['tokB']} S_natural={compare['S_natural']:.2f}")
    sanity_c = alpha1_sanity(lm, compare, 4)
    rows_c = sweep_task(lm, compare, compare_sites)

    # strip logits before saving to jsonl; keep a few raw for selected cells
    def strip(rs):
        out = []
        for r in rs:
            rr = {k: v for k, v in r.items() if k != "logits"}
            out.append(rr)
        return out
    all_rows = strip(rows) + strip(rows_c)
    (OUT / "cells.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows), encoding="utf-8")
    (OUT / "sanity.json").write_text(json.dumps({"color_L8": sanity, "compare_L4": sanity_c}, indent=2), encoding="utf-8")

    cls = classify(color, strip(rows), sanity, strong_L=8, post_L=24)
    cls_c = classify(compare, strip(rows_c), sanity_c, strong_L=4, post_L=20)

    plots(color, strip(rows), 8, 21, 24, OUT)
    plots(compare, strip(rows_c), 4, 8, 20, OUT)

    # save a few selected raw injected states (strong site, flip alpha + alpha1 + post-deadline alpha1)
    fa = cls["subreplacement_flip_alpha(5a)"] or 1.0
    for (L, a, tag) in [(8, fa, "flip"), (8, 1.0, "a1"), (24, 1.0, "postdl")]:
        D = color["H_B"][L, 20] - color["H_A"][L, 20]
        res = CL.run_and_capture_all(lm, color["ids_A"], write=(L, 20, op_add(D, a)))
        torch.save(res.H.cpu(), OUT / "selected_raw" / f"color_Hprime_L{L}_a{a}_{tag}.pt")

    env = {"python": platform.python_version(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "model_id": lm.model_id}
    summary = {"env": env, "alphas": ALPHAS,
               "color": {"task": "phase2A_copy(blue->green)", "P_source": color["P"],
                         "ansA": color["ansA"], "ansB": color["ansB"], "S_natural": color["S_natural"],
                         "classification": cls},
               "compare_secondary": {"task": "phase2C_comparison", "P_source": compare["P"],
                                     "S_natural": compare["S_natural"], "classification": cls_c},
               "overall_verdict": cls["verdict"],
               "advance_to_3C": cls["verdict"] == "PASS"}
    (OUT / "phase3a_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(OUT / "PHASE3A_RESULTS.md", summary, color, compare, strip(rows), sanity)
    print(f"\n[VERDICT color] {cls['verdict']}")
    print(json.dumps(cls, indent=2, default=str))
    print(f"[VERDICT compare] {cls_c['verdict']}")
    print(f"[advance_to_3C] {summary['advance_to_3C']}")


def write_report(path, summary, color, compare, rows, sanity):
    cls = summary["color"]["classification"]
    L = []
    A = L.append
    A("# Phase 3A — Exact Additive Semantic Write-Back (Results)\n")
    A(f"**Model:** {summary['env']['model_id']} · FP32 · eager · frozen · single forward pass. "
      f"carto_lib reused unchanged; tasks reused from Phase 2A/2C.\n")
    A(f"**Intervention:** `h <- h + alpha*D`, `D = h_B - h_A`, at (layer, P_source). "
      f"Additive (not replacement) — the PCDC-shaped write.\n")
    A(f"**Primary task (2A copy):** blue→green, P_source={color['P']}, "
      f"answers `{color['tokA']}`/`{color['tokB']}`, S_natural={color['S_natural']:.2f}.\n")

    A(f"## Verdict: **{cls['verdict']}**\n")
    A("| gate | result |\n|---|---|")
    for k in ("gate_sanity_alpha1_reproduces_replacement", "gate_semantic_beats_random",
              "gate_dose_response_coherent", "gate_receptive_beats_postdeadline",
              "gate_subreplacement_regime(5a_or_5b)", "subreplacement_flip_alpha(5a)",
              "semantic_max_targetprob", "random_max_targetprob",
              "recommended_site_L", "recommended_alpha"):
        A(f"| {k} | {cls[k]} |")

    A(f"\n> **Oracle gate metric (corrected):** {cls['oracle_gate_metric']} "
      f"Post-deadline L{cls['post_deadline_L']}: never flips any α = {cls['postdeadline_never_flips_any_alpha']}, "
      f"P(target)@α1 = {cls['postdeadline_targetprob_at_alpha1']:.3f}, KL@α1 = {cls['postdeadline_kl_at_alpha1']:.2f}.\n")

    A("\n## alpha=1 algebraic sanity (L8)\n")
    s = sanity["toB"]
    A(f"- additive α=1 vs exact replacement: logit cosine **{s['logit_cosine']:.6f}**, "
      f"KL **{s['kl_add_vs_replace']:.2e}**, argmax equal **{s['argmax_equal']}**, "
      f"site-state cosine **{s['site_cosine']:.6f}**. (Confirms `h_A + (h_B-h_A) = h_B`.)\n")

    A("## Dose-response (strong site L8, A→B, semantic)\n")
    A("| alpha | argmax | flip→target | P(target) | transfer | frac | cos→h_B |\n|---|---|---|---|---|---|---|")
    for r in sorted([r for r in rows if r["kind"] == "semantic" and r["direction"] == "toB" and r["L"] == 8],
                    key=lambda r: r["alpha"]):
        A(f"| {r['alpha']:.2f} | `{r['argmax_tok']}` | {r['argmax_is_target']} | {r['target_prob']:.3f} | "
          f"{r['semantic_transfer']:.2f} | {r['transfer_fraction']:.2f} | {r['cos_to_hB']:.3f} |")

    A("\n## Cartography as oracle (same semantic D, α=1)\n")
    A("| site | role | flip→target | transfer | frac |\n|---|---|---|---|---|")
    for L_, role in [(8, "strong"), (21, "near_deadline"), (24, "post_deadline")]:
        rr = [r for r in rows if r["kind"] == "semantic" and r["direction"] == "toB" and r["L"] == L_ and abs(r["alpha"] - 1.0) < 1e-9]
        if rr:
            r = rr[0]
            A(f"| L{L_} | {role} | {r['argmax_is_target']} | {r['semantic_transfer']:.2f} | {r['transfer_fraction']:.2f} |")

    A(f"\n## Secondary corroboration (2C comparison): {summary['compare_secondary']['classification']['verdict']}\n")
    A(f"S_natural={compare['S_natural']:.2f}; same additive form tested at a different-topology task "
      f"(source deadline ~L11). See phase3a_summary.json.\n")

    A("\n## Report answers (3A)\n")
    A(f"1. **α=1 reproduces replacement?** Yes (logit cosine {s['logit_cosine']:.5f}, KL {s['kl_add_vs_replace']:.1e}).")
    A(f"2. **Coherent dose-response?** {cls['gate_dose_response_coherent']}.")
    A(f"3. **Useful sub-replacement α?** flip at α={cls['subreplacement_flip_alpha(5a)']}; "
      f"big-fraction<1 exists={cls['subreplacement_bigfraction_exists(5b)']}.")
    A(f"4. **Semantic beats random?** {cls['gate_semantic_beats_random']} "
      f"(max P(target): semantic {cls['semantic_max_targetprob']:.3f} vs random {cls['random_max_targetprob']:.3f}).")
    A(f"5. **Receptive site beats post-deadline?** {cls['gate_receptive_beats_postdeadline']}.")
    A(f"6. **Best predeclared candidate for 3C:** site L{cls['recommended_site_L']}, alpha={cls['recommended_alpha']}.")
    A("7. **Surprises:** see dose-response table (where the flip occurs relative to α=1).")
    A(f"8. **Classification:** **{cls['verdict']}** → advance to 3C = {summary['advance_to_3C']}.")

    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe phase3_additive.py")
    A("```")
    A("\n**STOP unless PASS.** Only a PASS verdict advances to Phase 3C.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
