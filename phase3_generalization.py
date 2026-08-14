"""
phase3_generalization.py -- Phase 3C: held-out generalization of a CONSTRUCTED direction.

Only run after Phase 3A PASS. Tests whether the useful additive signal is
prompt-specific activation surgery or a reusable semantic DIRECTION.

Construct, from a TRAIN set of matched blue->green pairs (varying context):
    D_mean = mean_i (h_B_i - h_A_i)   at the 3A-selected site (layer 8, per-pair color token)
and a direction-normalized variant (mean direction scaled to typical magnitude).
Inject D_mean into HELD-OUT prompts never used to build it. NO training, NO probes.

Injection policy is FIXED from 3A: layer L=8, additive `h + alpha*D_mean`, primary
alpha=0.5 (fallback/reference alpha=1.0). No per-example tuning.

Controls: no-injection; unrelated single-pair D; norm-matched random; oracle exact
held-out D (ceiling). Plus specificity (unrelated prompts) and site-portability
(L8 vs post-deadline L24). carto_lib reused unchanged. Writes only results/phase3c/.

Run:  python phase3_generalization.py
"""

from __future__ import annotations

import json
import platform
import random as pyrandom
from pathlib import Path

import numpy as np
import torch

import model as M
import intervention as IV
import carto_lib as CL
from phase3_additive import op_add, cos, SYSTEM

OUT = Path(__file__).parent / "results" / "phase3c"
INJECT_L = 8            # 3A recommended strong site
POST_L = 24            # 3A post-deadline site (portability control)
ALPHAS = [0.5, 1.0]    # 0.5 = 3A primary; 1.0 = reference
SPLIT_SEED = 7
RAND_SEEDS = (2024, 4048)

ENTITIES = ["dax", "wug", "tom", "sam", "cat", "dog", "car", "cup", "pen", "hat",
            "key", "book", "star", "tree", "fish", "lamp", "ring", "coin", "box", "leaf"]

# Stage-2 cross-phrasing templates (validated at runtime)
TEMPLATES = {
    "fact": lambda e, c: f"Fact: the {e} is {c}. Question: what color is the {e}? Reply with only one lowercase word.",
    "has": lambda e, c: f"The {e} has the color {c}. What color is the {e}? Answer in one word.",
    "colon": lambda e, c: f"Color of the {e}: {c}. What color is the {e}? Answer in one word.",
    "painted": lambda e, c: f"Someone painted the {e} {c}. What color is the {e} now? Answer in one word.",
}


def cap(lm, ids):
    return CL.run_and_capture_all(lm, ids)


def make_pair(lm, tmpl, e, blue="blue", green="green"):
    """Build+validate one A/B pair. Returns dict or None."""
    tok = lm.tokenizer
    ids_A = M.build_inputs(lm, tmpl(e, blue), SYSTEM)
    ids_B = M.build_inputs(lm, tmpl(e, green), SYSTEM)
    if ids_A.shape[1] != ids_B.shape[1]:
        return None
    diff = (ids_A[0] != ids_B[0]).nonzero(as_tuple=True)[0].tolist()
    if len(diff) != 1:
        return None
    P = diff[0]
    if tok.decode([int(ids_A[0][P])]).strip() != blue or tok.decode([int(ids_B[0][P])]).strip() != green:
        return None
    bA, bB = cap(lm, ids_A), cap(lm, ids_B)
    ansA, ansB = int(torch.argmax(bA.logits).item()), int(torch.argmax(bB.logits).item())
    # case-insensitive: some phrasings elicit "Blue"/"Green"; answer token is stored per-pair
    if tok.decode([ansA]).strip().lower() != blue or tok.decode([ansB]).strip().lower() != green:
        return None
    pA = float(torch.softmax(bA.logits.double(), -1)[ansA].item())
    pB = float(torch.softmax(bB.logits.double(), -1)[ansB].item())
    if min(pA, pB) < 0.95:
        return None
    D = (bB.H[INJECT_L, P] - bA.H[INJECT_L, P]).clone()
    return {"entity": e, "P": P, "ids_A": ids_A, "ids_B": ids_B,
            "H_A": bA.H, "H_B": bB.H, "base_A": bA.logits,
            "ansA": ansA, "ansB": ansB, "pA": pA, "pB": pB,
            "C_A": CL.answer_contrast(bA.logits, ansA, ansB)["C"],
            "C_B": CL.answer_contrast(bB.logits, ansA, ansB)["C"],
            "D": D, "Dnorm": float(D.norm().item())}


def evaluate(lm, pair, delta, alpha):
    """Inject `delta` additively at (INJECT_L or given L, pair P) on pair.ids_A.
    Uses this pair's OWN answer tokens (blue/green variant), so cross-phrasing
    pairs with capitalized answers are scored correctly."""
    L = pair.get("_inject_L", INJECT_L)
    P = pair["P"]
    BLUE, GREEN = pair["ansA"], pair["ansB"]
    res = CL.run_and_capture_all(lm, pair["ids_A"], write=(L, P, op_add(delta, alpha)))
    lm_m = CL.logit_metrics(pair["base_A"], res.logits, lm.tokenizer)
    ac = CL.answer_contrast(res.logits, BLUE, GREEN)
    C = ac["C"]
    S = pair["C_B"] - pair["C_A"]
    return {"argmax_tok": lm_m["top1_tok"], "argmax_is_target": lm_m["top1_id"] == GREEN,
            "p_green": ac["p_B"], "p_blue": ac["p_A"], "kl": lm_m["kl_from_control"],
            "entropy": lm_m["entropy"], "transfer": C - pair["C_A"], "transfer_fraction": (C - pair["C_A"]) / S,
            "cos_to_exact_D": cos(delta, pair["D"]), "norm_ratio": float(delta.norm().item()) / (pair["Dnorm"] + 1e-9)}


def build_dmean(train, device, dtype):
    Ds = torch.stack([p["D"] for p in train])          # [n,1536]
    raw = Ds.mean(0)
    mean_indiv_norm = float(Ds.norm(dim=-1).mean().item())
    dirn = raw / raw.norm() * mean_indiv_norm            # direction of mean, typical magnitude
    return raw.to(device, dtype), dirn.to(device, dtype), mean_indiv_norm, float(raw.norm().item())


def run_stage(lm, pairs, label):
    """Split, build D_mean, evaluate held-out under all conditions + controls.
    Each pair is scored with its own answer tokens."""
    keys = sorted(range(len(pairs)), key=lambda i: pairs[i]["entity"] if "entity" in pairs[i] else str(i))
    rng = pyrandom.Random(SPLIT_SEED); rng.shuffle(keys)
    n_test = max(2, round(0.3 * len(pairs)))
    test_idx = set(keys[:n_test]); train = [pairs[i] for i in keys[n_test:]]; test = [pairs[i] for i in keys[:n_test]]
    raw, dirn, mni, rawnorm = build_dmean(train, lm.device, lm.dtype)
    # unrelated single-pair D (a fixed train example)
    single = train[0]["D"].to(lm.device, lm.dtype)
    rows = []
    for pair in test:
        exact = pair["D"].to(lm.device, lm.dtype)
        conds = {"D_mean_dir": dirn, "D_mean_raw": raw, "unrelated_single_D": single,
                 "oracle_exact_D": exact}
        for si, seed in enumerate(RAND_SEEDS):
            u = CL.random_direction(lm.hidden_size, seed, lm.device, lm.dtype)
            conds[f"random{si+1}"] = u * float(dirn.norm().item())
        # baseline (alpha 0 on any delta = no-op)
        base = evaluate(lm, pair, dirn, 0.0)
        base.update({"entity": pair.get("entity", "?"), "cond": "baseline", "alpha": 0.0})
        rows.append(base)
        for cname, dvec in conds.items():
            for a in ALPHAS:
                r = evaluate(lm, pair, dvec, a)
                r.update({"entity": pair.get("entity", "?"), "cond": cname, "alpha": a})
                rows.append(r)
    meta = {"label": label, "n_pairs": len(pairs), "n_train": len(train), "n_test": len(test),
            "train_entities": [p.get("entity", "?") for p in train],
            "test_entities": [p.get("entity", "?") for p in test],
            "D_mean_raw_norm": rawnorm, "mean_individual_norm": mni,
            "cos_dir_vs_raw": cos(dirn.cpu(), raw.cpu())}
    return rows, meta, (raw, dirn), test


def aggregate(rows):
    out = {}
    conds = sorted(set(r["cond"] for r in rows))
    for c in conds:
        for a in sorted(set(r["alpha"] for r in rows if r["cond"] == c)):
            rs = [r for r in rows if r["cond"] == c and abs(r["alpha"] - a) < 1e-9]
            if not rs:
                continue
            out[f"{c}@{a}"] = {
                "flip_rate": float(np.mean([r["argmax_is_target"] for r in rs])),
                "mean_p_green": float(np.mean([r["p_green"] for r in rs])),
                "median_transfer_fraction": float(np.median([r["transfer_fraction"] for r in rs])),
                "mean_cos_to_exact_D": float(np.mean([r["cos_to_exact_D"] for r in rs])),
                "mean_kl": float(np.mean([r["kl"] for r in rs])), "n": len(rs)}
    return out


def classify_stage(agg, alpha):
    def get(c):
        return agg.get(f"{c}@{alpha}", {"flip_rate": 0, "mean_p_green": 0, "median_transfer_fraction": 0})
    dm = get("D_mean_dir")
    rnd = max(get("random1")["flip_rate"], get("random2")["flip_rate"])
    unr = get("unrelated_single_D")["flip_rate"]
    orc = get("oracle_exact_D")
    strong = dm["flip_rate"] >= 0.5 and dm["flip_rate"] > rnd + 0.25 and dm["flip_rate"] >= unr
    partial = (not strong) and (dm["mean_p_green"] > 0.2 and dm["median_transfer_fraction"] >= 0.3
                                and dm["flip_rate"] > rnd)
    null = dm["flip_rate"] <= rnd + 0.05 and dm["mean_p_green"] < 0.15
    if strong:
        cls = "STRONG_POSITIVE"
    elif partial:
        cls = "PARTIAL_POSITIVE"
    elif null:
        cls = "NULL"
    else:
        cls = "AMBIGUOUS"
    return {"classification": cls, "alpha": alpha, "D_mean_dir_flip_rate": dm["flip_rate"],
            "D_mean_dir_mean_p_green": dm["mean_p_green"],
            "D_mean_dir_median_transfer_fraction": dm["median_transfer_fraction"],
            "random_flip_rate": rnd, "unrelated_single_flip_rate": unr,
            "oracle_flip_rate": orc["flip_rate"], "mean_cos_dmean_to_exact": dm["mean_cos_to_exact_D"]}


def specificity(lm, dirn, BLUE, GREEN):
    """Inject the green-direction into UNRELATED prompts; it should not green-ify them."""
    tok = lm.tokenizer
    probes = {
        "arithmetic": "Is 3 plus 4 equal to 7? Answer in one word.",
        "capital": "What is the capital of France? Reply with only one lowercase word.",
        "comparison": "Is 5 greater than 4? Answer in one word.",
        "unrelated_color": "Fact: the dax is red. Question: what color is the dax? Reply with only one lowercase word.",
    }
    out = {}
    for name, text in probes.items():
        ids = M.build_inputs(lm, text, SYSTEM)
        base = M.forward_logits(lm, ids)[0]
        P = ids.shape[1] // 2
        for a in ALPHAS:
            res = CL.run_and_capture_all(lm, ids, write=(INJECT_L, P, op_add(dirn, a)))
            lm_m = CL.logit_metrics(base, res.logits, tok)
            pg = float(torch.softmax(res.logits.double(), -1)[GREEN].item())
            pg0 = float(torch.softmax(base.double(), -1)[GREEN].item())
            out[f"{name}@{a}"] = {"base_argmax": tok.decode([int(base.argmax())]).strip(),
                                  "inj_argmax": lm_m["top1_tok"], "argmax_changed": lm_m["top1_changed"],
                                  "p_green_base": pg0, "p_green_inj": pg, "kl": lm_m["kl_from_control"]}
    return out


def portability(lm, dirn, test, alpha):
    """Same D_mean at strong L8 vs post-deadline L24 on held-out pairs."""
    res = {}
    for L in (INJECT_L, POST_L):
        flips = []
        for pair in test:
            pair["_inject_L"] = L
            r = evaluate(lm, pair, dirn, alpha)
            flips.append(r["argmax_is_target"])
            pair.pop("_inject_L", None)
        res[f"L{L}"] = {"flip_rate": float(np.mean(flips))}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lm = M.load()
    tok = lm.tokenizer
    print(f"[load] {lm.model_id}")

    # ---- Stage 1: within-template (vary entity) ----
    fact = TEMPLATES["fact"]
    pairs = []
    for e in ENTITIES:
        p = make_pair(lm, fact, e)
        if p is not None:
            pairs.append(p)
    print(f"[stage1] validated {len(pairs)}/{len(ENTITIES)} entity pairs")
    if len(pairs) < 6:
        (OUT / "phase3c_summary.json").write_text(json.dumps(
            {"status": "HALT_INSUFFICIENT_PAIRS", "n_valid": len(pairs)}, indent=2), encoding="utf-8")
        print("[HALT] insufficient clean pairs"); return
    BLUE, GREEN = pairs[0]["ansA"], pairs[0]["ansB"]
    assert all(p["ansA"] == BLUE and p["ansB"] == GREEN for p in pairs), "answer tokens inconsistent"

    rows1, meta1, (raw1, dir1), test1 = run_stage(lm, pairs, "stage1_entities")
    agg1 = aggregate(rows1)
    cls1 = {a: classify_stage(agg1, a) for a in ALPHAS}
    spec = specificity(lm, dir1, BLUE, GREEN)
    port = {a: portability(lm, dir1, test1, a) for a in ALPHAS}
    print(f"[stage1] D_mean_dir flip_rate @0.5={cls1[0.5]['D_mean_dir_flip_rate']:.2f} "
          f"@1.0={cls1[1.0]['D_mean_dir_flip_rate']:.2f} | random={cls1[0.5]['random_flip_rate']:.2f} "
          f"oracle@1.0={cls1[1.0]['oracle_flip_rate']:.2f}")

    stage1_strong = any(cls1[a]["classification"] == "STRONG_POSITIVE" for a in ALPHAS)

    # ---- Stage 2: cross-phrasing (only if Stage 1 strong positive) ----
    rows2 = meta2 = agg2 = cls2 = None
    if stage1_strong:
        print("[stage2] Stage 1 STRONG -> running cross-phrasing")
        tpairs = []
        for tname, tf in TEMPLATES.items():
            for e in ENTITIES[:8]:
                p = make_pair(lm, tf, e)
                if p is not None:
                    p["template"] = tname
                    p["entity"] = f"{tname}:{e}"
                    tpairs.append(p)
        # split by TEMPLATE (hold out whole phrasings). >=2 templates suffices for a
        # minimal cross-phrasing split; the p>=0.95 quality bar is NOT relaxed to add N.
        if len(set(p["template"] for p in tpairs)) >= 2:
            tmpls = sorted(set(p["template"] for p in tpairs))
            rng = pyrandom.Random(SPLIT_SEED); rng.shuffle(tmpls)
            held = set(tmpls[:1]); train2 = [p for p in tpairs if p["template"] not in held]
            test2 = [p for p in tpairs if p["template"] in held]
            if train2 and test2:
                raw2, dir2, mni2, rn2 = build_dmean(train2, lm.device, lm.dtype)
                single2 = train2[0]["D"].to(lm.device, lm.dtype)
                rows2 = []
                for pair in test2:
                    exact = pair["D"].to(lm.device, lm.dtype)
                    conds = {"D_mean_dir": dir2, "unrelated_single_D": single2, "oracle_exact_D": exact}
                    u = CL.random_direction(lm.hidden_size, RAND_SEEDS[0], lm.device, lm.dtype)
                    conds["random1"] = u * float(dir2.norm().item())
                    b = evaluate(lm, pair, dir2, 0.0); b.update({"entity": pair["entity"], "cond": "baseline", "alpha": 0.0}); rows2.append(b)
                    for cname, dvec in conds.items():
                        for a in ALPHAS:
                            r = evaluate(lm, pair, dvec, a)
                            r.update({"entity": pair["entity"], "cond": cname, "alpha": a}); rows2.append(r)
                meta2 = {"held_out_templates": sorted(held), "train_templates": sorted(set(p["template"] for p in train2)),
                         "n_train": len(train2), "n_test": len(test2)}
                agg2 = aggregate(rows2); cls2 = {a: classify_stage(agg2, a) for a in ALPHAS}
                print(f"[stage2] held-out template(s) {sorted(held)}: D_mean_dir flip @0.5={cls2[0.5]['D_mean_dir_flip_rate']:.2f} @1.0={cls2[1.0]['D_mean_dir_flip_rate']:.2f}")
    else:
        print("[stage2] skipped (Stage 1 not STRONG_POSITIVE)")

    # ---- persist ----
    torch.save({"D_mean_raw": raw1.cpu(), "D_mean_dir": dir1.cpu()}, OUT / "D_mean_stage1.pt")
    (OUT / "stage1_cells.jsonl").write_text("\n".join(json.dumps({k: v for k, v in r.items()}) for r in rows1), encoding="utf-8")
    if rows2:
        (OUT / "stage2_cells.jsonl").write_text("\n".join(json.dumps(r) for r in rows2), encoding="utf-8")

    plots(agg1, cls1, spec, port, OUT)

    summary = {
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0), "model_id": lm.model_id},
        "policy": {"inject_layer": INJECT_L, "post_deadline_layer": POST_L, "alphas": ALPHAS,
                   "primary_alpha": 0.5, "split_seed": SPLIT_SEED},
        "stage1": {"meta": meta1, "aggregate": agg1, "classification": cls1,
                   "specificity": spec, "portability": port},
        "stage2": ({"meta": meta2, "aggregate": agg2, "classification": cls2} if agg2 else "skipped_or_unavailable"),
        "stage1_within_template": (cls1[0.5]["classification"], cls1[1.0]["classification"]),
        "stage2_cross_template": ((cls2[0.5]["classification"], cls2[1.0]["classification"]) if cls2 else "N/A"),
    }
    (OUT / "phase3c_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(OUT / "PHASE3C_GENERALIZATION_RESULTS.md", summary)
    print(f"[done] stage1={summary['stage1_within_template']} stage2={summary['stage2_cross_template']}")


def plots(agg, cls, spec, port, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = 0.5
    conds = ["baseline", "random1", "unrelated_single_D", "D_mean_raw", "D_mean_dir", "oracle_exact_D"]
    fr = [agg.get(f"{c}@{a}", agg.get(f"{c}@0.0", {})).get("flip_rate", 0) for c in conds]
    pg = [agg.get(f"{c}@{a}", agg.get(f"{c}@0.0", {})).get("mean_p_green", 0) for c in conds]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(conds))
    ax.bar([i - 0.2 for i in x], fr, 0.4, label="held-out flip rate")
    ax.bar([i + 0.2 for i in x], pg, 0.4, label="mean P(green)")
    ax.set_xticks(list(x)); ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"Phase 3C Stage-1 held-out generalization (alpha={a})"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(outdir / "stage1_conditions.png", dpi=130); plt.close(fig)

    # portability
    fig, ax = plt.subplots(figsize=(5, 4))
    for a_ in ALPHAS:
        ax.plot(["L8 strong", "L24 post-dl"], [port[a_]["L8"]["flip_rate"], port[a_]["L24"]["flip_rate"]],
                "-o", label=f"alpha={a_}")
    ax.set_title("D_mean portability: strong vs post-deadline site"); ax.set_ylabel("held-out flip rate")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "portability.png", dpi=130); plt.close(fig)


def write_report(path, s):
    L = []
    A = L.append
    A("# Phase 3C — Held-Out Generalization of a Constructed Direction (Results)\n")
    A(f"**Model:** {s['env']['model_id']} · FP32 · frozen · single forward pass. carto_lib reused unchanged.\n")
    A(f"**Policy (fixed from 3A):** inject at layer **L{s['policy']['inject_layer']}**, additive "
      f"`h + alpha*D_mean`, primary **alpha=0.5**, reference alpha=1.0. No per-example tuning.\n")
    A(f"**D_mean** built from TRAIN pairs' `h_green - h_blue` at L{s['policy']['inject_layer']}; "
      f"injected into HELD-OUT prompts. No training, no probes.\n")

    m1 = s["stage1"]["meta"]
    A(f"## Stage 1 — within-template, held-out ENTITIES\n")
    A(f"- {m1['n_pairs']} validated blue→green pairs; train {m1['n_train']} / held-out test {m1['n_test']}.")
    A(f"- train entities: {m1['train_entities']}; **held-out**: {m1['test_entities']}.")
    A(f"- D_mean cos(dir vs raw)={m1['cos_dir_vs_raw']:.3f}, mean individual ‖D‖={m1['mean_individual_norm']:.1f}, raw mean ‖D‖={m1['D_mean_raw_norm']:.1f}.\n")
    A("| classification | primary α=0.5 | reference α=1.0 |")
    A("|---|---|---|")
    A(f"| **within-template** | **{s['stage1']['classification'][0.5]['classification']}** | {s['stage1']['classification'][1.0]['classification']} |")
    A("\n**Held-out flip rate by condition** (α=0.5 unless α=1.0 shown):\n")
    A("| condition | flip rate | mean P(green) | median transfer frac | mean cos→exact D |")
    A("|---|---|---|---|---|")
    for c in ["baseline@0.0", "random1@0.5", "unrelated_single_D@0.5", "D_mean_raw@0.5",
              "D_mean_dir@0.5", "D_mean_dir@1.0", "oracle_exact_D@0.5", "oracle_exact_D@1.0"]:
        v = s["stage1"]["aggregate"].get(c)
        if v:
            A(f"| {c} | {v['flip_rate']:.2f} | {v['mean_p_green']:.3f} | {v['median_transfer_fraction']:.2f} | {v['mean_cos_to_exact_D']:.3f} |")

    A("\n## Specificity (green-direction injected into UNRELATED prompts)\n")
    A("| probe@α | base argmax | injected argmax | changed | P(green) base→inj | KL |")
    A("|---|---|---|---|---|---|")
    for k, v in s["stage1"]["specificity"].items():
        A(f"| {k} | {v['base_argmax']} | {v['inj_argmax']} | {v['argmax_changed']} | {v['p_green_base']:.3f}→{v['p_green_inj']:.3f} | {v['kl']:.2f} |")

    A("\n## Site portability (same D_mean, strong L8 vs post-deadline L24)\n")
    A("| alpha | L8 flip rate | L24 flip rate |\n|---|---|---|")
    for a_ in s["policy"]["alphas"]:
        p = s["stage1"]["portability"][a_]
        A(f"| {a_} | {p['L8']['flip_rate']:.2f} | {p['L24']['flip_rate']:.2f} |")

    if isinstance(s["stage2"], dict):
        A("\n## Stage 2 — cross-phrasing, held-out TEMPLATES\n")
        A(f"- held-out templates: {s['stage2']['meta']['held_out_templates']}; train templates: {s['stage2']['meta']['train_templates']}.")
        A(f"- **cross-template classification:** α=0.5 **{s['stage2']['classification'][0.5]['classification']}**, "
          f"α=1.0 {s['stage2']['classification'][1.0]['classification']}.")
        for c in ["D_mean_dir@0.5", "D_mean_dir@1.0", "random1@0.5", "oracle_exact_D@1.0"]:
            v = s["stage2"]["aggregate"].get(c)
            if v:
                A(f"  - {c}: flip {v['flip_rate']:.2f}, P(green) {v['mean_p_green']:.3f}, cos→exact {v['mean_cos_to_exact_D']:.3f}")
    else:
        A("\n## Stage 2 — cross-phrasing: **skipped** (Stage 1 not STRONG_POSITIVE).\n")

    A("\n## Answers (3C)\n")
    c05 = s["stage1"]["classification"][0.5]
    A(f"9. **Generalizes to held-out entities?** within-template = {c05['classification']} "
      f"(D_mean flip {c05['D_mean_dir_flip_rate']:.2f} vs random {c05['random_flip_rate']:.2f}).")
    A(f"10. **Generalizes to held-out phrasings?** cross-template = {s['stage2_cross_template']}.")
    A(f"11. **Beats random?** flip {c05['D_mean_dir_flip_rate']:.2f} vs {c05['random_flip_rate']:.2f}.")
    A(f"12. **Beats unrelated single delta?** {c05['D_mean_dir_flip_rate']:.2f} vs {c05['unrelated_single_flip_rate']:.2f}.")
    A(f"13. **How close to oracle?** oracle flip {c05['oracle_flip_rate']:.2f}; mean cos(D_mean,exact)={c05['mean_cos_dmean_to_exact']:.3f}.")
    A("14. **Target-specific?** see specificity table (P(green) movement on unrelated prompts).")
    A("15. **Leaves unrelated intact?** see specificity argmax-changed / KL.")
    A("16. **Cartography predicts site?** see portability (L8 vs L24 flip rate).")
    A(f"17. **Reusable object best described as:** {'context-general (within task)' if c05['classification'].endswith('POSITIVE') else 'prompt/template-specific'} — see stage1/stage2.")
    A("18. **Implication for a memory→latent translator:** a fixed averaged direction "
      + ("DOES steer unseen contexts → a reusable latent write primitive is plausible." if c05['classification'] == 'STRONG_POSITIVE'
         else "does not cleanly steer unseen contexts by simple averaging → a learned map is likely needed."))
    A("19. **Unsupported:** anything beyond the tested task/scale; compression/probe/learned routes (not run).")
    A("20. **Phase 4 (for discussion):** " +
      ("compression/low-rank of D_mean, then a learned memory→direction map." if c05['classification'] == 'STRONG_POSITIVE'
       else "why averaging fails (direction variance across contexts) and whether a learned direction recovers it."))

    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe phase3_generalization.py")
    A("```")
    A("\n**STRICT STOP after Phase 3C.** No compression/PCA/probe/learned map/PCDC — later decisions.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
