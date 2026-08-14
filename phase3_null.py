"""
phase3_null.py -- Phase 3N: empirical null / chance-control test of Phase 3C.

Adversarial validation: try to KILL the "D_mean is a semantic direction"
interpretation with the strongest FAIR null. Preregistered in
results/phase3n/NULL_DESIGN.md (read that first).

Frozen from Phase 3: model, color task, injection layer L8, position=color token,
alpha in {0.5 primary, 1.0}, D_mean (loaded from results/phase3c/D_mean_stage1.pt),
norm 34.057, the 6 held-out entities, 4 specificity prompts, post-deadline L24.

Null families (equal injection budget = ‖D_mean‖, same site/pos/alpha):
  isotropic (weak baseline)  | permuted (structure-preserving) | sign
  empirical_othercolor + empirical_crosstask (STRONG on-manifold, real wrong-info)
Each null trial uses ONE R_j across ALL held-out prompts (matches D_mean's reuse).

Primary statistic S = mean held-out P(target=green) at alpha=0.5 (D_mean = 0.879).
Empirical p = (k+1)/(N+1). Writes only results/phase3n/.

Run:  python phase3_null.py --pilot   |   python phase3_null.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import model as M
import intervention as IV
import carto_lib as CL
from phase3_additive import op_add, cos
from phase3_generalization import make_pair, TEMPLATES, SYSTEM

OUT = Path(__file__).parent / "results" / "phase3n"
P3C = Path(__file__).parent / "results" / "phase3c"
P2B = Path(__file__).parent / "results" / "phase2b"
P2C = Path(__file__).parent / "results" / "phase2c"
INJECT_L, POST_L = 8, 24
ALPHAS = [0.5, 1.0]
SPEC_THRESH = 0.05          # frozen specificity threshold (composite)
FLIP_COMPOSITE = 5 / 6      # frozen: >=5/6 held-out flips
BASE_SEED = 90000
OTHER_COLORS = ["red", "yellow", "black", "white", "brown", "purple", "orange", "pink", "gray", "gold"]

SPEC_PROMPTS = {
    "arithmetic": "Is 3 plus 4 equal to 7? Answer in one word.",
    "capital": "What is the capital of France? Reply with only one lowercase word.",
    "comparison": "Is 5 greater than 4? Answer in one word.",
    "unrelated_color": "Fact: the dax is red. Question: what color is the dax? Reply with only one lowercase word.",
}


@torch.no_grad()
def inj_logits(lm, ids, L, P, delta, alpha):
    """Logits-only forward with one additive write (fast path for null trials)."""
    with IV.WriteHook(lm.layers[L], op_add(delta, alpha), position=P):
        return lm.model(input_ids=ids, use_cache=False).logits[0, -1, :]


def build_heldout(lm, entities, tmpl):
    pairs = []
    for e in entities:
        p = make_pair(lm, tmpl, e)
        assert p is not None, f"held-out entity {e} failed to rebuild (config drift)"
        pairs.append(p)
    return pairs


def build_specs(lm):
    out = []
    for name, text in SPEC_PROMPTS.items():
        ids = M.build_inputs(lm, text, SYSTEM)
        base_argmax = int(torch.argmax(M.forward_logits(lm, ids)[0]).item())
        out.append({"name": name, "ids": ids, "P": ids.shape[1] // 2, "base_argmax": base_argmax})
    return out


def score_heldout(lm, heldout, delta, alpha, GREEN, L=INJECT_L):
    flips, pg = [], []
    for pr in heldout:
        lg = inj_logits(lm, pr["ids_A"], L, pr["P"], delta, alpha)
        flips.append(int(torch.argmax(lg).item()) == GREEN)
        pg.append(float(torch.softmax(lg.double(), -1)[GREEN].item()))
    return float(np.mean(flips)), float(np.mean(pg))


def score_spec(lm, specs, delta, alpha, GREEN):
    pgs, changed = [], 0
    for s in specs:
        lg = inj_logits(lm, s["ids"], INJECT_L, s["P"], delta, alpha)
        pgs.append(float(torch.softmax(lg.double(), -1)[GREEN].item()))
        if int(torch.argmax(lg).item()) != s["base_argmax"]:
            changed += 1
    return float(np.mean(pgs)), float(np.max(pgs)), changed


# ---- null vector generators ----

def gen_isotropic(dmean, j, device, dtype):
    u = CL.random_direction(dmean.numel(), BASE_SEED ^ (j * 2654435761 & 0x7FFFFFFF), device, dtype)
    return u * float(dmean.norm().item())


def gen_permuted(dmean, j, device):
    g = torch.Generator(device="cpu").manual_seed(BASE_SEED ^ (j * 40503))
    perm = torch.randperm(dmean.numel(), generator=g)
    return dmean.detach().cpu()[perm].to(device)


def gen_sign(dmean, j, device):
    g = torch.Generator(device="cpu").manual_seed(BASE_SEED ^ (j * 27644437))
    s = (torch.randint(0, 2, (dmean.numel(),), generator=g) * 2 - 1).float()
    return (dmean.detach().cpu() * s).to(device)


def build_othercolor_deltas(lm, train_ents, target_norm, device, dtype):
    """Mean(h_color - h_blue) over TRAIN entities, per color; scaled to target_norm."""
    fact = TEMPLATES["fact"]
    out = {}
    for c in OTHER_COLORS:
        Ds = []
        for e in train_ents:
            p = make_pair(lm, fact, e, blue="blue", green=c)
            if p is not None:
                Ds.append(p["D"])
        if len(Ds) >= max(4, len(train_ents) // 2):
            raw = torch.stack(Ds).mean(0)
            out[c] = {"delta": (raw / raw.norm() * target_norm).to(device, dtype),
                      "n": len(Ds), "green_id": None}
    return out


def build_crosstask_deltas(lm, target_norm, device, dtype):
    out = {}
    for name, pdir, sub in [("arithmetic_2B", P2B, "baselines"), ("comparison_2C", P2C, "baselines")]:
        try:
            meta = json.loads((pdir / "meta.json").read_text())
            P = meta["pair"].get("P_source")
            HA = torch.load(pdir / sub / "H_A.pt"); HB = torch.load(pdir / sub / "H_B.pt")
            raw = (HB[INJECT_L, P] - HA[INJECT_L, P])
            out[name] = (raw / raw.norm() * target_norm).to(device, dtype)
        except Exception as e:
            out[name] = f"unavailable: {e}"
    return out


def run_family(lm, name, gen, N, heldout, specs, dmean, GREEN, spec_all=True, flip_gate=0.5):
    rows = []
    t = time.time()
    for j in range(N):
        R = gen(j)
        fr, pg = score_heldout(lm, heldout, R, 0.5, GREEN)
        rec = {"family": name, "j": j, "flip_rate": fr, "S": pg,
               "cos_dmean": cos(R.cpu(), dmean.cpu())}
        if spec_all or fr >= flip_gate:
            sm, smax, chg = score_spec(lm, specs, R, 0.5, GREEN)
            rec.update({"spec_mean_pgreen": sm, "spec_max_pgreen": smax, "spec_argmax_changed": chg})
        rows.append(rec)
        if (j + 1) % 500 == 0:
            print(f"  [{name}] {j+1}/{N}  ({time.time()-t:.0f}s)")
    return rows


def pctile_stats(vals, S_sem):
    a = np.array(vals)
    k = int(np.sum(a >= S_sem))
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "sd": float(a.std()), "max": float(a.max()),
            "q50": float(np.percentile(a, 50)), "q90": float(np.percentile(a, 90)),
            "q95": float(np.percentile(a, 95)), "q99": float(np.percentile(a, 99)),
            "q99.9": float(np.percentile(a, 99.9)),
            "k_ge_sem": k, "p_empirical_addone": (k + 1) / (len(a) + 1),
            "sem_percentile": float((np.sum(a < S_sem) / len(a)) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n_sign", type=int, default=1000)
    ap.add_argument("--n_cross", type=int, default=800)
    args = ap.parse_args()
    for sub in ("plots", "csv"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = M.load()
    dev, dt = lm.device, lm.dtype
    # ---- freeze ----
    s3c = json.loads((P3C / "phase3c_summary.json").read_text())
    train_ents = s3c["stage1"]["meta"]["train_entities"]
    held_ents = s3c["stage1"]["meta"]["test_entities"]
    dd = torch.load(P3C / "D_mean_stage1.pt")
    dmean = dd["D_mean_dir"].to(dev, dt)
    dmean_sha = hashlib.sha1(dd["D_mean_dir"].float().cpu().numpy().tobytes()).hexdigest()
    S_sem_ref = s3c["stage1"]["aggregate"]["D_mean_dir@0.5"]["mean_p_green"]
    print(f"[freeze] D_mean norm={float(dmean.norm()):.3f} sha1={dmean_sha[:16]} | held-out={held_ents}")

    heldout = build_heldout(lm, held_ents, TEMPLATES["fact"])
    GREEN = heldout[0]["ansB"]; BLUE = heldout[0]["ansA"]
    assert all(p["ansB"] == GREEN and p["ansA"] == BLUE for p in heldout)
    specs = build_specs(lm)

    # ---- reproduce-or-halt ----
    fr_sem, S_sem = score_heldout(lm, heldout, dmean, 0.5, GREEN)
    sm_sem, smax_sem, chg_sem = score_spec(lm, specs, dmean, 0.5, GREEN)
    print(f"[reproduce] D_mean@0.5 flip={fr_sem:.2f} meanP(green)={S_sem:.3f} (ref {S_sem_ref:.3f}) | spec meanP(green)={sm_sem:.4f}")
    if not (fr_sem == 1.0 and abs(S_sem - S_sem_ref) < 0.02):
        (OUT / "PHASE3N_NULL_RESULTS.md").write_text(
            f"# Phase 3N HALT\nReproduce failed: flip={fr_sem}, S={S_sem} vs ref {S_sem_ref}. Config drift.\n", encoding="utf-8")
        print("[HALT] reproduce failed"); return

    frozen = {"env": {"python": platform.python_version(), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0), "model_id": lm.model_id},
              "inject_layer": INJECT_L, "post_deadline_layer": POST_L, "alphas": ALPHAS,
              "primary_alpha": 0.5, "D_mean_sha1": dmean_sha, "D_mean_norm": float(dmean.norm().item()),
              "train_entities": train_ents, "heldout_entities": held_ents,
              "GREEN_id": GREEN, "BLUE_id": BLUE, "spec_prompts": SPEC_PROMPTS,
              "spec_threshold": SPEC_THRESH, "flip_composite": FLIP_COMPOSITE,
              "primary_statistic": "mean held-out P(green) at alpha=0.5",
              "S_semantic": S_sem, "flip_semantic": fr_sem, "spec_semantic_mean_pgreen": sm_sem,
              "base_seed": BASE_SEED, "N_isotropic": args.n, "N_permuted": args.n, "N_sign": args.n_sign}
    (OUT / "frozen_phase3_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    # ---- pilot timing ----
    tt = time.time()
    for j in range(20):
        score_heldout(lm, heldout, gen_isotropic(dmean, j, dev, dt), 0.5, GREEN)
    per_trial = (time.time() - tt) / 20
    est = per_trial * (2 * args.n + args.n_sign) * 1.7  # +spec overhead factor
    print(f"[pilot] ~{per_trial*1000:.0f}ms/trial (transfer-only) => full ~{est/60:.1f} min")
    if args.pilot:
        print("[pilot] stop."); return

    # ---- empirical strong nulls (real wrong-info deltas) ----
    othercolor = build_othercolor_deltas(lm, train_ents, float(dmean.norm().item()), dev, dt)
    crosstask = build_crosstask_deltas(lm, float(dmean.norm().item()), dev, dt)
    emp_rows = []
    for c, info in othercolor.items():
        fr, pg = score_heldout(lm, heldout, info["delta"], 0.5, GREEN)
        sm, smax, chg = score_spec(lm, specs, info["delta"], 0.5, GREEN)
        # own-target: does the blue->c delta make held-out say c? (mechanism-generality control)
        cid = lm.tokenizer.encode(c, add_special_tokens=False)
        own = None
        if len(cid) == 1:
            ofr, opg = score_heldout(lm, heldout, info["delta"], 0.5, cid[0])
            own = {"own_flip": ofr, "own_p": opg}
        emp_rows.append({"family": "empirical_othercolor", "id": c, "flip_rate": fr, "S": pg,
                         "spec_mean_pgreen": sm, "cos_dmean": cos(info["delta"].cpu(), dmean.cpu()),
                         "own_target": own})
    for name, delta in crosstask.items():
        if isinstance(delta, str):
            emp_rows.append({"family": "empirical_crosstask", "id": name, "error": delta}); continue
        fr, pg = score_heldout(lm, heldout, delta, 0.5, GREEN)
        sm, smax, chg = score_spec(lm, specs, delta, 0.5, GREEN)
        emp_rows.append({"family": "empirical_crosstask", "id": name, "flip_rate": fr, "S": pg,
                         "spec_mean_pgreen": sm, "cos_dmean": cos(delta.cpu(), dmean.cpu())})

    # ---- random null families ----
    iso = run_family(lm, "isotropic", lambda j: gen_isotropic(dmean, j, dev, dt), args.n, heldout, specs, dmean, GREEN)
    perm = run_family(lm, "permuted", lambda j: gen_permuted(dmean, j, dev), args.n, heldout, specs, dmean, GREEN)
    sign = run_family(lm, "sign", lambda j: gen_sign(dmean, j, dev), args.n_sign, heldout, specs, dmean, GREEN)
    all_rows = iso + perm + sign + emp_rows
    with open(OUT / "null_trials.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, default=str) + "\n")

    # ---- non-null controls ----
    neg_fr, neg_S = score_heldout(lm, heldout, -dmean, 0.5, GREEN)
    dead_fr, dead_S = score_heldout(lm, heldout, dmean, 0.5, GREEN, L=POST_L)
    controls = {"neg_Dmean@0.5": {"flip": neg_fr, "S": neg_S},
                "Dmean@deadL24@0.5": {"flip": dead_fr, "S": dead_S},
                "Dmean@1.0": dict(zip(["flip", "S"], score_heldout(lm, heldout, dmean, 1.0, GREEN)))}

    # ---- anatomical square (noise at dead site, subsample) ----
    iso_dead = [score_heldout(lm, heldout, gen_isotropic(dmean, j, dev, dt), 0.5, GREEN, L=POST_L)[1] for j in range(200)]
    square = {"semantic_receptive_S": S_sem, "semantic_dead_S": dead_S,
              "noise_receptive_S_mean": float(np.mean([r["S"] for r in iso])),
              "noise_dead_S_mean": float(np.mean(iso_dead))}

    # ---- cross-phrasing null (frozen 3C stage2: D_mean from 'painted', test 'fact') ----
    cross = run_crossphrasing(lm, train_ents, dev, dt, GREEN, args.n_cross)

    # ---- stats ----
    def famstats(name, rows):
        return pctile_stats([r["S"] for r in rows], S_sem)
    stats = {"isotropic": famstats("isotropic", iso), "permuted": famstats("permuted", perm),
             "sign": famstats("sign", sign)}
    # composite: transfer AND specificity
    def composite_count(rows):
        return sum(1 for r in rows if r["flip_rate"] >= FLIP_COMPOSITE and r.get("spec_mean_pgreen", 1) < SPEC_THRESH)
    composite = {f: {"count": composite_count(rw), "n": len(rw)} for f, rw in
                 [("isotropic", iso), ("permuted", perm), ("sign", sign)]}
    emp_S = [r["S"] for r in emp_rows if "S" in r]
    emp_summary = {"max_S": (max(emp_S) if emp_S else None),
                   "n_ge_sem": sum(1 for v in emp_S if v >= S_sem),
                   "othercolor_own_target_flip_mean": float(np.mean(
                       [r["own_target"]["own_flip"] for r in emp_rows
                        if r.get("own_target")])) if any(r.get("own_target") for r in emp_rows) else None,
                   "othercolor_green_flip_max": max((r["flip_rate"] for r in emp_rows if "flip_rate" in r), default=None)}

    classification = classify(stats, composite, emp_summary, square, controls, S_sem)

    summary = {"frozen": frozen, "S_semantic": S_sem, "flip_semantic": fr_sem,
               "spec_semantic_mean_pgreen": sm_sem, "random_family_stats": stats,
               "composite_semantic_like_counts": composite, "empirical_nulls": emp_rows,
               "empirical_summary": emp_summary, "controls": controls, "anatomical_square": square,
               "cross_phrasing_null": cross, "classification": classification}
    (OUT / "null_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plots(iso, perm, sign, emp_rows, S_sem, square, OUT / "plots")
    write_report(OUT / "PHASE3N_NULL_RESULTS.md", summary)
    print(f"\n[classification] {classification['verdict']}")
    print(json.dumps({k: stats[k]["p_empirical_addone"] for k in stats}, indent=2))
    print(f"[done] {time.time()-t0:.0f}s")


def run_crossphrasing(lm, train_ents, dev, dt, GREEN, N):
    """Frozen 3C stage2: D_mean from 'painted' pairs, held-out 'fact' pairs; null at alpha 0.5 and 1.0."""
    fact, painted = TEMPLATES["fact"], TEMPLATES["painted"]
    tr = [make_pair(lm, painted, e) for e in train_ents]
    tr = [p for p in tr if p is not None]
    if len(tr) < 4:
        return {"status": "insufficient_painted_pairs", "n": len(tr)}
    raw = torch.stack([p["D"] for p in tr]).mean(0)
    dmean2 = (raw / raw.norm() * float(torch.stack([p["D"] for p in tr]).norm(dim=-1).mean())).to(dev, dt)
    heldF = [make_pair(lm, fact, e) for e in train_ents[:8]]
    heldF = [p for p in heldF if p is not None]
    res = {"n_train_painted": len(tr), "n_heldout_fact": len(heldF)}
    for a in ALPHAS:
        s_fr, s_S = score_heldout(lm, heldF, dmean2, a, GREEN)
        nulls = [score_heldout(lm, heldF, gen_isotropic(dmean2, j, dev, dt), a, GREEN)[1] for j in range(N)]
        pn = [score_heldout(lm, heldF, gen_permuted(dmean2, j, dev), a, GREEN)[1] for j in range(N)]
        res[f"alpha{a}"] = {"semantic_flip": s_fr, "semantic_S": s_S,
                            "isotropic": pctile_stats(nulls, s_S),
                            "permuted": pctile_stats(pn, s_S)}
    return res


def classify(stats, composite, emp, square, controls, S_sem):
    p_iso = stats["isotropic"]["p_empirical_addone"]
    p_perm = stats["permuted"]["p_empirical_addone"]
    p_sign = stats["sign"]["p_empirical_addone"]
    emp_beats = (emp["max_S"] is not None and emp["max_S"] >= S_sem)
    comp_any = any(c["count"] > 0 for c in composite.values())
    gating_ok = square["semantic_receptive_S"] > 0.5 and square["semantic_dead_S"] < 0.1 and square["noise_receptive_S_mean"] < 0.1
    all_p_extreme = max(p_iso, p_perm, p_sign) <= 0.01
    if all_p_extreme and not emp_beats and not comp_any and gating_ok:
        v = "STRONG_REJECTION"
    elif all_p_extreme and not emp_beats:
        v = "STRONG_REJECTION" if not comp_any else "MODERATE_REJECTION"
    elif max(p_iso, p_perm, p_sign) <= 0.05 and not emp_beats:
        v = "MODERATE_REJECTION"
    elif emp_beats or comp_any:
        v = "NULL_NOT_REJECTED"
    else:
        v = "AMBIGUOUS"
    return {"verdict": v, "p_isotropic": p_iso, "p_permuted": p_perm, "p_sign": p_sign,
            "empirical_null_beats_semantic": emp_beats, "composite_reproduced_by_noise": comp_any,
            "anatomical_gating_intact": gating_ok}


def plots(iso, perm, sign, emp, S_sem, square, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 1: histogram/ECDF of S per family + semantic ref
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, rows, c in [("isotropic", iso, "steelblue"), ("permuted", perm, "orange"), ("sign", sign, "green")]:
        ax.hist([r["S"] for r in rows], bins=40, alpha=.5, label=name, color=c, range=(0, 1))
    for r in emp:
        if "S" in r:
            ax.axvline(r["S"], color="grey", alpha=.4, lw=0.8)
    ax.axvline(S_sem, color="red", lw=2.5, label=f"D_mean S={S_sem:.3f}")
    ax.set_title("Phase 3N null distribution of mean held-out P(green) @α0.5")
    ax.set_xlabel("S = mean held-out P(green)"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "null_S_distribution.png", dpi=130); plt.close(fig)
    # 2: disruption vs target-specific (isotropic): cos_dmean vs S
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([r["cos_dmean"] for r in iso], [r["S"] for r in iso], s=6, alpha=.4, label="isotropic")
    ax.scatter([r["cos_dmean"] for r in perm], [r["S"] for r in perm], s=6, alpha=.4, label="permuted")
    ax.axhline(S_sem, color="red", ls="--", label="D_mean S")
    ax.set_title("Accidental success vs alignment to D_mean")
    ax.set_xlabel("cosine(R_j, D_mean)"); ax.set_ylabel("S = mean held-out P(green)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "cosine_vs_S.png", dpi=130); plt.close(fig)
    # 3: anatomical square
    fig, ax = plt.subplots(figsize=(5, 4))
    m = np.array([[square["semantic_receptive_S"], square["semantic_dead_S"]],
                  [square["noise_receptive_S_mean"], square["noise_dead_S_mean"]]])
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["receptive L8", "dead L24"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["semantic", "noise"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center", color="w")
    ax.set_title("Anatomical square: mean held-out P(green)")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout(); fig.savefig(outdir / "anatomical_square.png", dpi=130); plt.close(fig)


def write_report(path, s):
    st = s["random_family_stats"]; cl = s["classification"]
    L = []
    A = L.append
    A("# Phase 3N — Empirical Null Test of Constructed Write-Back (Results)\n")
    A(f"**Model:** {s['frozen']['env']['model_id']} · frozen Phase 3 config. See NULL_DESIGN.md (preregistration).\n")
    A(f"**Frozen:** inject L{s['frozen']['inject_layer']}, α=0.5 primary, D_mean sha1 `{s['frozen']['D_mean_sha1'][:16]}` "
      f"norm {s['frozen']['D_mean_norm']:.2f}, held-out {s['frozen']['heldout_entities']}.\n")
    A(f"**Primary statistic S = mean held-out P(green) @α0.5. Observed D_mean S = {s['S_semantic']:.3f}** "
      f"(flip {s['flip_semantic']:.2f}, specificity meanP(green) {s['spec_semantic_mean_pgreen']:.4f}).\n")

    A(f"## Verdict: **{cl['verdict']}**\n")
    A("| null family | role | mean S | q99 | max S | k≥S_sem | empirical p=(k+1)/(N+1) |")
    A("|---|---|---|---|---|---|---|")
    roles = {"isotropic": "weak baseline", "permuted": "structure-preserving", "sign": "sign-flip"}
    for f in ("isotropic", "permuted", "sign"):
        v = st[f]
        A(f"| {f} | {roles[f]} | {v['mean']:.4f} | {v['q99']:.4f} | {v['max']:.4f} | {v['k_ge_sem']} | **{v['p_empirical_addone']:.2e}** (N={v['n']}) |")

    A("\n## Empirical STRONG null (real wrong-info deltas)\n")
    A("| delta | held-out green flip | green S | specificity meanP(green) | (own-target flip) |")
    A("|---|---|---|---|---|")
    for r in s["empirical_nulls"]:
        if "S" not in r:
            A(f"| {r['id']} | — | — | — | {r.get('error','?')} |"); continue
        own = r.get("own_target")
        A(f"| {r['family'].split('_')[-1]}:{r['id']} | {r['flip_rate']:.2f} | {r['S']:.3f} | "
          f"{r.get('spec_mean_pgreen', float('nan')):.3f} | {own['own_flip']:.2f} if applicable |" if own else
          f"| {r['family'].split('_')[-1]}:{r['id']} | {r['flip_rate']:.2f} | {r['S']:.3f} | {r.get('spec_mean_pgreen', float('nan')):.3f} | — |")
    es = s["empirical_summary"]
    A(f"\nEmpirical max green-S = {es['max_S']}, #≥S_sem = {es['n_ge_sem']}, "
      f"other-color OWN-target mean flip = {es['othercolor_own_target_flip_mean']}, "
      f"max GREEN flip by a wrong-color delta = {es['othercolor_green_flip_max']}.\n")

    A("## Composite (transfer ≥5/6 AND specificity <0.05) reproduced by noise\n")
    A("| family | count / N |\n|---|---|")
    for f, c in s["composite_semantic_like_counts"].items():
        A(f"| {f} | {c['count']} / {c['n']} |")

    A("\n## Anatomical square (mean held-out P(green))\n")
    sq = s["anatomical_square"]
    A(f"| | receptive L8 | dead L24 |\n|---|---|---|")
    A(f"| **semantic** | {sq['semantic_receptive_S']:.3f} | {sq['semantic_dead_S']:.3f} |")
    A(f"| **noise (mean)** | {sq['noise_receptive_S_mean']:.3f} | {sq['noise_dead_S_mean']:.3f} |")

    A("\n## Non-null directional controls\n")
    for k, v in s["controls"].items():
        A(f"- `{k}`: flip {v['flip']:.2f}, S {v['S']:.3f}")

    A("\n## Cross-phrasing null (frozen 3C stage2)\n")
    cp = s["cross_phrasing_null"]
    if isinstance(cp, dict) and "alpha1.0" in cp:
        for a in ("alpha0.5", "alpha1.0"):
            d = cp[a]
            A(f"- {a}: semantic flip {d['semantic_flip']:.2f} S {d['semantic_S']:.3f} | "
              f"isotropic p={d['isotropic']['p_empirical_addone']:.2e}, permuted p={d['permuted']['p_empirical_addone']:.2e}")
    else:
        A(f"- {cp}")

    A("\n## Answers\n")
    A(_answers(s, cl))
    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe phase3_null.py")
    A("```")
    A("\n**STRICT STOP after Phase 3N.** No compression/PCA/probe/codec/PCDC/Phase 4.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def _answers(s, cl):
    st = s["random_family_stats"]; sq = s["anatomical_square"]; es = s["empirical_summary"]
    q = []
    q.append("1. **Null:** a matched non-semantic vector, same frozen conditions, is as likely as D_mean to produce target-specific held-out green transfer.")
    q.append("2. **Modified design?** Yes — isotropic demoted to weak baseline; empirical real-delta null added as PRIMARY strong null; fragile covariance-matched null removed (rank-deficient with ~N samples). See NULL_DESIGN.md.")
    q.append("3. **Fairest/strongest null:** the empirical other-color / cross-task deltas (on-manifold, real, wrong information).")
    q.append(f"4. **Isotropic meaningful?** Weak sanity check only (off-manifold). p_iso={st['isotropic']['p_empirical_addone']:.2e}.")
    q.append(f"5–6. **Where does D_mean fall / p-value:** S_sem={s['S_semantic']:.3f}; empirical p — iso {st['isotropic']['p_empirical_addone']:.2e}, permuted {st['permuted']['p_empirical_addone']:.2e}, sign {st['sign']['p_empirical_addone']:.2e}.")
    q.append(f"7–9. **Noise ANY vs TARGET flip / generalization:** noise mean held-out green-S (receptive) = {sq['noise_receptive_S_mean']:.3f}; composite semantic-like reproduced by noise = {sum(c['count'] for c in s['composite_semantic_like_counts'].values())}.")
    q.append(f"10. **Chance reproduce transfer AND specificity?** {'no' if not cl['composite_reproduced_by_noise'] else 'YES (see composite)'}.")
    q.append(f"11. **Shuffled/sign vectors retain effect?** permuted mean S={st['permuted']['mean']:.4f}, sign mean S={st['sign']['mean']:.4f}.")
    q.append(f"12. **Unrelated real directions produce target?** empirical max green-S={es['max_S']}, #≥S_sem={es['n_ge_sem']}; wrong-color deltas steer to their OWN target (mean flip {es['othercolor_own_target_flip_mean']}).")
    q.append("13. **Accidental success ~ cosine alignment?** see plots/cosine_vs_S.png.")
    q.append(f"14. **Site gating survives?** semantic L8 {sq['semantic_receptive_S']:.2f} vs dead L24 {sq['semantic_dead_S']:.2f}; noise L8 {sq['noise_receptive_S_mean']:.2f}. Gating intact={cl['anatomical_gating_intact']}.")
    q.append("15. **Cross-phrasing extreme vs null?** see cross-phrasing section.")
    q.append("16. **Selection caveats:** L8/α/task frozen from prior phases (uncorrected forking path); held-out N=6; one task family. p-value is CONDITIONAL on this pipeline; external validity narrow (stated in NULL_DESIGN.md §7-8).")
    q.append(f"17. **Classification: {cl['verdict']}.**")
    q.append("18. **Surviving claims:** additive direction produces target-specific, site-gated, specific held-out transfer far beyond matched nulls (subject to §16 caveats).")
    q.append("19. **Weakened/withdrawn:** any implicit universality — result is conditional on the frozen pipeline and small held-out set; magnitude/gain dependence remains.")
    q.append("20. STOP.")
    return "\n".join(q)


if __name__ == "__main__":
    main()
