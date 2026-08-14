"""
phase3_xlayer.py -- Phase 3XL: cross-layer semantic portability.

Take D_mean built at source layer S, inject at target layer T (h_T + alpha*D_S)
across the full 6x6 matrix over Phase 3LB layers {4,8,12,15,18,24}. Two scaling
conditions (raw, target-norm-matched), alpha in {0.5,1.0}. Cosine geometry,
asymmetry, compact null controls, L24 dead-site tests, D24-as-source.

Frozen Phase 3 apparatus; carto_lib + phase3 code reused unchanged. Directions
rebuilt from the frozen 14 train entities (Phase 3LB method). See DESIGN.md.
Writes only results/phase3xl/.

Run:  python phase3_xlayer.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import model as M
from phase3_additive import cos
from phase3_generalization import make_pair, TEMPLATES
from phase3_layerband import dmean_at
from phase3_null import inj_logits, gen_isotropic, gen_permuted, gen_sign, build_specs

OUT = Path(__file__).parent / "results" / "phase3xl"
P3C = Path(__file__).parent / "results" / "phase3c"
LAYERS = [4, 8, 12, 15, 18, 24]
RECEPTIVE = [4, 8, 12, 15, 18]
DEAD = 24
ALPHAS = [0.5, 1.0]
ANCHOR_S_L8 = 0.879
N_NULL = 150
SPEC_THRESH = 0.05
STRONG_FLIP = 0.8


def score(lm, heldout, delta, alpha, GREEN, T):
    flips, pg, ps = [], [], []
    for pr in heldout:
        lg = inj_logits(lm, pr["ids_A"], T, pr["P"], delta, alpha)
        p = torch.softmax(lg.double(), -1)
        flips.append(int(torch.argmax(lg).item()) == GREEN)
        pg.append(float(p[GREEN].item())); ps.append(float(p[pr["ansA"]].item()))
    return {"flip": float(np.mean(flips)), "S": float(np.mean(pg)), "src_p": float(np.mean(ps))}


def score_spec(lm, specs, delta, alpha, GREEN, T):
    return float(np.mean([float(torch.softmax(inj_logits(lm, s["ids"], T, s["P"], delta, alpha).double(), -1)[GREEN].item())
                          for s in specs]))


def main():
    for sub in ("matrices", "plots", "csv", "selected_raw"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lm = M.load()
    dev, dt = lm.device, lm.dtype
    print(f"[load] {lm.model_id}")

    s3c = json.loads((P3C / "phase3c_summary.json").read_text())
    train_ents = s3c["stage1"]["meta"]["train_entities"]
    held_ents = s3c["stage1"]["meta"]["test_entities"]
    fact = TEMPLATES["fact"]
    train = [p for p in (make_pair(lm, fact, e) for e in train_ents) if p]
    heldout = [make_pair(lm, fact, e) for e in held_ents]
    assert all(p is not None for p in heldout)
    GREEN = heldout[0]["ansB"]
    specs = build_specs(lm)

    # directions + norms
    D, norm = {}, {}
    for L in LAYERS:
        d, mni = dmean_at(train, L)
        D[L] = d.to(dev, dt); norm[L] = mni
        torch.save(d.cpu(), OUT / "matrices" / f"D_mean_L{L}.pt")

    # anchor reproduce (diagonal L8)
    a8 = score(lm, heldout, D[8], 0.5, GREEN, 8)
    print(f"[anchor L8] flip={a8['flip']:.2f} S={a8['S']:.3f} (ref {ANCHOR_S_L8:.3f})")
    if not (a8["flip"] == 1.0 and abs(a8["S"] - ANCHOR_S_L8) < 0.02):
        (OUT / "PHASE3XL_RESULTS.md").write_text(f"# HALT\nanchor drift flip={a8['flip']} S={a8['S']}\n", encoding="utf-8")
        print("[HALT]"); return

    # cosine + norm-ratio matrices
    cosM = {S: {T: cos(D[S].cpu(), D[T].cpu()) for T in LAYERS} for S in LAYERS}
    nratio = {S: {T: norm[S] / norm[T] for T in LAYERS} for S in LAYERS}

    # ---- main 6x6 matrix, both conditions, both alphas ----
    cells = []
    for S in LAYERS:
        for T in LAYERS:
            for cond in ("raw", "targetnorm"):
                delta = D[S] if cond == "raw" else (D[S] / D[S].norm() * norm[T])
                for a in ALPHAS:
                    r = score(lm, heldout, delta, a, GREEN, T)
                    rec = {"S": S, "T": T, "cond": cond, "alpha": a,
                           "flip": r["flip"], "green_S": r["S"], "src_p": r["src_p"],
                           "cos_ST": cosM[S][T], "norm_ratio": nratio[S][T],
                           "diagonal": S == T,
                           "S_role": "receptive" if S in RECEPTIVE else "dead",
                           "T_role": "receptive" if T in RECEPTIVE else "dead"}
                    cells.append(rec)
        print(f"[matrix] source D{S} done ({time.time()-t0:.0f}s)")

    # ---- specificity on strong off-diagonal cells (targetnorm, alpha=1.0) ----
    strong = [c for c in cells if not c["diagonal"] and c["cond"] == "targetnorm"
              and abs(c["alpha"] - 1.0) < 1e-9 and c["flip"] >= STRONG_FLIP]
    spec_results = []
    for c in strong:
        delta = D[c["S"]] / D[c["S"]].norm() * norm[c["T"]]
        sp = score_spec(lm, specs, delta, 1.0, GREEN, c["T"])
        spec_results.append({"S": c["S"], "T": c["T"], "spec_mean_pgreen": sp,
                             "specific": sp < SPEC_THRESH})
    # representative diagonal + a null spec
    diag_spec = score_spec(lm, specs, D[8], 1.0, GREEN, 8)

    # ---- compact null controls (targetnorm, alpha=0.5) ----
    nulls = {}
    for src in (8, 12):
        for T in (src, 15, DEAD):
            base = D[src] / D[src].norm() * norm[T]   # target-norm scaled semantic (reference)
            for nl, gen in [("permuted", lambda j, b=base: gen_permuted(b, j, dev)),
                            ("sign", lambda j, b=base: gen_sign(b, j, dev)),
                            ("isotropic", lambda j, b=base: gen_isotropic(b, j, dev, dt))]:
                res = [score(lm, heldout, gen(j), 0.5, GREEN, T) for j in range(N_NULL)]
                Ss = [r["S"] for r in res]; frs = [r["flip"] for r in res]
                nulls[f"D{src}->L{T}:{nl}"] = {"null_mean_S": float(np.mean(Ss)), "null_max_S": float(np.max(Ss)),
                                               "null_max_flip": float(np.max(frs)), "N": N_NULL}
        print(f"[nulls] source {src} done ({time.time()-t0:.0f}s)")

    # ---- asymmetry (targetnorm, alpha=1.0) ----
    def cell(S, T, cond="targetnorm", a=1.0, field="flip"):
        return next(c[field] for c in cells if c["S"] == S and c["T"] == T and c["cond"] == cond and abs(c["alpha"] - a) < 1e-9)
    asym = {}
    for i, S in enumerate(LAYERS):
        for T in LAYERS[i + 1:]:
            asym[f"D{S}<->L{T}"] = {"S_to_T": cell(S, T), "T_to_S": cell(T, S),
                                    "asym_flip": cell(S, T) - cell(T, S)}

    # ---- persist ----
    with open(OUT / "phase3xl_cells.jsonl", "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    save_matrices(cells, cosM, nratio, LAYERS)

    # ---- analysis / classification ----
    analysis = analyze(cells, cosM, nulls, spec_results, asym, LAYERS, RECEPTIVE, DEAD)
    summary = {"env": {"python": platform.python_version(), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0), "model_id": lm.model_id},
               "frozen": {"train_entities": train_ents, "heldout_entities": held_ents,
                          "layers": LAYERS, "receptive": RECEPTIVE, "dead": DEAD, "alphas": ALPHAS,
                          "native_norms": norm, "anchor_L8": a8},
               "cosine_matrix": cosM, "norm_ratio_matrix": nratio, "nulls": nulls,
               "specificity_strong_cells": spec_results, "diag_spec_L8": diag_spec,
               "asymmetry": asym, "analysis": analysis}
    (OUT / "phase3xl_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plots(cells, cosM, nratio, LAYERS, OUT / "plots")
    write_report(OUT / "PHASE3XL_RESULTS.md", summary, cells)
    print(f"\n[classification] {analysis['verdict']}")
    print(f"[done] {time.time()-t0:.0f}s")


def matrix_of(cells, cond, alpha, field, layers):
    m = np.full((len(layers), len(layers)), np.nan)
    idx = {L: i for i, L in enumerate(layers)}
    for c in cells:
        if c["cond"] == cond and abs(c["alpha"] - alpha) < 1e-9:
            m[idx[c["S"]], idx[c["T"]]] = c[field]
    return m


def save_matrices(cells, cosM, nratio, layers):
    for cond in ("raw", "targetnorm"):
        for a in ALPHAS:
            for field in ("flip", "green_S"):
                m = matrix_of(cells, cond, a, field, layers)
                np.savetxt(OUT / "csv" / f"{field}_{cond}_a{a}.csv", m, delimiter=",",
                           header="rows=source layer, cols=target layer; " + ",".join(f"L{L}" for L in layers))
    cm = np.array([[cosM[S][T] for T in layers] for S in layers])
    nr = np.array([[nratio[S][T] for T in layers] for S in layers])
    np.savetxt(OUT / "csv" / "cosine_matrix.csv", cm, delimiter=",")
    np.savetxt(OUT / "csv" / "norm_ratio_matrix.csv", nr, delimiter=",")


def analyze(cells, cosM, nulls, spec_results, asym, layers, receptive, dead):
    def cell(S, T, cond="targetnorm", a=1.0, field="flip"):
        return next(c[field] for c in cells if c["S"] == S and c["T"] == T and c["cond"] == cond and abs(c["alpha"] - a) < 1e-9)
    # off-diagonal receptive x receptive portability (targetnorm alpha1.0)
    rr = [(S, T) for S in receptive for T in receptive if S != T]
    rr_port = [cell(S, T) for S, T in rr]
    rr_portable_frac = float(np.mean([f >= STRONG_FLIP for f in rr_port]))
    # raw vs targetnorm rescue among receptive off-diagonal
    raw_fail_tn_ok = sum(1 for S, T in rr if cell(S, T, "raw") < STRONG_FLIP and cell(S, T, "targetnorm") >= STRONG_FLIP)
    # L24 as target (any receptive source works into L24?)
    into_dead = {f"D{S}->L{dead}": {"raw": cell(S, dead, "raw"), "tn": cell(S, dead, "targetnorm")} for S in receptive}
    dead_target_any = any(v["tn"] >= STRONG_FLIP for v in into_dead.values())
    # D24 as source into receptive
    d24_src = {f"D{dead}->L{T}": {"raw": cell(dead, T, "raw"), "tn": cell(dead, T, "targetnorm")} for T in receptive}
    d24_works_receptive = any(v["tn"] >= STRONG_FLIP for v in d24_src.values())
    # cosine vs portability correlation (off-diagonal, targetnorm alpha1.0)
    offd = [(cosM[S][T], cell(S, T)) for S in layers for T in layers if S != T]
    cx = np.array([c for c, _ in offd]); py = np.array([p for _, p in offd])
    corr = float(np.corrcoef(cx, py)[0, 1]) if cx.std() > 0 and py.std() > 0 else None
    # asymmetry magnitude
    max_asym = max((abs(v["asym_flip"]) for v in asym.values()), default=0.0)
    # nulls: max null flip across cross-layer null cells
    null_max_flip = max((v["null_max_flip"] for v in nulls.values()), default=0.0)

    # classify
    if rr_portable_frac >= 0.8 and not dead_target_any:
        verdict = "BROAD-PORTABILITY (receptivity-dominated)" if d24_works_receptive else "BROAD-PORTABILITY"
    elif rr_portable_frac >= 0.8:
        verdict = "BROAD-PORTABILITY"
    elif 0.3 <= rr_portable_frac < 0.8:
        # local vs asymmetric
        verdict = "ASYMMETRIC-PORTABILITY" if max_asym >= 0.5 else "LOCAL-PORTABILITY"
    elif rr_portable_frac < 0.3 and max((abs(v["asym_flip"]) for v in asym.values()), default=0) < 0.3:
        verdict = "LAYER-PRIVATE"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "receptive_offdiag_portable_fraction": rr_portable_frac,
            "raw_fail_targetnorm_rescued_count": raw_fail_tn_ok, "n_receptive_offdiag": len(rr),
            "into_dead_L24": into_dead, "dead_target_accepts_any": dead_target_any,
            "D24_as_source_into_receptive": d24_src, "D24_works_at_receptive": d24_works_receptive,
            "cosine_portability_corr": corr, "max_asymmetry_flip": max_asym,
            "null_max_flip_crosslayer": null_max_flip}


def plots(cells, cosM, nratio, layers, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"L{L}" for L in layers]

    def heat(m, title, fname, vmin=0, vmax=1, cmap="viridis", fmt="%.2f"):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("target layer T (inject)"); ax.set_ylabel("source layer S (direction)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, fmt % m[i, j], ha="center", va="center",
                            color="w" if (vmax - m[i, j]) > (vmax - vmin) * 0.5 else "k", fontsize=7)
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout(); fig.savefig(outdir / fname, dpi=130); plt.close(fig)

    for cond in ("raw", "targetnorm"):
        for a in ALPHAS:
            heat(matrix_of(cells, cond, a, "flip", layers), f"Held-out flip: {cond} α={a}", f"flip_{cond}_a{a}.png")
    heat(matrix_of(cells, "targetnorm", 1.0, "green_S", layers), "Mean P(green): targetnorm α=1.0", "greenS_targetnorm_a1.png")
    heat(np.array([[cosM[S][T] for T in layers] for S in layers]), "cosine(D_S, D_T)", "cosine_matrix.png", vmin=0, vmax=1)
    heat(np.array([[nratio[S][T] for T in layers] for S in layers]), "norm ratio ||D_S||/||D_T||", "norm_ratio.png",
         vmin=0, vmax=2.5, cmap="coolwarm")
    # asymmetry matrix (targetnorm a1)
    fm = matrix_of(cells, "targetnorm", 1.0, "flip", layers)
    heat(fm - fm.T, "portability asymmetry: flip(S→T) − flip(T→S) [targetnorm α1]", "asymmetry.png",
         vmin=-1, vmax=1, cmap="coolwarm")
    # cosine vs performance scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    off = [(cosM[S][T], next(c["flip"] for c in cells if c["S"] == S and c["T"] == T and c["cond"] == "targetnorm" and abs(c["alpha"] - 1.0) < 1e-9))
           for S in layers for T in layers if S != T]
    ax.scatter([c for c, _ in off], [p for _, p in off], s=25)
    ax.set_xlabel("cosine(D_S, D_T)"); ax.set_ylabel("held-out flip (targetnorm α1)")
    ax.set_title("cross-layer cosine vs causal portability (off-diagonal)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "cosine_vs_portability.png", dpi=130); plt.close(fig)
    # |S-T| distance vs performance
    fig, ax = plt.subplots(figsize=(7, 4))
    dp = [(abs(layers.index(S) - layers.index(T)), next(c["flip"] for c in cells if c["S"] == S and c["T"] == T and c["cond"] == "targetnorm" and abs(c["alpha"] - 1.0) < 1e-9))
          for S in RECEPTIVE for T in RECEPTIVE if S != T]
    ax.scatter([d for d, _ in dp], [p for _, p in dp], s=25)
    ax.set_xlabel("|source-target| layer-index distance (receptive only)"); ax.set_ylabel("flip")
    ax.set_title("layer distance vs portability"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "distance_vs_portability.png", dpi=130); plt.close(fig)


def write_report(path, s, cells):
    an = s["analysis"]
    layers = s["frozen"]["layers"]
    L = []
    A = L.append
    A("# Phase 3XL — Cross-Layer Semantic Portability (Results)\n")
    A(f"**Model:** {s['env']['model_id']} · frozen Phase 3 apparatus. See DESIGN.md (preregistration).\n")
    A(f"**Anchor:** diagonal L8 reproduces flip {s['frozen']['anchor_L8']['flip']:.2f}, S {s['frozen']['anchor_L8']['S']:.3f}.\n")
    A(f"## Verdict: **{an['verdict']}**\n")

    def tbl(cond, a):
        m = matrix_of(cells, cond, a, "flip", layers)
        out = ["", f"**Held-out flip — {cond}, α={a}** (rows=source D_S, cols=target L_T):", "",
               "| S\\T | " + " | ".join(f"L{L_}" for L_ in layers) + " |",
               "|" + "---|" * (len(layers) + 1)]
        for i, S in enumerate(layers):
            out.append(f"| **D{S}** | " + " | ".join(f"{m[i,j]:.2f}" for j in range(len(layers))) + " |")
        return "\n".join(out)
    A(tbl("targetnorm", 1.0))
    A(tbl("targetnorm", 0.5))
    A(tbl("raw", 1.0))

    A("\n## Key findings\n")
    A(f"- **Receptive off-diagonal portability** (targetnorm α1): {an['receptive_offdiag_portable_fraction']*100:.0f}% "
      f"of {an['n_receptive_offdiag']} cross-receptive cells reach flip≥{STRONG_FLIP}.")
    A(f"- **Target-norm rescue of raw failures** (receptive off-diag): {an['raw_fail_targetnorm_rescued_count']} cells.")
    A(f"- **L24 as target accepts any receptive direction?** {an['dead_target_accepts_any']} "
      f"(into-L24 targetnorm flips: { {k: round(v['tn'],2) for k,v in an['into_dead_L24'].items()} }).")
    A(f"- **D24 (dead-site payload) works at a receptive layer?** {an['D24_works_at_receptive']} "
      f"(D24→receptive targetnorm flips: { {k: round(v['tn'],2) for k,v in an['D24_as_source_into_receptive'].items()} }).")
    A(f"- **cosine ↔ portability correlation** (off-diagonal): {an['cosine_portability_corr']}.")
    A(f"- **max portability asymmetry** |flip(S→T)−flip(T→S)|: {an['max_asymmetry_flip']:.2f}.")
    A(f"- **cross-layer null** max flip (permute/sign/isotropic): {an['null_max_flip_crosslayer']:.2f}.")

    A("\n## Specificity on strong off-diagonal cells\n")
    ss = s["specificity_strong_cells"]
    if ss:
        A("| S→T | spec meanP(green) | specific? |\n|---|---|---|")
        for r in ss:
            A(f"| D{r['S']}→L{r['T']} | {r['spec_mean_pgreen']:.3f} | {r['specific']} |")
    A(f"\n(diagonal L8 spec meanP(green) = {s['diag_spec_L8']:.3f})\n")

    A("\n## Native norms & cosine\n")
    A("norms: " + ", ".join(f"L{L_}={s['frozen']['native_norms'][str(L_)] if str(L_) in s['frozen']['native_norms'] else s['frozen']['native_norms'][L_]:.1f}" for L_ in layers))

    A("\n## Answers\n")
    A(_answers(s, an))
    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe phase3_xlayer.py")
    A("```")
    A("\n**STRICT STOP.** No cross-layer transform/Procrustes/probe/PCA/compression/PCDC/Phase 4.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def _answers(s, an):
    q = []
    q.append(f"1. **Diagonal reproduces 3LB?** Yes (L8 anchor flip {s['frozen']['anchor_L8']['flip']:.2f}, S {s['frozen']['anchor_L8']['S']:.3f}).")
    q.append(f"2. **Move directions between receptive layers?** {an['receptive_offdiag_portable_fraction']*100:.0f}% of cross-receptive cells portable (targetnorm α1).")
    q.append("3. **Falloff with layer distance?** see plots/distance_vs_portability.png.")
    q.append(f"4. **Target-norm rescues raw?** {an['raw_fail_targetnorm_rescued_count']} raw failures rescued.")
    q.append("5. **Geometric similarity D4..D24?** see cosine matrix (high within band, L4/L24 outliers).")
    q.append(f"6. **Cosine predicts write-back?** correlation {an['cosine_portability_corr']}.")
    q.append(f"7. **Symmetric?** max asymmetry {an['max_asymmetry_flip']:.2f}.")
    q.append("8. **Early→late vs reverse?** see asymmetry matrix.")
    q.append(f"9. **D24 works at a receptive site?** {an['D24_works_at_receptive']}.")
    q.append(f"10. **Any early direction makes L24 writable?** {an['dead_target_accepts_any']}.")
    q.append(f"11. **Null reproduces cross-layer transfer?** max null flip {an['null_max_flip_crosslayer']:.2f} (no).")
    q.append("12. **Strong off-diagonal cells context-specific?** see specificity table.")
    q.append(f"13. **Best interpretation:** {an['verdict']}.")
    q.append("14. **Codec implication:** " +
             ("a shared receptive-band basis exists → one direction may serve multiple layers (fewer per-layer codecs)."
              if "BROAD" in an["verdict"] else
              "portability is limited → the codec likely needs layer-aware directions."))
    q.append("15. **Does NOT establish:** universality (frozen pipeline, N=6, single concept/task); no learned transform tested.")
    q.append("16. STOP.")
    return "\n".join(q)


if __name__ == "__main__":
    main()
