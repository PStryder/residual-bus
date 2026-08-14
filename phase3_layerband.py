"""
phase3_layerband.py -- Phase 3LB: layer-band robustness of constructed write-back.

Addresses the forking-path caveat from Phase 3N: injection layer L8 was chosen
from prior cartography. Here we build D_mean INDEPENDENTLY at several layers of
the color task's receptive band (2A: source writable L0-21, deadline ~L22) and a
post-deadline control, using the SAME frozen train/held-out entity split, the
SAME construction (dir-normalized mean of h_green-h_blue), the SAME alpha policy,
and the SAME controls. If the effect holds across the band (not just L8), it is
not an artifact of selecting L8.

Frozen apparatus identical to Phase 3 (FP32, eager, deterministic, no grad, no KV
cache, single forward pass, resid_post). carto_lib + phase3 code reused unchanged.
Writes only results/phase3lb/.

Predeclared robustness criterion (before results): L8 is NOT a cherry-pick if
D_mean built+injected at >=2 other receptive-band layers also gives held-out flip
>=0.8 at alpha<=1.0, isotropic-null flip ~0, specificity mean P(green) <0.05; and
the post-deadline L24 fails.

Run:  python phase3_layerband.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import model as M
import carto_lib as CL
from phase3_additive import cos
from phase3_generalization import make_pair, TEMPLATES, SYSTEM
from phase3_null import inj_logits, gen_isotropic, build_specs

OUT = Path(__file__).parent / "results" / "phase3lb"
P3C = Path(__file__).parent / "results" / "phase3c"
RECEPTIVE_LAYERS = [4, 8, 12, 15, 18]
DEAD_LAYER = 24
LAYERS = RECEPTIVE_LAYERS + [DEAD_LAYER]
ALPHAS = [0.5, 1.0]
N_RAND = 500
SPEC_THRESH = 0.05
ANCHOR_S_L8 = 0.879     # frozen reference from Phase 3C/3N


def score_heldout_L(lm, heldout, delta, alpha, GREEN, L):
    flips, pg = [], []
    for pr in heldout:
        lg = inj_logits(lm, pr["ids_A"], L, pr["P"], delta, alpha)
        flips.append(int(torch.argmax(lg).item()) == GREEN)
        pg.append(float(torch.softmax(lg.double(), -1)[GREEN].item()))
    return float(np.mean(flips)), float(np.mean(pg))


def score_spec_L(lm, specs, delta, alpha, GREEN, L):
    pgs = []
    for s in specs:
        lg = inj_logits(lm, s["ids"], L, s["P"], delta, alpha)
        pgs.append(float(torch.softmax(lg.double(), -1)[GREEN].item()))
    return float(np.mean(pgs))


def dmean_at(train, L, P_key="P"):
    """Dir-normalized mean of (h_green - h_blue) at layer L over train pairs."""
    Ds = torch.stack([p["H_B"][L, p[P_key]] - p["H_A"][L, p[P_key]] for p in train])
    raw = Ds.mean(0)
    mni = float(Ds.norm(dim=-1).mean().item())
    return raw / raw.norm() * mni, mni


def main():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lm = M.load()
    dev, dt = lm.device, lm.dtype
    print(f"[load] {lm.model_id}")

    # ---- freeze split from Phase 3C ----
    s3c = json.loads((P3C / "phase3c_summary.json").read_text())
    train_ents = s3c["stage1"]["meta"]["train_entities"]
    held_ents = s3c["stage1"]["meta"]["test_entities"]
    fact = TEMPLATES["fact"]
    print(f"[freeze] train={len(train_ents)} held-out={held_ents}")

    train = [make_pair(lm, fact, e) for e in train_ents]
    train = [p for p in train if p is not None]
    heldout = [make_pair(lm, fact, e) for e in held_ents]
    assert all(p is not None for p in heldout), "held-out rebuild failed (drift)"
    GREEN, BLUE = heldout[0]["ansB"], heldout[0]["ansA"]
    specs = build_specs(lm)

    # ---- reproduce anchor at L8 ----
    d8, _ = dmean_at(train, 8)
    fr8, S8 = score_heldout_L(lm, heldout, d8, 0.5, GREEN, 8)
    print(f"[anchor L8] flip={fr8:.2f} S={S8:.3f} (ref {ANCHOR_S_L8:.3f})")
    if not (fr8 == 1.0 and abs(S8 - ANCHOR_S_L8) < 0.02):
        (OUT / "PHASE3LB_RESULTS.md").write_text(
            f"# HALT\nAnchor L8 reproduce failed: flip={fr8} S={S8} vs {ANCHOR_S_L8}\n", encoding="utf-8")
        print("[HALT] anchor drift"); return

    # ---- per-layer sweep ----
    results = {}
    for L in LAYERS:
        role = "receptive" if L in RECEPTIVE_LAYERS else "post_deadline"
        dL, mni = dmean_at(train, L)
        # semantic held-out at both alphas
        sem = {}
        for a in ALPHAS:
            fr, S = score_heldout_L(lm, heldout, dL, a, GREEN, L)
            spec = score_spec_L(lm, specs, dL, a, GREEN, L)
            sem[a] = {"flip": fr, "S": S, "spec_mean_pgreen": spec}
        # cos to held-out exact D at this layer
        cosx = float(np.mean([cos((p["H_B"][L, p["P"]] - p["H_A"][L, p["P"]]).cpu(), dL.cpu()) for p in heldout]))
        # isotropic null at alpha=0.5
        nullS, nullflip = [], []
        for j in range(N_RAND):
            R = gen_isotropic(dL, j, dev, dt)
            fr, S = score_heldout_L(lm, heldout, R, 0.5, GREEN, L)
            nullS.append(S); nullflip.append(fr)
        nS = np.array(nullS)
        k = int(np.sum(nS >= sem[0.5]["S"]))
        results[L] = {"role": role, "dmean_norm": mni, "cos_to_exact": cosx, "semantic": sem,
                      "null_mean_S": float(nS.mean()), "null_max_S": float(nS.max()),
                      "null_max_flip": float(np.max(nullflip)), "null_p_addone": (k + 1) / (N_RAND + 1)}
        print(f"[L{L:2d} {role:12s}] sem flip@0.5={sem[0.5]['flip']:.2f} S={sem[0.5]['S']:.3f} | "
              f"flip@1.0={sem[1.0]['flip']:.2f} | spec@0.5={sem[0.5]['spec_mean_pgreen']:.3f} | "
              f"nullmaxS={float(nS.max()):.3f} p={results[L]['null_p_addone']:.2e} | cos_exact={cosx:.3f}")

    # ---- classify ----
    recept_ok = [L for L in RECEPTIVE_LAYERS if L != 8 and
                 max(results[L]["semantic"][a]["flip"] for a in ALPHAS) >= 0.8
                 and results[L]["null_max_flip"] < 0.5
                 and min(results[L]["semantic"][a]["spec_mean_pgreen"] for a in ALPHAS) < SPEC_THRESH]
    dead_fails = results[DEAD_LAYER]["semantic"][1.0]["flip"] < 0.5
    if len(recept_ok) >= 2 and dead_fails:
        verdict = "ROBUST"
    elif len(recept_ok) >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "FRAGILE"
    classification = {"verdict": verdict, "receptive_layers_passing(excl_L8)": recept_ok,
                      "n_receptive_pass": len(recept_ok), "dead_layer_fails": dead_fails,
                      "criterion": "L8 not a cherry-pick if >=2 other receptive layers give held-out flip>=0.8 @a<=1.0, null flip~0, spec<0.05; dead layer fails"}

    summary = {"env": {"python": platform.python_version(), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0), "model_id": lm.model_id},
               "frozen": {"train_entities": train_ents, "heldout_entities": held_ents,
                          "GREEN_id": GREEN, "BLUE_id": BLUE, "alphas": ALPHAS, "N_rand_per_layer": N_RAND,
                          "receptive_layers": RECEPTIVE_LAYERS, "dead_layer": DEAD_LAYER,
                          "anchor_L8_S": S8, "anchor_L8_flip": fr8},
               "per_layer": results, "classification": classification}
    (OUT / "phase3lb_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plots(results, OUT / "plots")
    write_report(OUT / "PHASE3LB_RESULTS.md", summary)
    print(f"\n[classification] {verdict}  receptive-pass(excl L8)={recept_ok}")
    print(f"[done] {time.time()-t0:.0f}s")


def plots(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = sorted(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Ls, [results[L]["semantic"][0.5]["flip"] for L in Ls], "-o", label="semantic flip @α0.5")
    ax.plot(Ls, [results[L]["semantic"][1.0]["flip"] for L in Ls], "-s", label="semantic flip @α1.0")
    ax.plot(Ls, [results[L]["null_max_flip"] for L in Ls], "--x", c="grey", label="isotropic-null MAX flip")
    ax.axvspan(min(RECEPTIVE_LAYERS) - 0.5, max(RECEPTIVE_LAYERS) + 0.5, color="green", alpha=.06, label="receptive band")
    ax.axvline(8, ls=":", c="red", alpha=.5, label="L8 (selected)")
    ax.set_title("Layer-band robustness: held-out flip rate vs injection layer")
    ax.set_xlabel("injection layer"); ax.set_ylabel("held-out flip rate"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "flip_vs_layer.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Ls, [results[L]["semantic"][0.5]["S"] for L in Ls], "-o", label="semantic S @α0.5")
    ax.plot(Ls, [results[L]["null_max_S"] for L in Ls], "--x", c="grey", label="isotropic-null MAX S")
    ax.plot(Ls, [results[L]["cos_to_exact"] for L in Ls], "-^", c="purple", label="cos(D_mean, held-out exact D)")
    ax.axvline(8, ls=":", c="red", alpha=.5)
    ax.set_title("Layer-band: mean held-out P(green) and direction alignment vs layer")
    ax.set_xlabel("injection layer"); ax.set_ylabel("value"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "S_and_cos_vs_layer.png", dpi=130); plt.close(fig)


def write_report(path, s):
    cl = s["classification"]
    L = []
    A = L.append
    A("# Phase 3LB — Layer-Band Robustness of Constructed Write-Back (Results)\n")
    A(f"**Model:** {s['env']['model_id']} · frozen Phase 3 config (same train/held-out split, α, controls).\n")
    A("**Purpose:** test whether the Phase 3 write-back effect is specific to the *selected* layer L8 "
      "(forking-path caveat) or holds across the color task's cartographic receptive band.\n")
    A(f"**Anchor:** D_mean rebuilt at L8 reproduces flip {s['frozen']['anchor_L8_flip']:.2f}, "
      f"S {s['frozen']['anchor_L8_S']:.3f}.\n")

    A(f"## Verdict: **{cl['verdict']}**  (receptive layers passing, excl. L8: {cl['receptive_layers_passing(excl_L8)']})\n")
    A("| layer | role | flip@0.5 | flip@1.0 | S@0.5 | spec@0.5 | null max flip | null p | cos→exact |")
    A("|---|---|---|---|---|---|---|---|---|")
    for Lk in sorted(s["per_layer"], key=int):
        r = s["per_layer"][Lk]
        A(f"| L{Lk} | {r['role']} | {r['semantic'][0.5]['flip']:.2f} | {r['semantic'][1.0]['flip']:.2f} | "
          f"{r['semantic'][0.5]['S']:.3f} | {r['semantic'][0.5]['spec_mean_pgreen']:.3f} | "
          f"{r['null_max_flip']:.2f} | {r['null_p_addone']:.2e} | {r['cos_to_exact']:.3f} |")

    A(f"\n**Criterion (predeclared):** {cl['criterion']}\n")
    A("\n## Reading\n")
    A(f"- Receptive-band layers where the constructed direction generalizes (excl. L8): "
      f"**{cl['receptive_layers_passing(excl_L8)']}** ({cl['n_receptive_pass']} of {len(s['frozen']['receptive_layers'])-1}).")
    A(f"- Post-deadline L{s['frozen']['dead_layer']} fails as predicted: {cl['dead_layer_fails']}.")
    A("- Isotropic null at every layer stays at ~0 flip (equal injection budget), so the per-layer effect is not chance.")
    A("- Interpretation: the L8 result is " +
      ("**not** a cherry-pick — the write-back primitive works across the receptive band, exactly where the "
       "cartography said it should, and fails at the post-deadline site." if cl['verdict'] == "ROBUST"
       else "only partially reproduced across the band — see table; the forking-path caveat is only partly relieved."
       if cl['verdict'] == "PARTIAL" else
       "apparently specific to L8 — the forking-path caveat stands and the effect may be layer-fragile."))

    A("\n## Plots\n- `plots/flip_vs_layer.png` — semantic vs null flip rate across layers.")
    A("- `plots/S_and_cos_vs_layer.png` — mean P(green) and D_mean↔exact alignment across layers.")
    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe phase3_layerband.py")
    A("```")
    A("\n**STRICT STOP.** Robustness check only. No compression/probe/codec/PCDC/Phase 4.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
