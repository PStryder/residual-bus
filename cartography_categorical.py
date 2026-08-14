"""
cartography_categorical.py -- Phase 2C: Causal Cartography of CATEGORICAL inference.

Phase 2A: copy  (semantic fact -> same value).
Phase 2B: arithmetic (numeric premise -> different numeric answer). Found a sharp
          source->final causal handoff at ~L22.
Phase 2C: threshold comparison (numeric premise -> CATEGORICAL yes/no). Tests
          whether that L21/L22 topology survives when the answer is a different
          KIND of information than the source.

Same validated apparatus as 2B (frozen Qwen2.5-1.5B-Instruct, FP32, eager,
deterministic, resid_post hooks, full layer x position capture, no KV cache,
no grad, single forward pass). carto_lib.py is reused UNCHANGED; the pure
IO/plot helpers are imported from cartography.py UNCHANGED. All new logic lives
here; all outputs go to results/phase2c/ only.

Convention: H[L] = resid_post of decoder block L = HF hidden_states[L+1].

Run:
  python cartography_categorical.py --pilot
  python cartography_categorical.py
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import model as M
import intervention as IV
import carto_lib as CL
from cartography import save_heatmap, rows_to_matrix, write_csv, load_done, append_jsonl

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "phase2c"
P2A = ROOT / "results" / "phase2a"
P2B = ROOT / "results" / "phase2b"
SYSTEM = "You are a helpful assistant."
CARTO_SEED = 13579

OPS = ["BtoA", "AtoB", "selfA", "selfB", "randA", "randB"]
T_CANDS = [2, 3, 4, 5]
X_CANDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# candidate templates: numeric premise -> categorical yes/no (answer never in prompt).
# The suffix nudges a single-word answer WITHOUT naming yes/no (naming them would
# put the answer token in the prompt and break the non-copy property).
SUFFIX = ' Answer in one word.'
TEMPLATES = [
    ("greater_than",
     lambda x, t: f"Is {x} greater than {t}?" + SUFFIX),
    ("more_than_holds",
     lambda x, t: f"The dax has {x} stones. The box holds {t} stones. "
                  f"Are there more stones than the box can hold?" + SUFFIX),
    ("over_capacity",
     lambda x, t: f"The dax has {x} stones. A box holds {t} stones. Is the dax over capacity?" + SUFFIX),
    ("over_limit",
     lambda x, t: f"The dax has {x} stones. The limit is {t} stones. Is the dax over the limit?" + SUFFIX),
    ("exceed_limit",
     lambda x, t: f"The dax has {x} stones. The limit is {t} stones. Does the dax exceed the limit?" + SUFFIX),
]

# globals set once the pair is chosen
ANS_A = ANS_B = None
P_SOURCE = P_THRESHOLD = None
C_A = C_B = S_NATURAL = None


# ---------------------------------------------------------------------------
# task / pair search
# ---------------------------------------------------------------------------

def find_categorical_pair(lm):
    tok = lm.tokenizer
    search_log = []
    best = None
    for tname, tf in TEMPLATES:
        cache = {}
        n_correct = 0
        for t in T_CANDS:
            for x in X_CANDS:
                if x == t:
                    continue
                ids = M.build_inputs(lm, tf(x, t), SYSTEM)
                logits = M.forward_logits(lm, ids)[0]
                ans_id = int(torch.argmax(logits).item())
                ans = tok.decode([ans_id]).strip().lower()
                expected = "yes" if x > t else "no"
                p = float(torch.softmax(logits.double(), -1)[ans_id].item())
                cache[(t, x)] = {"ids": ids, "ans_id": ans_id, "ans": ans,
                                 "correct": ans == expected, "p_ans": p, "seq": ids.shape[1]}
                n_correct += int(ans == expected)
        # find aligned non-copy opposite pair (one below, one above threshold)
        tbest = None
        for t in T_CANDS:
            below = [x for x in X_CANDS if x < t and cache[(t, x)]["correct"]]
            above = [x for x in X_CANDS if x > t and cache[(t, x)]["correct"]]
            for xa in below:
                for xb in above:
                    A, B = cache[(t, xa)], cache[(t, xb)]
                    if A["seq"] != B["seq"]:
                        continue
                    ia, ib = A["ids"][0], B["ids"][0]
                    diff = (ia != ib).nonzero(as_tuple=True)[0].tolist()
                    if len(diff) != 1:
                        continue
                    p = diff[0]
                    if tok.decode([int(ia[p])]).strip() != str(xa):
                        continue
                    if tok.decode([int(ib[p])]).strip() != str(xb):
                        continue
                    if A["ans_id"] == B["ans_id"]:
                        continue
                    # non-copy: neither yes/no token appears in its prompt
                    if A["ans_id"] in ia.tolist() or B["ans_id"] in ib.tolist():
                        continue
                    score = min(A["p_ans"], B["p_ans"])
                    cand = {"template": tname, "t": t, "xA": xa, "xB": xb,
                            "ids_A": A["ids"], "ids_B": B["ids"], "seq": A["seq"], "P_source": p,
                            "answer_A": A["ans_id"], "answer_B": B["ans_id"],
                            "answer_tok_A": tok.decode([A["ans_id"]]),
                            "answer_tok_B": tok.decode([B["ans_id"]]), "score": score}
                    if tbest is None or score > tbest["score"]:
                        tbest = cand
        search_log.append({"template": tname, "n_correct_baselines": n_correct,
                           "found_pair": tbest is not None,
                           "best_score": (tbest["score"] if tbest else None)})
        if tbest and (best is None or tbest["score"] > best["score"]):
            best = tbest
    if best is None:
        raise RuntimeError("No aligned non-copy categorical pair found. Log: "
                           + json.dumps(search_log))
    best["search_log"] = search_log
    best["tf"] = dict(TEMPLATES)[best["template"]]
    return best


def token_role_cat(frag, pos, seq, p_source, p_threshold):
    f = frag.strip().lower()
    if pos == p_source:
        return "source_value"
    if pos == p_threshold:
        return "threshold"
    if "<|im_start|>" in frag or "<|im_end|>" in frag or f in ("system", "user", "assistant"):
        return "chat_template"
    if pos >= seq - 3:
        return "assistant_prefix"
    if "?" in frag:
        return "question_end"
    if f in ("over", "exceed", "exceeds", "above", "too", "full", "limit", "capacity"):
        return "predicate"
    if f in ("is", "does", "the", "dax"):
        return "question"
    if f in ("stones", "stone", "box", "holds", "hold", "has"):
        return "object"
    if f in ("yes", "no", "reply", "just", "with", "or"):
        return "instruction"
    return "content"


def build_token_map(lm, ids, other_ids, p_source, p_threshold):
    tok = lm.tokenizer
    ia, ib = ids[0].tolist(), other_ids[0].tolist()
    seq = len(ia)
    return [{"pos": p, "id": ia[p], "frag": tok.decode([ia[p]]),
             "same": ia[p] == ib[p],
             "role": token_role_cat(tok.decode([ia[p]]), p, seq, p_source, p_threshold)}
            for p in range(seq)]


# ---------------------------------------------------------------------------
# cell runner (self-contained; identical metric semantics to Phase 2B)
# ---------------------------------------------------------------------------

def build_op(op, L, P, H_A, H_B, hidden, device, dtype):
    if op == "BtoA":
        return IV.op_replace(H_B[L, P]), "A"
    if op == "AtoB":
        return IV.op_replace(H_A[L, P]), "B"
    if op == "selfA":
        return IV.op_replace(H_A[L, P]), "A"
    if op == "selfB":
        return IV.op_replace(H_B[L, P]), "B"
    if op == "randA":
        d = CL.random_direction(hidden, CARTO_SEED ^ (L * 911 + P * 2003 + 1), device, dtype)
        return IV.op_replace(d * H_B[L, P].norm()), "A"
    if op == "randB":
        d = CL.random_direction(hidden, CARTO_SEED ^ (L * 911 + P * 2003 + 2), device, dtype)
        return IV.op_replace(d * H_A[L, P].norm()), "B"
    raise ValueError(op)


def run_cell(lm, op, L, P, ctx):
    H_A, H_B = ctx["H_A"], ctx["H_B"]
    op_fn, side = build_op(op, L, P, H_A, H_B, lm.hidden_size, lm.device, lm.dtype)
    if side == "A":
        run_ids, base_logits, other_logits = ctx["ids_A"], ctx["base_A"], ctx["base_B"]
        H_src, H_tgt, target_ans = H_A, H_B, ANS_B
    else:
        run_ids, base_logits, other_logits = ctx["ids_B"], ctx["base_B"], ctx["base_A"]
        H_src, H_tgt, target_ans = H_B, H_A, ANS_A
    res = CL.run_and_capture_all(lm, run_ids, write=(L, P, op_fn))
    lm_m = CL.logit_metrics(base_logits, res.logits, lm.tokenizer)
    ac = CL.answer_contrast(res.logits, ANS_A, ANS_B)
    C = ac["C"]
    transfer = (C - C_A) if side == "A" else (C_B - C)
    rec = {"op": op, "L": L, "P": P, "side": side,
           "kl_from_base": lm_m["kl_from_control"],
           "kl_from_other": CL.kl_div(other_logits, res.logits),
           "C": C, "semantic_transfer": transfer, "transfer_fraction": transfer / S_NATURAL,
           "p_ansA": ac["p_A"], "p_ansB": ac["p_B"],
           "argmax_id": lm_m["top1_id"], "argmax_tok": lm_m["top1_tok"],
           "argmax_is_target": lm_m["top1_id"] == target_ans,
           "top1_changed": lm_m["top1_changed"], "topk_overlap": lm_m["topk_overlap"],
           "entropy": lm_m["entropy"], "norm_before": res.norm_before, "norm_after": res.norm_after}
    inv = CL.causal_invariants(H_src, res.H, L, P)
    rec.update({"inv_earlier_zero": inv["earlier_positions_zero"],
                "inv_lower_zero": inv["lower_layers_zero"]})
    if op in ("BtoA", "AtoB"):
        prop = CL.propagation_semantic(H_src, H_tgt, res.H, L, P)
        rec.update({"prop_cos_final": prop["mean_cos_finalpos"],
                    "prop_distimprove_final": prop["mean_distimprove_finalpos"],
                    "prop_cos_site": prop["mean_cos_site"]})
    return rec


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def transfer_vs_layer_plot(rows_BA, rows_AB, pos, title, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ba = sorted([r for r in rows_BA if r["P"] == pos], key=lambda r: r["L"])
    ab = sorted([r for r in rows_AB if r["P"] == pos], key=lambda r: r["L"])
    if ba:
        ax.plot([r["L"] for r in ba], [r["semantic_transfer"] for r in ba], "-o", label="B->A", ms=3)
    if ab:
        ax.plot([r["L"] for r in ab], [r["semantic_transfer"] for r in ab], "-s", label="A->B", ms=3)
    ax.axhline(S_NATURAL, ls="--", c="grey", label="S_natural")
    ax.set_title(title); ax.set_xlabel("layer"); ax.set_ylabel("semantic_transfer")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def natural_final_plot(dn_final, jump_layer, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(dn_final)), dn_final, "-o", ms=3)
    ax.axvline(jump_layer, ls="--", c="red", label=f"max jump @L{jump_layer}")
    ax.set_title("Natural ||H_B - H_A|| at final position vs layer")
    ax.set_xlabel("layer"); ax.set_ylabel("||D_natural|| (final pos)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def migration_plot(rows_BA, n, seq, p_source, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    xs, ys, cs = [], [], []
    for L in range(n):
        lay = [r for r in rows_BA if r["L"] == L]
        if lay:
            b = max(lay, key=lambda r: r["semantic_transfer"])
            xs.append(L); ys.append(b["P"]); cs.append("red" if b["P"] == p_source else "blue")
    ax.scatter(xs, ys, c=cs, s=25)
    ax.axhline(p_source, ls="--", c="red", alpha=.5, label="P_source")
    ax.set_title("Strongest B->A transfer position vs layer (red=source)")
    ax.set_xlabel("layer"); ax.set_ylabel("token position"); ax.set_ylim(-1, seq)
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# cross-phase helpers
# ---------------------------------------------------------------------------

def natural_final_series(H_A, H_B):
    d = (H_B - H_A).norm(dim=-1)[:, -1]         # [layers] at final pos
    diffs = (d[1:] - d[:-1])
    jump = int(torch.argmax(diffs).item()) + 1
    return d.cpu().numpy(), jump


def load_phase(pdir, baseline_sub, sem_op, rand_op, flip_field, kl_field, source_pos, seq):
    """Compute a comparable stat block for a completed phase (read-only)."""
    cells = [json.loads(l) for l in (pdir / "patch_map" / "cells.jsonl").read_text().splitlines() if l.strip()]
    sem = [r for r in cells if r["op"] == sem_op]
    rnd = [r for r in cells if r["op"] == rand_op]
    final_pos = seq - 1
    src = [r for r in sem if r["P"] == source_pos]
    fin = [r for r in sem if r["P"] == final_pos]
    src_flip_layers = sorted(r["L"] for r in src if r.get(flip_field))
    fin_flip_layers = sorted(r["L"] for r in fin if r.get(flip_field))
    interior_flip = [r for r in sem if r.get(flip_field) and r["P"] not in (source_pos, final_pos)]
    H_A = torch.load(pdir / baseline_sub / "H_A.pt")
    H_B = torch.load(pdir / baseline_sub / "H_B.pt")
    dn_final, jump = natural_final_series(H_A, H_B)
    def rand_kl_band(lo, hi):
        v = [r[kl_field] for r in rnd if r["P"] == source_pos and lo <= r["L"] <= hi]
        return float(np.mean(v)) if v else None
    return {
        "source_deadline": (max(src_flip_layers) if src_flip_layers else None),
        "source_stripe_width": len(src_flip_layers),
        "final_onset": (min(fin_flip_layers) if fin_flip_layers else None),
        "final_flip_layers": fin_flip_layers,
        "n_interior_flip_sites": len(interior_flip),
        "natural_final_jump_layer": jump,
        "max_transfer": max((r["semantic_transfer"] for r in sem), default=None),
        "rand_kl_shallow_L0_5": rand_kl_band(0, 5),
        "rand_kl_deep_L16_21": rand_kl_band(16, 21),
        "rand_target_flip_rate": (sum(1 for r in rnd if r.get(flip_field)) / max(1, len(rnd))),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    global ANS_A, ANS_B, P_SOURCE, P_THRESHOLD, C_A, C_B, S_NATURAL

    for sub in ("baselines", "natural_delta", "patch_map", "propagation", "controls",
                "selected_raw", "heatmaps"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = M.load()
    n = lm.num_layers
    tok = lm.tokenizer
    print(f"[load] {lm.model_id}: {n} layers, hidden={lm.hidden_size}")

    pair = find_categorical_pair(lm)
    ANS_A, ANS_B = pair["answer_A"], pair["answer_B"]
    P_SOURCE = pair["P_source"]
    ids_A, ids_B, seq = pair["ids_A"], pair["ids_B"], pair["seq"]
    P_THRESHOLD = next((p for p in range(seq) if p != P_SOURCE
                        and tok.decode([int(ids_A[0][p])]).strip() == str(pair["t"])), None)
    print(f"[pair] template={pair['template']} | A: has {pair['xA']} (limit {pair['t']}) -> "
          f"{pair['answer_tok_A']!r}(id {ANS_A}) | B: {pair['xB']} -> {pair['answer_tok_B']!r}(id {ANS_B}) | "
          f"seq={seq} P_source={P_SOURCE} P_threshold={P_THRESHOLD}")

    assert ANS_A != ANS_B
    assert ANS_A not in ids_A[0].tolist(), "answer A in prompt (copy!)"
    assert ANS_B not in ids_B[0].tolist(), "answer B in prompt (copy!)"
    assert (ids_A[0] != ids_B[0]).sum().item() == 1, "prompts differ at >1 position"

    tmapA = build_token_map(lm, ids_A, ids_B, P_SOURCE, P_THRESHOLD)
    tmapB = build_token_map(lm, ids_B, ids_A, P_SOURCE, P_THRESHOLD)
    (OUT / "token_map_A.json").write_text(json.dumps(tmapA, indent=2), encoding="utf-8")
    (OUT / "token_map_B.json").write_text(json.dumps(tmapB, indent=2), encoding="utf-8")
    (OUT / "task_selection.json").write_text(json.dumps({
        "template": pair["template"], "template_str": pair["tf"](pair["xA"], pair["t"]).replace(str(pair["xA"]), "{X}"),
        "xA": pair["xA"], "xB": pair["xB"], "threshold": pair["t"],
        "answer_A": {"id": ANS_A, "tok": pair["answer_tok_A"]},
        "answer_B": {"id": ANS_B, "tok": pair["answer_tok_B"]},
        "P_source": P_SOURCE, "P_threshold": P_THRESHOLD, "seq": seq,
        "non_copy_verified": True, "search_log": pair["search_log"],
    }, indent=2), encoding="utf-8")
    frags = [r["frag"] for r in tmapA]

    # ---- Part 1: baselines + reproducibility ----
    bA1, bA2 = CL.run_and_capture_all(lm, ids_A), CL.run_and_capture_all(lm, ids_A)
    bB1, bB2 = CL.run_and_capture_all(lm, ids_B), CL.run_and_capture_all(lm, ids_B)
    H_A, H_B = bA1.H, bB1.H
    base_A, base_B = bA1.logits, bB1.logits
    repro = {"A_H": bool(torch.equal(bA1.H, bA2.H)), "A_logits": bool(torch.equal(bA1.logits, bA2.logits)),
             "B_H": bool(torch.equal(bB1.H, bB2.H)), "B_logits": bool(torch.equal(bB1.logits, bB2.logits))}
    acA, acB = CL.answer_contrast(base_A, ANS_A, ANS_B), CL.answer_contrast(base_B, ANS_A, ANS_B)
    C_A, C_B = acA["C"], acB["C"]
    S_NATURAL = C_B - C_A
    correctA = int(torch.argmax(base_A).item()) == ANS_A
    correctB = int(torch.argmax(base_B).item()) == ANS_B
    print(f"[baseline] A correct={correctA}(p={acA['p_A']:.3f}) B correct={correctB}(p={acB['p_B']:.3f}) | "
          f"C_A={C_A:.2f} C_B={C_B:.2f} S_natural={S_NATURAL:.2f} | repro={repro}")
    assert correctA and correctB, "baseline does not produce the correct categorical answer"

    torch.save(H_A.cpu(), OUT / "baselines" / "H_A.pt")
    torch.save(H_B.cpu(), OUT / "baselines" / "H_B.pt")
    torch.save(base_A.cpu(), OUT / "baselines" / "logits_A.pt")
    torch.save(base_B.cpu(), OUT / "baselines" / "logits_B.pt")

    # ---- Part 2/8: natural atlas + final-position divergence ----
    D = H_B - H_A
    dnorm = D.norm(dim=-1).cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(H_A, H_B, dim=-1).cpu().numpy()
    np.savetxt(OUT / "natural_delta" / "dnorm.csv", dnorm, delimiter=",")
    np.savetxt(OUT / "natural_delta" / "cos.csv", cos, delimiter=",")
    before = D[:, :P_SOURCE]
    before_zero = bool(before.abs().max().item() == 0.0) if before.numel() else True
    save_heatmap(dnorm, "Natural |D| = ||H_B - H_A||", "token position", "layer (resid_post)",
                 OUT / "heatmaps" / "natural_dnorm.png", token_frags=frags)
    dn_final, jump_layer = natural_final_series(H_A, H_B)
    np.savetxt(OUT / "natural_delta" / "final_dnorm.csv", dn_final, delimiter=",")
    natural_final_plot(dn_final, jump_layer, OUT / "heatmaps" / "natural_final_divergence.png")
    print(f"[natural] before-source zero={before_zero} | final-pos jump @L{jump_layer}")

    ctx = {"ids_A": ids_A, "ids_B": ids_B, "H_A": H_A, "H_B": H_B, "base_A": base_A, "base_B": base_B}

    # ---- timing estimate ----
    ncells = n * seq * len(OPS)
    t = time.time()
    for kk in range(6):
        run_cell(lm, "BtoA", kk % n, seq - 1, ctx)
    per = (time.time() - t) / 6
    print(f"[estimate] {ncells} cells @ {per*1000:.0f}ms => ~{ncells*per/60:.1f} min")

    meta = {"env": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "model_id": lm.model_id, "dtype": str(lm.dtype)},
            "n_layers": n, "hidden": lm.hidden_size, "seq": seq,
            "task": "numeric_threshold -> categorical (non-copy)",
            "pair": {"template": pair["template"], "xA": pair["xA"], "xB": pair["xB"], "threshold": pair["t"],
                     "answer_A": ANS_A, "answer_B": ANS_B,
                     "answer_tok_A": pair["answer_tok_A"], "answer_tok_B": pair["answer_tok_B"],
                     "P_source": P_SOURCE, "P_threshold": P_THRESHOLD},
            "C_A": C_A, "C_B": C_B, "S_natural": S_NATURAL,
            "baseline_correct": {"A": correctA, "B": correctB}, "baseline_repro": repro,
            "natural_final_jump_layer": jump_layer, "natural_before_source_zero": before_zero,
            "n_cells": ncells, "per_cell_s": per,
            "layer_convention": "H[L] = resid_post of block L = HF hidden_states[L+1]"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.pilot:
        print(f"[pilot] done {time.time()-t0:.1f}s")
        return

    # ---- Part 3/4/5: bidirectional sweep (resumable) ----
    cellsf = OUT / "patch_map" / "cells.jsonl"
    done, rows = load_done(cellsf)
    print(f"[sweep] resuming {len(done)}/{ncells}")
    t = time.time()
    for L in range(n):
        for P in range(seq):
            for op in OPS:
                if (op, L, P) in done:
                    continue
                rec = run_cell(lm, op, L, P, ctx)
                append_jsonl(cellsf, rec)
                rows.append(rec)
        print(f"[sweep] layer {L} done ({time.time()-t:.0f}s)")

    # ---- Part 14 regression ----
    reA, reB = CL.run_and_capture_all(lm, ids_A), CL.run_and_capture_all(lm, ids_B)
    regr = {"A_H": bool(torch.equal(reA.H, H_A)), "A_logits": bool(torch.equal(reA.logits, base_A)),
            "B_H": bool(torch.equal(reB.H, H_B)), "B_logits": bool(torch.equal(reB.logits, base_B))}
    print(f"[regression] {regr}")

    analyze_and_report(lm, meta, rows, dnorm, dn_final, jump_layer, frags, tmapA, regr, repro, ctx, before_zero)
    print(f"[done] total {time.time()-t0:.0f}s. Outputs in {OUT}")


def analyze_and_report(lm, meta, rows, dnorm, dn_final, jump_layer, frags, tmapA, regr, repro, ctx, before_zero):
    n, seq = meta["n_layers"], meta["seq"]
    PS, PT = meta["pair"]["P_source"], meta["pair"]["P_threshold"]
    role = {r["pos"]: r["role"] for r in tmapA}
    frag = {r["pos"]: r["frag"] for r in tmapA}
    def rws(op): return [r for r in rows if r["op"] == op]
    BtoA, AtoB = rws("BtoA"), rws("AtoB")
    selfA, selfB, randA = rws("selfA"), rws("selfB"), rws("randA")

    selfA_ok = max((abs(r["semantic_transfer"]) for r in selfA), default=0) < 1e-6 and \
        max((r["kl_from_base"] for r in selfA), default=0) < 1e-9
    selfB_ok = max((abs(r["semantic_transfer"]) for r in selfB), default=0) < 1e-6 and \
        max((r["kl_from_base"] for r in selfB), default=0) < 1e-9
    inv_ok = all(r["inv_earlier_zero"] and r["inv_lower_zero"] for r in rows)
    fBA = [r for r in BtoA if r["argmax_is_target"]]
    fAB = [r for r in AtoB if r["argmax_is_target"]]

    # bidirectional
    dBA = {(r["L"], r["P"]): r for r in BtoA}
    dAB = {(r["L"], r["P"]): r for r in AtoB}
    bidir_flip = [k for k in dBA if k in dAB and dBA[k]["argmax_is_target"] and dAB[k]["argmax_is_target"]]

    # source / final deadlines
    def flip_layers(rs, pos): return sorted(r["L"] for r in rs if r["P"] == pos and r["argmax_is_target"])
    src_dl_BA = (max(flip_layers(BtoA, PS)) if flip_layers(BtoA, PS) else None)
    src_dl_AB = (max(flip_layers(AtoB, PS)) if flip_layers(AtoB, PS) else None)
    fin_on_BA = (min(flip_layers(BtoA, seq - 1)) if flip_layers(BtoA, seq - 1) else None)
    fin_on_AB = (min(flip_layers(AtoB, seq - 1)) if flip_layers(AtoB, seq - 1) else None)

    # interior sufficiency
    interior_flip = [r for r in fBA if r["P"] not in (PS, seq - 1)]
    nonsrc = sorted([r for r in BtoA if r["P"] != PS], key=lambda r: r["semantic_transfer"], reverse=True)
    role_maxT = {}
    for r in BtoA:
        role_maxT.setdefault(role[r["P"]], -1e9)
        role_maxT[role[r["P"]]] = max(role_maxT[role[r["P"]]], r["semantic_transfer"])

    # random target-specificity
    rand_flip = [r for r in randA if r["argmax_is_target"]]

    # ---- heatmaps ----
    hm = OUT / "heatmaps"
    def HM(rs, op, field, title, **kw):
        mat = rows_to_matrix(rs, op, field, n, seq)
        np.savetxt(hm / f"{op}_{field}.csv", mat, delimiter=",")
        save_heatmap(mat, title, "token position", "layer (resid_post)",
                     hm / f"{op}_{field}.png", token_frags=frags, **kw)
    HM(BtoA, "BtoA", "semantic_transfer", "B->A semantic_transfer (categorical)", center0=True)
    HM(AtoB, "AtoB", "semantic_transfer", "A->B semantic_transfer (categorical)", center0=True)
    HM(BtoA, "BtoA", "argmax_is_target", "B->A argmax -> B category")
    HM(AtoB, "AtoB", "argmax_is_target", "A->B argmax -> A category")
    HM(BtoA, "BtoA", "kl_from_base", "B->A KL(A||patched)")
    HM(randA, "randA", "kl_from_base", "random-control KL")
    HM(randA, "randA", "semantic_transfer", "random-control semantic_transfer", center0=True)
    HM(BtoA, "BtoA", "prop_distimprove_final", "B->A distance-to-B improvement (final)", center0=True)
    HM(BtoA, "BtoA", "prop_cos_final", "B->A downstream alignment to natural A->B")
    bmat = np.full((n, seq), np.nan)
    for (L, P) in dBA:
        if (L, P) in dAB:
            bmat[L, P] = min(dBA[(L, P)]["semantic_transfer"], dAB[(L, P)]["semantic_transfer"])
    np.savetxt(hm / "bidirectional_min_transfer.csv", bmat, delimiter=",")
    save_heatmap(bmat, "bidirectional min(transfer)", "token position", "layer (resid_post)",
                 hm / "bidirectional_min_transfer.png", token_frags=frags, center0=True)

    transfer_vs_layer_plot(BtoA, AtoB, PS, f"Source (P={PS}) semantic_transfer vs layer",
                           hm / "source_transfer_vs_layer.png")
    transfer_vs_layer_plot(BtoA, AtoB, seq - 1, f"Final (P={seq-1}) semantic_transfer vs layer",
                           hm / "final_transfer_vs_layer.png")
    # csv for source/final vs layer
    def vl_csv(pos, path):
        with open(path, "w") as f:
            f.write("layer,BtoA_transfer,BtoA_flip,AtoB_transfer,AtoB_flip\n")
            for L in range(n):
                a = next((r for r in BtoA if r["L"] == L and r["P"] == pos), None)
                b = next((r for r in AtoB if r["L"] == L and r["P"] == pos), None)
                f.write(f"{L},{a['semantic_transfer']:.4f},{a['argmax_is_target']},"
                        f"{b['semantic_transfer']:.4f},{b['argmax_is_target']}\n")
    vl_csv(PS, hm / "source_transfer_vs_layer.csv")
    vl_csv(seq - 1, hm / "final_transfer_vs_layer.csv")
    migration_plot(BtoA, n, seq, PS, hm / "strongest_position_vs_layer.png")

    # ---- ranked tables ----
    def dump(rs, name, extra=True):
        for r in rs:
            r["role"] = role.get(r["P"], "?"); r["frag"] = frag.get(r["P"], "?")
        write_csv(rs, OUT / name, ["L", "P", "frag", "role", "semantic_transfer", "transfer_fraction",
                                   "argmax_is_target", "kl_from_base", "kl_from_other", "p_ansA", "p_ansB",
                                   *(["prop_cos_final", "prop_distimprove_final"] if extra else [])])
    dump(sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:40], "ranked_BtoA.csv")
    dump(sorted(AtoB, key=lambda r: r["semantic_transfer"], reverse=True)[:40], "ranked_AtoB.csv")
    dump(nonsrc[:40], "ranked_nonsource_BtoA.csv")
    dump(sorted(randA, key=lambda r: r["kl_from_base"], reverse=True)[:20], "ranked_random_disruption.csv", extra=False)

    # ---- selected raw ----
    sel = {("BtoA", r["L"], r["P"]) for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:3]}
    sel |= {("BtoA", r["L"], r["P"]) for r in nonsrc[:2]}
    if src_dl_BA is not None:
        sel.add(("BtoA", src_dl_BA, PS))
        if src_dl_BA + 1 < n:
            sel.add(("BtoA", src_dl_BA + 1, PS))   # just after deadline
    if fin_on_BA is not None:
        sel.add(("BtoA", fin_on_BA, seq - 1))
    sel |= {("randA", r["L"], r["P"]) for r in sorted(randA, key=lambda r: r["kl_from_base"], reverse=True)[:1]}
    traces = []
    for (opn, L, P) in sorted(sel):
        op_fn, _ = build_op(opn, L, P, ctx["H_A"], ctx["H_B"], lm.hidden_size, lm.device, lm.dtype)
        res = CL.run_and_capture_all(lm, ctx["ids_A"], write=(L, P, op_fn))
        tr = CL.propagation_trace_semantic(ctx["H_A"], ctx["H_B"], res.H, P)
        tr.update({"op": opn, "L": L, "P": P})
        traces.append(tr)
        torch.save(res.H.cpu(), OUT / "selected_raw" / f"Hprime_{opn}_L{L}_P{P}.pt")
    (OUT / "propagation" / "traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")

    # ---- Part 12: three-phase comparison ----
    cmp = three_phase_compare(meta, BtoA, randA, src_dl_BA, fin_on_BA, jump_layer, interior_flip)

    # ---- Part 13: boundary hypothesis ----
    boundary = classify_boundary(meta, cmp, src_dl_BA, fin_on_BA, jump_layer)

    coincide = (src_dl_BA is not None and fin_on_BA is not None
                and abs((src_dl_BA + 1) - fin_on_BA) <= 1 and abs(fin_on_BA - jump_layer) <= 1)

    verdict = {
        "cartography_sound": bool(selfA_ok and selfB_ok and inv_ok and all(regr.values())),
        "selfA_noop": selfA_ok, "selfB_noop": selfB_ok, "invariants_all_cells": inv_ok,
        "regression_exact": all(regr.values()), "baselines_reproducible": all(repro.values()),
        "categorical_transfer_BtoA": bool(fBA), "categorical_transfer_AtoB": bool(fAB),
        "bidirectional": bool(bidir_flip),
        "n_flip_BtoA": len(fBA), "n_flip_AtoB": len(fAB), "n_flip_bidirectional": len(bidir_flip),
        "max_transfer_BtoA": max((r["semantic_transfer"] for r in BtoA), default=None),
        "S_natural": meta["S_natural"],
        "source_write_deadline_BtoA": src_dl_BA, "source_write_deadline_AtoB": src_dl_AB,
        "final_onset_BtoA": fin_on_BA, "final_onset_AtoB": fin_on_AB,
        "natural_final_jump_layer": jump_layer,
        "three_signals_coincide": bool(coincide),
        "any_interior_sufficient": bool(interior_flip),
        "random_target_flip_count": len(rand_flip),
        "boundary_hypothesis": boundary["classification"],
    }
    summary = {"verdict": verdict, "regression": regr, "repro": repro,
               "controls": {"selfA_ok": selfA_ok, "selfB_ok": selfB_ok, "invariants_ok": inv_ok,
                            "random_target_flip_count": len(rand_flip),
                            "random_max_kl": max((r["kl_from_base"] for r in randA), default=None)},
               "interior_role_max_transfer": role_maxT,
               "top_BtoA": [{k: r[k] for k in ("L", "P", "semantic_transfer", "transfer_fraction",
                                               "argmax_is_target", "p_ansB", "prop_cos_final",
                                               "prop_distimprove_final")}
                            for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:15]],
               "top_nonsource": [{"L": r["L"], "P": r["P"], "role": role[r["P"]],
                                  "transfer": r["semantic_transfer"], "argmax_is_target": r["argmax_is_target"]}
                                 for r in nonsrc[:15]],
               "three_phase_comparison": cmp, "boundary": boundary,
               "natural_before_source_zero": before_zero}
    (OUT / "phase2c_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps({"n_cells": len(rows), "ops": OPS}, indent=2), encoding="utf-8")
    write_report(OUT / "CARTOGRAPHY_CATEGORICAL_RESULTS.md", summary, meta, tmapA, BtoA, AtoB,
                 nonsrc, cmp, boundary, role, frag, coincide)
    print(f"[verdict] {json.dumps(verdict, indent=2, default=str)}")


def three_phase_compare(meta, BtoA, randA, src_dl, fin_on, jump, interior_flip):
    PS = meta["pair"]["P_source"]
    out = {"phase2c": {
        "task": "categorical", "source_deadline": src_dl, "final_onset": fin_on,
        "natural_final_jump_layer": jump, "n_interior_flip_sites": len(interior_flip),
        "max_transfer": max((r["semantic_transfer"] for r in BtoA), default=None),
        "S_natural": meta["S_natural"],
        "source_stripe_width": len([r for r in BtoA if r["P"] == PS and r["argmax_is_target"]]),
        "rand_kl_shallow_L0_5": float(np.mean([r["kl_from_base"] for r in randA if r["P"] == PS and r["L"] <= 5]) or 0),
        "rand_kl_deep_L16_21": float(np.mean([r["kl_from_base"] for r in randA if r["P"] == PS and 16 <= r["L"] <= 21]) or 0),
        "rand_target_flip_rate": sum(1 for r in randA if r["argmax_is_target"]) / max(1, len(randA)),
    }}
    try:
        m2a = json.loads((P2A / "meta.json").read_text())
        out["phase2a"] = load_phase(P2A, "baseline", "patchB", "randnorm", "argmax_is_B", "kl_from_A",
                                    m2a["pair"]["diff_pos"], m2a["seq"])
        out["phase2a"]["task"] = "copy"
    except Exception as e:
        out["phase2a"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        m2b = json.loads((P2B / "meta.json").read_text())
        out["phase2b"] = load_phase(P2B, "baselines", "BtoA", "randA", "argmax_is_target", "kl_from_base",
                                    m2b["pair"]["P_source"], m2b["seq"])
        out["phase2b"]["task"] = "arithmetic"
    except Exception as e:
        out["phase2b"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def classify_boundary(meta, cmp, src_dl, fin_on, jump):
    """Classify the L21/L22 candidate-handoff hypothesis given 2A/2B/2C."""
    dls = []
    for ph in ("phase2a", "phase2b", "phase2c"):
        d = cmp.get(ph, {})
        if isinstance(d, dict) and d.get("source_deadline") is not None:
            dls.append(d["source_deadline"])
    # 2c signals
    coincide = (src_dl is not None and fin_on is not None
                and abs((src_dl + 1) - fin_on) <= 1 and abs(fin_on - jump) <= 1)
    spread = (max(dls) - min(dls)) if dls else None
    if coincide and spread is not None and spread <= 2:
        cls = "strengthened"
        why = (f"2C reproduces the coincident source->final handoff (source_dl={src_dl}, final_onset={fin_on}, "
               f"natural jump=L{jump}), and the source deadline across 2A/2B/2C spans only {spread} layers "
               f"({sorted(set(dls))}) despite three different output representation types.")
    elif coincide:
        cls = "strengthened_with_shift"
        why = (f"2C shows the same coincident handoff but the deadline moved (2A/2B/2C={sorted(set(dls))}); "
               f"boundary is real but task-modulated.")
    elif src_dl is not None and fin_on is not None:
        cls = "weakened"
        why = (f"2C shows source_dl={src_dl}, final_onset={fin_on}, natural jump=L{jump} that do NOT cleanly "
               f"coincide; the sharp 2B handoff did not fully reproduce for categorical output.")
    else:
        cls = "still_ambiguous"
        why = "2C did not produce a clean source or final flip pattern; insufficient to classify."
    return {"classification": cls, "why": why,
            "source_deadlines_2a2b2c": dls, "deadline_spread": spread,
            "coincide_2c": coincide, "conservative_label": "candidate architectural handoff region"}


def write_report(path, summary, meta, tmapA, BtoA, AtoB, nonsrc, cmp, boundary, role, frag, coincide):
    v = summary["verdict"]
    L = []
    A = L.append
    A("# Phase 2C — Causal Cartography of Categorical Inference (Results)\n")
    A(f"**Model:** {meta['env']['model_id']} · {meta['n_layers']} layers · hidden {meta['hidden']} · FP32 · eager · {meta['env']['gpu']}\n")
    A(f"**Task (numeric→categorical, non-copy):** template `{meta['pair']['template']}` — "
      f"*dax has X, limit/box {meta['pair']['threshold']}, is it over? yes/no*.\n")
    A(f"- A: X={meta['pair']['xA']} → `{meta['pair']['answer_tok_A']}` (id {meta['pair']['answer_A']}); "
      f"B: X={meta['pair']['xB']} → `{meta['pair']['answer_tok_B']}` (id {meta['pair']['answer_B']}).")
    A(f"- One-token diff at **P_source={meta['pair']['P_source']}**; threshold at P={meta['pair']['P_threshold']}; seq={meta['seq']}.")
    A(f"- **Non-copy asserted** (yes/no never in prompt). C_A={meta['C_A']:.2f}, C_B={meta['C_B']:.2f}, "
      f"S_natural={meta['S_natural']:.2f}.\n")

    A("## Verdict\n| check | result |\n|---|---|")
    for k in ("cartography_sound", "selfA_noop", "selfB_noop", "invariants_all_cells", "regression_exact",
              "baselines_reproducible", "categorical_transfer_BtoA", "categorical_transfer_AtoB",
              "bidirectional", "n_flip_BtoA", "n_flip_AtoB", "n_flip_bidirectional",
              "max_transfer_BtoA", "S_natural", "source_write_deadline_BtoA", "source_write_deadline_AtoB",
              "final_onset_BtoA", "final_onset_AtoB", "natural_final_jump_layer",
              "three_signals_coincide", "any_interior_sufficient", "random_target_flip_count",
              "boundary_hypothesis"):
        A(f"| {k} | {v[k]} |")

    A("\n## Source→final handoff\n")
    A(f"- **Source write deadline (B→A):** L{v['source_write_deadline_BtoA']} "
      f"(A→B: L{v['source_write_deadline_AtoB']}).")
    A(f"- **Final-position onset (B→A):** L{v['final_onset_BtoA']} (A→B: L{v['final_onset_AtoB']}).")
    A(f"- **Natural final-position divergence jump:** L{v['natural_final_jump_layer']}.")
    A(f"- **Three signals coincide:** {v['three_signals_coincide']}.")

    A("\n## Strongest B→A sites\n")
    A("| L | P | token | role | transfer | frac | flip | p_ansB | align | dist→B |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:12]:
        A(f"| {r['L']} | {r['P']} | `{frag[r['P']].strip()}` | {role[r['P']]} | {r['semantic_transfer']:.2f} | "
          f"{r['transfer_fraction']:.2f} | {r['argmax_is_target']} | {r['p_ansB']:.3f} | "
          f"{r.get('prop_cos_final', float('nan')):.3f} | {r.get('prop_distimprove_final', float('nan')):+.2f} |")

    A("\n## Interior causal sufficiency (max transfer by role)\n```json")
    A(json.dumps(summary["interior_role_max_transfer"], indent=2, default=str))
    A("```")
    A("\nStrongest NON-source sites:")
    A("\n| L | P | token | role | transfer | flip |\n|---|---|---|---|---|---|")
    for r in nonsrc[:8]:
        A(f"| {r['L']} | {r['P']} | `{frag[r['P']].strip()}` | {role[r['P']]} | {r['semantic_transfer']:.2f} | {r['argmax_is_target']} |")

    A("\n## Three-phase comparison (2A copy / 2B arithmetic / 2C categorical)\n```json")
    A(json.dumps(cmp, indent=2, default=str))
    A("```")

    A("\n## L21/L22 boundary hypothesis\n")
    A(f"**Classification: `{boundary['classification']}`** — {boundary['why']}")
    A(f"\nConservative label: *{boundary['conservative_label']}*. "
      f"Source deadlines 2A/2B/2C = {boundary['source_deadlines_2a2b2c']} (spread {boundary['deadline_spread']}).")

    A("\n## Report answers\n")
    A(_answers(summary, meta, cmp, boundary, nonsrc, role))

    A("\n## Files\n```")
    A("task_selection.json token_map_A/B.json baselines/ natural_delta/ patch_map/cells.jsonl")
    A("propagation/traces.json selected_raw/*.pt heatmaps/*.png *.csv ranked_*.csv phase2c_summary.json")
    A("```")
    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe cartography_categorical.py --pilot")
    A(".venv\\Scripts\\python.exe cartography_categorical.py")
    A("```")
    A("\n**STRICT STOP POINT.** Mapping pass only. No codec/adapter/probe/classifier training, "
      "no fine-tuning, no weight changes, no PCDC, no head-level decomposition.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def _answers(summary, meta, cmp, boundary, nonsrc, role):
    v = summary["verdict"]
    q = []
    q.append(f"1. **Numeric→categorical transfer?** {'Yes' if v['categorical_transfer_BtoA'] else 'No'} "
             f"— {v['n_flip_BtoA']} B→A sites flip the category to the correct opposite answer (never in prompt).")
    q.append(f"2. **Bidirectional?** {'Yes' if v['bidirectional'] else 'No'} ({v['n_flip_bidirectional']} sites both ways).")
    q.append(f"3. **Source writable where?** P{meta['pair']['P_source']}, through L{v['source_write_deadline_BtoA']}.")
    q.append(f"4. **Source write deadline:** L{v['source_write_deadline_BtoA']} (B→A) / L{v['source_write_deadline_AtoB']} (A→B).")
    q.append(f"5. **Final-position sufficiency onset:** L{v['final_onset_BtoA']} (B→A) / L{v['final_onset_AtoB']} (A→B).")
    q.append(f"6. **Natural final divergence transition:** jump at L{v['natural_final_jump_layer']}.")
    q.append(f"7. **Three signals coincide?** {v['three_signals_coincide']}.")
    q.append(f"8. **Interior sites causally sufficient?** {'Yes' if v['any_interior_sufficient'] else 'No'} "
             f"(strongest non-source role transfers below flip threshold unless noted).")
    q.append("9. **Migrate or jump?** see strongest_position_vs_layer.png — "
             + ("hard source→final jump." if not v['any_interior_sufficient'] else "intermediate site(s) present."))
    q.append(f"10. **Target-specific vs random?** random target-flip count = {v['random_target_flip_count']} "
             f"(vs {v['n_flip_BtoA']} real B→A flips).")
    q.append("11. **Downstream trajectory alignment?** top sites' prop_cos_final and dist→B improvement in the table.")
    q.append("12. **Vs 2A/2B?** see three-phase comparison JSON (deadlines, onsets, natural jump, random washout).")
    q.append(f"13. **Boundary hypothesis:** {boundary['classification']} — {boundary['why']}")
    q.append("14. **Ambiguous:** mechanism of the write (untested here); single-token small-integer idiosyncrasy; "
             "whether the answer token identity (yes/no vs digits) affects the final-onset sharpness.")
    q.append("15. **Surprises:** noted in the comparison (deadline stability across output types; interior inertness).")
    q.append("16. **Next anatomical test:** vary threshold magnitude / answer-token type, or probe WHICH component "
             "performs the L~22 source→final write (out of scope here).")
    return "\n".join(q)


if __name__ == "__main__":
    main()
