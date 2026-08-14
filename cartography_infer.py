"""
cartography_infer.py -- Phase 2B: Causal Cartography of DERIVED information.

Phase 2A mapped a COPY task (fact value == answer token). Phase 2B keeps the
exact same apparatus (frozen Qwen2.5-1.5B-Instruct, FP32, eager, deterministic,
resid_post hooks, full layer x position capture, no KV cache, no grad) but
changes the TASK to a minimal arithmetic transformation whose answer is NEVER
present in the prompt:

    "The dax starts with X stones. It receives K more stones. How many now?" -> X+K

A and B differ at exactly ONE source token (X). The answer (X+K) is a derived
value, not a copy. A semantic transfer therefore has to survive the model's
internal computation, not just be routed.

Reuses carto_lib.py (unchanged) and imports atlas/JSONL helpers from
cartography.py (unchanged). Writes ONLY to results/phase2b/. Never touches
results/phase2a/.

Run:
  python cartography_infer.py --pilot   # pair search + baselines + estimate, then stop
  python cartography_infer.py           # full bidirectional sweep + atlas + report
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
# reuse Phase 2A utilities (import has no side effects; __main__ guarded there)
from cartography import save_heatmap, rows_to_matrix, write_csv, load_done, append_jsonl

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "phase2b"
P2A = ROOT / "results" / "phase2a"
SYSTEM = "You are a helpful assistant."
CARTO_SEED = 24680

# candidate search space for the arithmetic pair
K_CANDIDATES = [4, 3, 2, 5]
X_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8]

def user_arith(x: int, k: int) -> str:
    return (f"The dax starts with {x} stones. It receives {k} more stones. "
            f"How many stones does the dax have now? Reply with just the number.")

# op set: 6 ops. B->A and A->B are the semantic patches; the rest are controls.
OPS = ["BtoA", "AtoB", "selfA", "selfB", "randA", "randB"]

# module globals set in main()
ANS_A = ANS_B = None
P_SOURCE = P_OPERAND = None
C_A = C_B = S_NATURAL = None


# ---------------------------------------------------------------------------
# Part 1 pre-req: task / pair search
# ---------------------------------------------------------------------------

def find_arithmetic_pair(lm) -> dict:
    """Search for a token-aligned NON-COPY arithmetic pair. Records the process."""
    tok = lm.tokenizer
    log = []
    # evaluate every (x,k): does the model compute x+k as a single next token?
    solved = {}  # k -> list of dicts
    for k in K_CANDIDATES:
        solved[k] = []
        for x in X_CANDIDATES:
            ids = M.build_inputs(lm, user_arith(x, k), SYSTEM)
            logits = M.forward_logits(lm, ids)[0]
            ans_id = int(torch.argmax(logits).item())
            ans_tok = tok.decode([ans_id]).strip()
            correct = ans_tok == str(x + k)
            log.append({"x": x, "k": k, "seq": ids.shape[1], "argmax": ans_tok,
                        "expected": str(x + k), "correct": correct})
            if correct:
                solved[k].append({"x": x, "k": k, "ids": ids, "seq": ids.shape[1],
                                  "ans_id": ans_id, "ans_tok": ans_tok})
    # find a valid pair (same k), token-aligned, non-copy
    for k in K_CANDIDATES:
        cand = solved[k]
        for i in range(len(cand)):
            for j in range(len(cand)):
                if i == j:
                    continue
                a, b = cand[i], cand[j]
                if a["seq"] != b["seq"]:
                    continue
                ia, ib = a["ids"][0], b["ids"][0]
                diff = (ia != ib).nonzero(as_tuple=True)[0].tolist()
                if len(diff) != 1:
                    continue
                p = diff[0]
                # source tokens must decode to the x values (single-token)
                if tok.decode([int(ia[p])]).strip() != str(a["x"]):
                    continue
                if tok.decode([int(ib[p])]).strip() != str(b["x"]):
                    continue
                # answers distinct and not equal to source tokens
                if a["ans_id"] == b["ans_id"]:
                    continue
                if a["ans_id"] == int(ia[p]) or b["ans_id"] == int(ib[p]):
                    continue
                # NON-COPY: answer token must not appear anywhere in its prompt
                if a["ans_id"] in ia.tolist() or b["ans_id"] in ib.tolist():
                    continue
                return {
                    "log": log,
                    "xA": a["x"], "xB": b["x"], "k": k,
                    "ids_A": a["ids"], "ids_B": b["ids"], "seq": a["seq"],
                    "P_source": p,
                    "answer_A": a["ans_id"], "answer_B": b["ans_id"],
                    "answer_tok_A": tok.decode([a["ans_id"]]),
                    "answer_tok_B": tok.decode([b["ans_id"]]),
                    "source_tok_A": tok.decode([int(ia[p])]),
                    "source_tok_B": tok.decode([int(ib[p])]),
                }
    raise RuntimeError("No token-aligned non-copy arithmetic pair found. "
                       "Solved cells: " + json.dumps({k: [c["x"] for c in v] for k, v in solved.items()}))


def token_role_arith(frag, pos, seq, p_source, p_operand):
    f = frag.strip()
    if pos == p_source:
        return "source_value"
    if pos == p_operand:
        return "operand"
    if "<|im_start|>" in frag or "<|im_end|>" in frag or f in ("system", "user", "assistant"):
        return "chat_template"
    if pos >= seq - 3:
        return "assistant_prefix"
    if "?" in frag:
        return "question_end"
    low = f.lower()
    if low in ("dax",):
        return "entity"
    if low in ("stones", "stone"):
        return "object"
    if low in ("how", "many", "now"):
        return "question"
    return "content"


def build_token_map(lm, ids, other_ids, p_source, p_operand):
    tok = lm.tokenizer
    ia = ids[0].tolist(); ib = other_ids[0].tolist()
    seq = len(ia)
    rows = []
    for p in range(seq):
        rows.append({"pos": p, "id": ia[p], "frag": tok.decode([ia[p]]),
                     "same": ia[p] == ib[p],
                     "role": token_role_arith(tok.decode([ia[p]]), p, seq, p_source, p_operand)})
    return rows


# ---------------------------------------------------------------------------
# op / vector builders
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
    """One patch cell. ctx bundles baselines. Returns a flat record dict."""
    H_A, H_B = ctx["H_A"], ctx["H_B"]
    op_fn, side = build_op(op, L, P, H_A, H_B, lm.hidden_size, lm.device, lm.dtype)
    if side == "A":
        run_ids, base_logits, other_logits = ctx["ids_A"], ctx["base_A"], ctx["base_B"]
        H_src, H_tgt, C_ref, target_ans = H_A, H_B, C_A, ANS_B
    else:
        run_ids, base_logits, other_logits = ctx["ids_B"], ctx["base_B"], ctx["base_A"]
        H_src, H_tgt, C_ref, target_ans = H_B, H_A, C_B, ANS_A

    res = CL.run_and_capture_all(lm, run_ids, write=(L, P, op_fn))
    lm_m = CL.logit_metrics(base_logits, res.logits, lm.tokenizer)
    ac = CL.answer_contrast(res.logits, ANS_A, ANS_B)
    C = ac["C"]
    # semantic transfer is always "movement toward the OTHER prompt's derived answer"
    transfer = (C - C_A) if side == "A" else (C_B - C)
    rec = {"op": op, "L": L, "P": P, "side": side,
           "kl_from_base": lm_m["kl_from_control"],
           "kl_from_other": CL.kl_div(other_logits, res.logits),
           "C": C, "semantic_transfer": transfer,
           "transfer_fraction": transfer / S_NATURAL,
           "p_ansA": ac["p_A"], "p_ansB": ac["p_B"],
           "logit_ansA": ac["logit_A"], "logit_ansB": ac["logit_B"],
           "argmax_id": lm_m["top1_id"], "argmax_tok": lm_m["top1_tok"],
           "argmax_is_target": lm_m["top1_id"] == target_ans,
           "top1_changed": lm_m["top1_changed"], "topk_overlap": lm_m["topk_overlap"],
           "entropy": lm_m["entropy"],
           "norm_before": res.norm_before, "norm_after": res.norm_after}
    inv = CL.causal_invariants(H_src, res.H, L, P)
    rec.update({"inv_earlier_zero": inv["earlier_positions_zero"],
                "inv_lower_zero": inv["lower_layers_zero"],
                "inv_earlier_max": inv["earlier_positions_max_abs"],
                "inv_lower_max": inv["lower_layers_max_abs"]})
    # propagation only for the semantic patches (controls don't need it)
    if op in ("BtoA", "AtoB"):
        prop = CL.propagation_semantic(H_src, H_tgt, res.H, L, P)
        rec.update({"prop_cos_final": prop["mean_cos_finalpos"],
                    "prop_ratio_final": prop["mean_ratio_finalpos"],
                    "prop_distimprove_final": prop["mean_distimprove_finalpos"],
                    "prop_cos_site": prop["mean_cos_site"],
                    "prop_distimprove_site": prop["mean_distimprove_site"]})
    return rec


# ---------------------------------------------------------------------------
# Part 2: natural A->B difference atlas
# ---------------------------------------------------------------------------

def natural_delta_atlas(H_A, H_B, p_source):
    n, seq, hid = H_A.shape
    D = H_B - H_A
    dnorm = D.norm(dim=-1)                                  # [n,seq]
    anorm = H_A.norm(dim=-1).clamp_min(1e-12)
    rel = (dnorm / anorm)
    cos = torch.nn.functional.cosine_similarity(H_A, H_B, dim=-1)  # [n,seq]
    maxabs = D.abs().amax(dim=-1)
    # causal invariant: positions strictly before source must be bitwise identical
    before = D[:, :p_source] if p_source > 0 else D[:, :0]
    before_max = float(before.abs().max().item()) if before.numel() else 0.0
    return {
        "dnorm": dnorm.cpu().numpy(), "rel": rel.cpu().numpy(),
        "cos": cos.cpu().numpy(), "maxabs": maxabs.cpu().numpy(),
        "before_source_max_abs": before_max,
        "before_source_bitwise_zero": before_max == 0.0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    global ANS_A, ANS_B, P_SOURCE, P_OPERAND, C_A, C_B, S_NATURAL

    for sub in ("baselines", "natural_delta", "patch_map", "propagation", "controls",
                "selected_raw", "heatmaps"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = M.load()
    n = lm.num_layers
    tok = lm.tokenizer
    print(f"[load] {lm.model_id}: {n} layers, hidden={lm.hidden_size}")

    # ---- task / pair ----
    pair = find_arithmetic_pair(lm)
    ANS_A, ANS_B = pair["answer_A"], pair["answer_B"]
    P_SOURCE = pair["P_source"]
    ids_A, ids_B = pair["ids_A"], pair["ids_B"]
    seq = pair["seq"]
    # operand position = the K digit that is not the source
    P_OPERAND = next((p for p in range(seq)
                      if p != P_SOURCE and tok.decode([int(ids_A[0][p])]).strip() == str(pair["k"])), None)
    print(f"[pair] A: start {pair['xA']}+{pair['k']} -> {pair['answer_tok_A']!r}(id {ANS_A}) | "
          f"B: {pair['xB']}+{pair['k']} -> {pair['answer_tok_B']!r}(id {ANS_B}) | "
          f"seq={seq} P_source={P_SOURCE} P_operand={P_OPERAND}")

    # non-copy + alignment assertions (fail loud)
    assert ANS_A != ANS_B, "answers not distinct"
    assert ANS_A not in ids_A[0].tolist(), "answer A appears in prompt A (copy!)"
    assert ANS_B not in ids_B[0].tolist(), "answer B appears in prompt B (copy!)"
    assert (ids_A[0] != ids_B[0]).sum().item() == 1, "prompts differ at >1 position"

    tmapA = build_token_map(lm, ids_A, ids_B, P_SOURCE, P_OPERAND)
    tmapB = build_token_map(lm, ids_B, ids_A, P_SOURCE, P_OPERAND)
    (OUT / "token_map_A.json").write_text(json.dumps(tmapA, indent=2), encoding="utf-8")
    (OUT / "token_map_B.json").write_text(json.dumps(tmapB, indent=2), encoding="utf-8")
    (OUT / "task_selection.json").write_text(json.dumps({
        "template": user_arith(pair["xA"], pair["k"]).replace(str(pair["xA"]), "{X}"),
        "xA": pair["xA"], "xB": pair["xB"], "k": pair["k"],
        "answer_A": {"id": ANS_A, "tok": pair["answer_tok_A"]},
        "answer_B": {"id": ANS_B, "tok": pair["answer_tok_B"]},
        "P_source": P_SOURCE, "P_operand": P_OPERAND, "seq": seq,
        "non_copy_verified": True,
        "search_log": pair["log"],
    }, indent=2), encoding="utf-8")
    frags = [r["frag"] for r in tmapA]

    # ---- Part 1: baselines (reproducibility) ----
    bA1 = CL.run_and_capture_all(lm, ids_A)
    bA2 = CL.run_and_capture_all(lm, ids_A)
    bB1 = CL.run_and_capture_all(lm, ids_B)
    bB2 = CL.run_and_capture_all(lm, ids_B)
    H_A, H_B = bA1.H, bB1.H
    base_A, base_B = bA1.logits, bB1.logits
    repro = {
        "A_H_exact": bool(torch.equal(bA1.H, bA2.H)),
        "A_logits_exact": bool(torch.equal(bA1.logits, bA2.logits)),
        "B_H_exact": bool(torch.equal(bB1.H, bB2.H)),
        "B_logits_exact": bool(torch.equal(bB1.logits, bB2.logits)),
    }
    acA = CL.answer_contrast(base_A, ANS_A, ANS_B)
    acB = CL.answer_contrast(base_B, ANS_A, ANS_B)
    C_A, C_B = acA["C"], acB["C"]
    S_NATURAL = C_B - C_A
    correctA = int(torch.argmax(base_A).item()) == ANS_A
    correctB = int(torch.argmax(base_B).item()) == ANS_B
    print(f"[baseline] A correct={correctA} (p_ans={acA['p_A']:.3f}) | "
          f"B correct={correctB} (p_ans={acB['p_B']:.3f}) | "
          f"C_A={C_A:.2f} C_B={C_B:.2f} S_natural={S_NATURAL:.2f} | repro={repro}")
    assert correctA and correctB, "a baseline does not produce the correct derived answer"

    torch.save(H_A.cpu(), OUT / "baselines" / "H_A.pt")
    torch.save(H_B.cpu(), OUT / "baselines" / "H_B.pt")
    torch.save(base_A.cpu(), OUT / "baselines" / "logits_A.pt")
    torch.save(base_B.cpu(), OUT / "baselines" / "logits_B.pt")

    # ---- Part 2: natural A->B atlas ----
    nat = natural_delta_atlas(H_A, H_B, P_SOURCE)
    np.savetxt(OUT / "natural_delta" / "dnorm.csv", nat["dnorm"], delimiter=",")
    np.savetxt(OUT / "natural_delta" / "rel.csv", nat["rel"], delimiter=",")
    np.savetxt(OUT / "natural_delta" / "cos.csv", nat["cos"], delimiter=",")
    save_heatmap(nat["dnorm"], "Natural |D| = ||H_B - H_A||", "token position",
                 "layer (resid_post)", OUT / "heatmaps" / "natural_dnorm.png", token_frags=frags)
    save_heatmap(1.0 - nat["cos"], "Natural 1 - cos(H_A,H_B)", "token position",
                 "layer (resid_post)", OUT / "heatmaps" / "natural_1mcos.png", token_frags=frags)
    print(f"[natural] before-source bitwise zero: {nat['before_source_bitwise_zero']} "
          f"(max {nat['before_source_max_abs']:.2e})")

    ctx = {"ids_A": ids_A, "ids_B": ids_B, "H_A": H_A, "H_B": H_B,
           "base_A": base_A, "base_B": base_B}

    # ---- timing estimate ----
    ncells = n * seq * len(OPS)
    t = time.time()
    for kk in range(6):
        run_cell(lm, "BtoA", kk % n, seq - 1, ctx)
    per = (time.time() - t) / 6
    est = ncells * per
    print(f"[estimate] {ncells} cells @ {per*1000:.0f}ms => ~{est/60:.1f} min")

    meta = {
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "model_id": lm.model_id, "dtype": str(lm.dtype)},
        "n_layers": n, "hidden": lm.hidden_size, "seq": seq,
        "task": "arithmetic_transform (non-copy)",
        "pair": {"xA": pair["xA"], "xB": pair["xB"], "k": pair["k"],
                 "answer_A": ANS_A, "answer_B": ANS_B,
                 "answer_tok_A": pair["answer_tok_A"], "answer_tok_B": pair["answer_tok_B"],
                 "P_source": P_SOURCE, "P_operand": P_OPERAND},
        "C_A": C_A, "C_B": C_B, "S_natural": S_NATURAL,
        "baseline_correct": {"A": correctA, "B": correctB}, "baseline_repro": repro,
        "n_cells": ncells, "per_cell_s": per, "est_minutes": est / 60,
        "layer_convention": "H[L] = resid_post of block L = HF hidden_states[L+1]",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.pilot:
        print(f"[pilot] done {time.time()-t0:.1f}s. Re-run without --pilot for the full sweep.")
        return

    # ---- Part 3/4/5: bidirectional sweep (resumable) ----
    cellsf = OUT / "patch_map" / "cells.jsonl"
    done, rows = load_done(cellsf)
    print(f"[sweep] resuming: {len(done)}/{ncells} cells done")
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

    # ---- Part 14: regression ----
    reA = CL.run_and_capture_all(lm, ids_A)
    reB = CL.run_and_capture_all(lm, ids_B)
    regr = {"A_H_exact": bool(torch.equal(reA.H, H_A)),
            "A_logits_exact": bool(torch.equal(reA.logits, base_A)),
            "B_H_exact": bool(torch.equal(reB.H, H_B)),
            "B_logits_exact": bool(torch.equal(reB.logits, base_B))}
    print(f"[regression] {regr}")

    # ---- analysis, atlas, report ----
    analyze_and_report(lm, meta, rows, nat, frags, tmapA, regr, repro, args.topk, ctx)
    print(f"[done] total {time.time()-t0:.0f}s. Outputs in {OUT}")


def analyze_and_report(lm, meta, rows, nat, frags, tmapA, regr, repro, topk, ctx):
    n, seq = meta["n_layers"], meta["seq"]
    role = {r["pos"]: r["role"] for r in tmapA}
    frag = {r["pos"]: r["frag"] for r in tmapA}
    def rws(op):
        return [r for r in rows if r["op"] == op]
    BtoA, AtoB = rws("BtoA"), rws("AtoB")
    selfA, selfB = rws("selfA"), rws("selfB")
    randA, randB = rws("randA"), rws("randB")

    # controls
    selfA_ok = max((abs(r["semantic_transfer"]) for r in selfA), default=0) < 1e-6 and \
        max((r["kl_from_base"] for r in selfA), default=0) < 1e-9
    selfB_ok = max((abs(r["semantic_transfer"]) for r in selfB), default=0) < 1e-6 and \
        max((r["kl_from_base"] for r in selfB), default=0) < 1e-9
    inv_ok = all(r["inv_earlier_zero"] and r["inv_lower_zero"] for r in rows)

    def flips(rs):
        return [r for r in rs if r["argmax_is_target"]]
    fBA, fAB = flips(BtoA), flips(AtoB)

    # bidirectional: join by (L,P)
    bd = {}
    dBA = {(r["L"], r["P"]): r for r in BtoA}
    dAB = {(r["L"], r["P"]): r for r in AtoB}
    for key in dBA:
        if key in dAB:
            bd[key] = min(dBA[key]["semantic_transfer"], dAB[key]["semantic_transfer"])
    bidir_flip = [k for k in dBA if k in dAB and dBA[k]["argmax_is_target"] and dAB[k]["argmax_is_target"]]

    # source-position transfer vs layer (both directions) + write deadline
    src_BA = sorted([r for r in BtoA if r["P"] == meta["pair"]["P_source"]], key=lambda r: r["L"])
    src_AB = sorted([r for r in AtoB if r["P"] == meta["pair"]["P_source"]], key=lambda r: r["L"])
    def deadline(src_rows):
        flip_layers = [r["L"] for r in src_rows if r["argmax_is_target"]]
        return max(flip_layers) if flip_layers else None
    src_deadline_BA = deadline(src_BA)
    src_deadline_AB = deadline(src_AB)

    # migration: strongest B->A transfer position per layer
    migr = []
    for L in range(n):
        lay = [r for r in BtoA if r["L"] == L]
        if lay:
            best = max(lay, key=lambda r: r["semantic_transfer"])
            migr.append({"L": L, "bestP": best["P"], "role": role[best["P"]],
                         "transfer": best["semantic_transfer"], "is_source": best["P"] == meta["pair"]["P_source"]})
    # non-source strong sites
    nonsrc = sorted([r for r in BtoA if r["P"] != meta["pair"]["P_source"]],
                    key=lambda r: r["semantic_transfer"], reverse=True)

    # ---- heatmaps ----
    hm = OUT / "heatmaps"
    def HM(rowset, op, field, title, **kw):
        mat = rows_to_matrix(rowset, op, field, n, seq)
        np.savetxt(hm / f"{op}_{field}.csv", mat, delimiter=",")
        save_heatmap(mat, title, "token position", "layer (resid_post)",
                     hm / f"{op}_{field}.png", token_frags=frags, **kw)
    HM(BtoA, "BtoA", "semantic_transfer", "B->A semantic_transfer (derived)", center0=True)
    HM(AtoB, "AtoB", "semantic_transfer", "A->B semantic_transfer (derived)", center0=True)
    HM(BtoA, "BtoA", "argmax_is_target", "B->A argmax -> B's derived answer")
    HM(AtoB, "AtoB", "argmax_is_target", "A->B argmax -> A's derived answer")
    HM(BtoA, "BtoA", "kl_from_base", "B->A KL(A || patched)")
    HM(randA, "randA", "kl_from_base", "random-control KL (A run)")
    HM(randA, "randA", "semantic_transfer", "random-control semantic_transfer", center0=True)
    HM(BtoA, "BtoA", "prop_distimprove_final", "B->A distance-to-B improvement (final pos)", center0=True)
    HM(BtoA, "BtoA", "prop_cos_final", "B->A downstream alignment to natural A->B")
    # bidirectional strength matrix
    bmat = np.full((n, seq), np.nan)
    for (L, P), v in bd.items():
        bmat[L, P] = v
    np.savetxt(hm / "bidirectional_min_transfer.csv", bmat, delimiter=",")
    save_heatmap(bmat, "bidirectional min(transfer) (both directions)", "token position",
                 "layer (resid_post)", hm / "bidirectional_min_transfer.png",
                 token_frags=frags, center0=True)

    # focused line plots
    line_plot(src_BA, src_AB, meta, hm / "source_transfer_vs_layer.png")
    migration_plot(migr, meta, seq, hm / "strongest_position_vs_layer.png")

    # ---- ranked tables ----
    def dump(rs, name, extra=()):
        for r in rs:
            r["role"] = role.get(r["P"], "?"); r["frag"] = frag.get(r["P"], "?")
        write_csv(rs, OUT / name, ["L", "P", "frag", "role", "semantic_transfer",
                                   "transfer_fraction", "argmax_is_target", "kl_from_base",
                                   "kl_from_other", "p_ansA", "p_ansB",
                                   *(["prop_cos_final", "prop_distimprove_final"] if extra else [])])
    dump(sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:40], "ranked_BtoA.csv", extra=True)
    dump(sorted(AtoB, key=lambda r: r["semantic_transfer"], reverse=True)[:40], "ranked_AtoB.csv", extra=True)
    dump(nonsrc[:40], "ranked_nonsource_BtoA.csv", extra=True)
    dump(sorted(randA, key=lambda r: r["kl_from_base"], reverse=True)[:20], "ranked_random_disruption.csv")
    bidir_rows = sorted([{"L": L, "P": P, "semantic_transfer": v,
                          "transfer_fraction": v / meta["S_natural"],
                          "argmax_is_target": (L, P) in [(k[0], k[1]) for k in bidir_flip],
                          "kl_from_base": dBA[(L, P)]["kl_from_base"], "kl_from_other": dBA[(L, P)]["kl_from_other"],
                          "p_ansA": dBA[(L, P)]["p_ansA"], "p_ansB": dBA[(L, P)]["p_ansB"]}
                         for (L, P), v in bd.items()], key=lambda r: r["semantic_transfer"], reverse=True)[:40]
    dump(bidir_rows, "ranked_bidirectional.csv")

    # ---- selected raw ----
    top_ba = sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:3]
    top_ns = nonsrc[:2]
    top_rand = sorted(randA, key=lambda r: r["kl_from_base"], reverse=True)[:1]
    sel = {("BtoA", r["L"], r["P"]) for r in top_ba} | {("BtoA", r["L"], r["P"]) for r in top_ns}
    if src_deadline_BA is not None:
        sel.add(("BtoA", src_deadline_BA, meta["pair"]["P_source"]))
    sel |= {("randA", r["L"], r["P"]) for r in top_rand}
    traces = []
    for (opn, L, P) in sorted(sel):
        op_fn, _ = build_op(opn, L, P, ctx["H_A"], ctx["H_B"], lm.hidden_size, lm.device, lm.dtype)
        res = CL.run_and_capture_all(lm, ctx["ids_A"], write=(L, P, op_fn))
        tr = CL.propagation_trace_semantic(ctx["H_A"], ctx["H_B"], res.H, P)
        tr.update({"op": opn, "L": L, "P": P})
        traces.append(tr)
        torch.save(res.H.cpu(), OUT / "selected_raw" / f"Hprime_{opn}_L{L}_P{P}.pt")
    (OUT / "propagation" / "traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")

    # ---- Phase 2A comparison ----
    cmp = compare_to_2a(meta, BtoA, randA, nat, src_deadline_BA, nonsrc)

    # ---- summary + report ----
    verdict = {
        "cartography_sound": bool(selfA_ok and selfB_ok and inv_ok
                                  and regr["A_H_exact"] and regr["B_H_exact"]
                                  and regr["A_logits_exact"] and regr["B_logits_exact"]),
        "selfA_noop": selfA_ok, "selfB_noop": selfB_ok,
        "invariants_all_cells": inv_ok,
        "regression_exact": all(regr.values()),
        "baselines_reproducible": all(repro.values()),
        "derived_transfer_BtoA_observed": bool(fBA),
        "derived_transfer_AtoB_observed": bool(fAB),
        "bidirectional_transfer_observed": bool(bidir_flip),
        "n_flip_BtoA": len(fBA), "n_flip_AtoB": len(fAB), "n_flip_bidirectional": len(bidir_flip),
        "max_transfer_BtoA": max((r["semantic_transfer"] for r in BtoA), default=None),
        "max_transfer_AtoB": max((r["semantic_transfer"] for r in AtoB), default=None),
        "S_natural": meta["S_natural"],
        "source_write_deadline_BtoA": src_deadline_BA,
        "source_write_deadline_AtoB": src_deadline_AB,
        "strongest_nonsource_site": ({"L": nonsrc[0]["L"], "P": nonsrc[0]["P"],
                                      "role": role[nonsrc[0]["P"]],
                                      "transfer": nonsrc[0]["semantic_transfer"],
                                      "argmax_is_target": nonsrc[0]["argmax_is_target"]} if nonsrc else None),
    }
    summary = {"verdict": verdict, "regression": regr, "baseline_repro": repro,
               "controls": {"selfA_ok": selfA_ok, "selfB_ok": selfB_ok, "invariants_ok": inv_ok},
               "migration": migr,
               "top_BtoA": [{k: r[k] for k in ("L", "P", "semantic_transfer", "transfer_fraction",
                                               "argmax_is_target", "p_ansB", "prop_cos_final",
                                               "prop_distimprove_final")}
                            for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:15]],
               "top_nonsource": [{"L": r["L"], "P": r["P"], "role": role[r["P"]],
                                  "transfer": r["semantic_transfer"], "argmax_is_target": r["argmax_is_target"]}
                                 for r in nonsrc[:15]],
               "phase2a_comparison": cmp,
               "natural_before_source_zero": nat["before_source_bitwise_zero"]}
    (OUT / "phase2b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps({"n_cells": len(rows),
                                                  "ops": OPS}, indent=2), encoding="utf-8")
    write_report(OUT / "CARTOGRAPHY_INFERENCE_RESULTS.md", summary, meta, tmapA,
                 BtoA, AtoB, nonsrc, migr, cmp, role, frag)
    print(f"[verdict] {json.dumps(verdict, indent=2, default=str)}")


def compare_to_2a(meta, BtoA, randA, nat, src_deadline_BA, nonsrc):
    """Quantitative comparison to the Phase 2A copy-task map (read-only)."""
    out = {"phase2a_available": False}
    try:
        s2a = json.loads((P2A / "cartography_summary.json").read_text())
        m2a = json.loads((P2A / "meta.json").read_text())
        p2a_rows = [json.loads(l) for l in (P2A / "patch_map" / "cells.jsonl").read_text().splitlines() if l.strip()]
        pB = [r for r in p2a_rows if r["op"] == "patchB"]
        rn = [r for r in p2a_rows if r["op"] == "randnorm"]
        src2a = m2a["pair"]["diff_pos"]
        # 2A source deadline: deepest layer with argmax flip at source pos
        src2a_flip_layers = [r["L"] for r in pB if r["P"] == src2a and r.get("argmax_is_B")]
        dl2a = max(src2a_flip_layers) if src2a_flip_layers else None
        # 2A non-source flip sites (excluding the trivial final-position/last-layer)
        ns2a = [r for r in pB if r["P"] != src2a and r.get("argmax_is_B")]
        # random washout: mean kl_from_A at source across shallow vs deep layers
        def rn_kl_band(rows, pos, lo, hi):
            v = [r["kl_from_A"] for r in rows if r["P"] == pos and lo <= r["L"] <= hi]
            return float(np.mean(v)) if v else None
        out.update({
            "phase2a_available": True,
            "task_2a": "copy", "task_2b": "arithmetic_derive",
            "source_deadline_2A": dl2a, "source_deadline_2B_BtoA": src_deadline_BA,
            "n_source_flip_layers_2A": len(src2a_flip_layers),
            "n_source_flip_layers_2B_BtoA": len([r for r in BtoA if r["P"] == meta["pair"]["P_source"] and r["argmax_is_target"]]),
            "n_nonsource_flip_sites_2A": len(ns2a),
            "n_nonsource_flip_sites_2B_BtoA": len([r for r in nonsrc if r["argmax_is_target"]]),
            "max_transfer_2A": s2a["verdict"]["max_semantic_transfer"],
            "S_natural_2A": s2a["verdict"]["natural_contrast_swing"],
            "max_transfer_2B_BtoA": max((r["semantic_transfer"] for r in BtoA), default=None),
            "S_natural_2B": meta["S_natural"],
            "rand_kl_shallow_2A(src,L0-5)": rn_kl_band(rn, src2a, 0, 5),
            "rand_kl_deep_2A(src,L16-21)": rn_kl_band(rn, src2a, 16, 21),
            "rand_kl_shallow_2B(src,L0-5)": float(np.mean([r["kl_from_base"] for r in randA
                                                          if r["P"] == meta["pair"]["P_source"] and r["L"] <= 5]) or 0),
            "rand_kl_deep_2B(src,L16-21)": float(np.mean([r["kl_from_base"] for r in randA
                                                          if r["P"] == meta["pair"]["P_source"] and 16 <= r["L"] <= 21]) or 0),
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def line_plot(src_BA, src_AB, meta, path):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    if src_BA:
        ax.plot([r["L"] for r in src_BA], [r["semantic_transfer"] for r in src_BA], "-o", label="B->A", ms=3)
    if src_AB:
        ax.plot([r["L"] for r in src_AB], [r["semantic_transfer"] for r in src_AB], "-s", label="A->B", ms=3)
    ax.axhline(meta["S_natural"], ls="--", c="grey", label="S_natural")
    ax.set_title(f"Source-token (P={meta['pair']['P_source']}) semantic_transfer vs layer")
    ax.set_xlabel("layer"); ax.set_ylabel("semantic_transfer"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def migration_plot(migr, meta, seq, path):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [m["L"] for m in migr]; ys = [m["bestP"] for m in migr]
    cs = ["red" if m["is_source"] else "blue" for m in migr]
    ax.scatter(xs, ys, c=cs, s=25)
    ax.axhline(meta["pair"]["P_source"], ls="--", c="red", alpha=.5, label="P_source")
    ax.set_title("Strongest B->A transfer position vs layer (red=source)")
    ax.set_xlabel("layer"); ax.set_ylabel("token position"); ax.set_ylim(-1, seq); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_report(path, summary, meta, tmapA, BtoA, AtoB, nonsrc, migr, cmp, role, frag):
    v = summary["verdict"]
    L = []
    A = L.append
    A("# Phase 2B — Causal Cartography of DERIVED Information (Results)\n")
    A(f"**Model:** {meta['env']['model_id']} · {meta['n_layers']} layers · hidden {meta['hidden']} · FP32 · eager · {meta['env']['gpu']}\n")
    A(f"**Task (non-copy):** `The dax starts with X stones. It receives {meta['pair']['k']} more … how many now?` → X+{meta['pair']['k']}.\n")
    A(f"- A: X={meta['pair']['xA']} → answer `{meta['pair']['answer_tok_A']}` (id {meta['pair']['answer_A']}); "
      f"B: X={meta['pair']['xB']} → `{meta['pair']['answer_tok_B']}` (id {meta['pair']['answer_B']}).")
    A(f"- One-token difference at **P_source={meta['pair']['P_source']}**; operand at P={meta['pair']['P_operand']}; seq={meta['seq']}.")
    A(f"- **Non-copy asserted:** answer token never appears in its prompt. C_A={meta['C_A']:.2f}, C_B={meta['C_B']:.2f}, "
      f"S_natural={meta['S_natural']:.2f}.\n")

    A("## Verdict\n")
    A("| check | result |")
    A("|---|---|")
    for k in ("cartography_sound", "selfA_noop", "selfB_noop", "invariants_all_cells",
              "regression_exact", "baselines_reproducible", "derived_transfer_BtoA_observed",
              "derived_transfer_AtoB_observed", "bidirectional_transfer_observed",
              "n_flip_BtoA", "n_flip_AtoB", "n_flip_bidirectional",
              "max_transfer_BtoA", "max_transfer_AtoB", "source_write_deadline_BtoA",
              "source_write_deadline_AtoB"):
        A(f"| {k} | {v[k]} |")

    A("\n## Strongest B→A sites (derived-answer transfer)\n")
    A("| L | P | token | role | transfer | frac | argmax→target | p_ansB | align | dist→B |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:12]:
        A(f"| {r['L']} | {r['P']} | `{frag[r['P']].strip()}` | {role[r['P']]} | {r['semantic_transfer']:.2f} | "
          f"{r['transfer_fraction']:.2f} | {r['argmax_is_target']} | {r['p_ansB']:.3f} | "
          f"{r.get('prop_cos_final', float('nan')):.3f} | {r.get('prop_distimprove_final', float('nan')):+.2f} |")

    A("\n## Strongest NON-source B→A sites (migration probe)\n")
    A("| L | P | token | role | transfer | argmax→target |")
    A("|---|---|---|---|---|---|")
    for r in nonsrc[:10]:
        A(f"| {r['L']} | {r['P']} | `{frag[r['P']].strip()}` | {role[r['P']]} | {r['semantic_transfer']:.2f} | {r['argmax_is_target']} |")

    A("\n## Phase 2A vs 2B comparison\n")
    A("```json")
    A(json.dumps(cmp, indent=2))
    A("```")

    A("\n## Report questions\n")
    A(_answer_questions(summary, meta, cmp, nonsrc, role))

    A("\n## Files\n```")
    A("baselines/ natural_delta/ patch_map/cells.jsonl propagation/ selected_raw/ heatmaps/")
    A("task_selection.json token_map_A.json token_map_B.json phase2b_summary.json metrics.json")
    A("```")
    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe cartography_infer.py --pilot")
    A(".venv\\Scripts\\python.exe cartography_infer.py")
    A("```")
    A("\n**STRICT STOP POINT.** Mapping pass only. No codec / adapter / probe training, "
      "no PCDC, no head-level decomposition, no weight changes.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def _answer_questions(summary, meta, cmp, nonsrc, role):
    v = summary["verdict"]
    q = []
    q.append(f"1. **Derived (non-copy) transfer?** {'Yes' if v['derived_transfer_BtoA_observed'] else 'No'} "
             f"— {v['n_flip_BtoA']} B→A sites flip to the *derived* answer (never in the prompt); "
             f"max transfer {v['max_transfer_BtoA']:.2f} of S_natural {v['S_natural']:.2f}.")
    q.append(f"2. **Bidirectional?** {'Yes' if v['bidirectional_transfer_observed'] else 'No'} "
             f"— {v['n_flip_bidirectional']} sites flip in BOTH directions (A→B: {v['n_flip_AtoB']} flips).")
    q.append(f"3. **Where is the source fact writable?** source position {meta['pair']['P_source']}; "
             f"B→A flips through layer {v['source_write_deadline_BtoA']}, A→B through {v['source_write_deadline_AtoB']}.")
    q.append(f"4. **Source write deadline:** L{v['source_write_deadline_BtoA']} (B→A).")
    ns = summary["verdict"]["strongest_nonsource_site"]
    q.append(f"5. **Does influence migrate off the source token?** strongest non-source site: "
             f"{ns} — {'yes, non-source sites carry the derived answer' if ns and ns['argmax_is_target'] else 'weak/none'}.")
    q.append("6. **Which later positions become causal?** see ranked_nonsource_BtoA.csv and the "
             "strongest-position-vs-layer plot; top non-source roles: "
             + ", ".join(sorted({role[r['P']] for r in nonsrc[:8]})) + ".")
    q.append("7. **Separate derived representation?** " +
             ("evidence for it if non-source sites (esp. operand/question/final) transfer the answer after the "
              "source deadline; see migration plot." ))
    q.append("8. **Derived-state commitment region?** operationally, the depth beyond which no site transfers "
             "the answer; reported in the summary if the map supports a boundary.")
    q.append(f"9. **Downstream trajectory alignment?** top B→A sites align with natural A→B "
             f"(prop_cos_final in the table); distance-to-B improves where positive.")
    q.append("10. **Vs Phase 2A:** see comparison JSON — deadlines, non-source flip counts, random washout bands.")
    q.append("11. **Most interesting sites:** top bidirectional + strongest non-source (ranked CSVs).")
    q.append("12. **Ambiguous:** anything relying on small transfer margins / single-token arithmetic idiosyncrasies.")
    q.append("13. **Surprises:** noted inline in the summary verdict and comparison.")
    return "\n".join(q)


if __name__ == "__main__":
    main()
