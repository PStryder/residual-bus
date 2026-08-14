"""
cartography.py -- Phase 2A: Residual Causal Cartography.

Maps, for a frozen local LLM, what happens downstream when we intervene at a
specific (layer, token-position) site of the residual stream, and whether an
activation patch carries a controlled semantic distinction from prompt B into
prompt A.

Parts (see the Phase 2A brief):
  1. Full baseline resid_post capture (all layers, all positions), A and B.
  2. Token position map (with A/B diff + roles).
  3. Perturbation map: identity / ablation / norm-relative random (a=0.3,1.0)
     over layer x position, with output + downstream-propagation metrics.
  4. Controlled minimal semantic pair (token-aligned, one-fact difference).
  5. Activation patching map: H_A[L,P] <- H_B[L,P] over layer x position,
     plus A<-A (no-op) and norm-matched random controls.
  6. Propagation toward the natural B state (alignment to D_natural = H_B-H_A).
  7. Atlas: heatmaps (PNG) + CSV + ranked causal-site table.

Controls / invariants enforced: identical baseline repeat (bitwise), A<-A patch
no-op, alpha=0 / identity no-op, hooks removed after every run, pristine-baseline
regression after the sweeps, A/B token alignment asserted, frozen model, no grad,
no KV cache, only activation state transferred (never prompt-B text into A).

Resumable: per-cell results append to JSONL; completed (op,L,P) cells are skipped
on restart. Baselines cached to disk.

Run:
  python cartography.py --pilot     # pair search + baselines + timing estimate, then stop
  python cartography.py             # full sweep (resumable) + atlas + report

STRICT STOP after cartography + patching analysis. No codec, no PCDC, no weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import model as M
import intervention as IV
import carto_lib as CL

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "phase2a"
SYSTEM = "You are a helpful assistant."
CARTO_SEED = 12345

# Minimal-pair template. Nonsense entity ("dax") so this tests token routing,
# not Qwen world knowledge. The fact VALUE is the only thing that varies.
CANDIDATE_VALUES = [
    "blue", "green", "red", "black", "white", "brown", "gray", "grey",
    "pink", "purple", "orange", "yellow", "silver", "gold", "cyan", "tan",
]

def user_msg(value: str) -> str:
    return (f"Fact: the dax is {value}. Question: what color is the dax? "
            f"Reply with exactly one lowercase word.")


# ---------------------------------------------------------------------------
# resumable jsonl helpers
# ---------------------------------------------------------------------------

def load_done(path: Path):
    done, rows = set(), []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done.add((r["op"], r["L"], r["P"]))
    return done, rows


def append_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Part 4: controlled minimal semantic pair
# ---------------------------------------------------------------------------

def find_semantic_pair(lm: M.LoadedModel) -> dict:
    """Search CANDIDATE_VALUES for a token-aligned pair (A,B) where:
      * seq lengths are equal and differ in exactly ONE position,
      * the differing token strips to the respective value (value = 1 token),
      * baseline argmax for A strips to vA and for B strips to vB (model parrots),
      * the two answer tokens are distinct.
    Fails loudly if no pair qualifies."""
    tok = lm.tokenizer
    cands = []
    for v in CANDIDATE_VALUES:
        ids = M.build_inputs(lm, user_msg(v), SYSTEM)   # [1, seq]
        logits = M.forward_logits(lm, ids)[0]           # [vocab]
        ans_id = int(torch.argmax(logits).item())
        ans_tok = tok.decode([ans_id]).strip().lower()
        cands.append({"v": v, "ids": ids, "seq": ids.shape[1],
                      "ans_id": ans_id, "ans_tok": ans_tok, "parrots": ans_tok == v})

    parroting = [c for c in cands if c["parrots"]]
    for i in range(len(parroting)):
        for j in range(len(parroting)):
            if i == j:
                continue
            a, b = parroting[i], parroting[j]
            if a["seq"] != b["seq"]:
                continue
            ia, ib = a["ids"][0], b["ids"][0]
            diff = (ia != ib).nonzero(as_tuple=True)[0].tolist()
            if len(diff) != 1:
                continue
            p = diff[0]
            va_tok = tok.decode([int(ia[p])]).strip().lower()
            vb_tok = tok.decode([int(ib[p])]).strip().lower()
            if va_tok != a["v"] or vb_tok != b["v"]:
                continue
            if a["ans_id"] == b["ans_id"]:
                continue
            return {
                "vA": a["v"], "vB": b["v"],
                "ids_A": a["ids"], "ids_B": b["ids"],
                "seq": a["seq"], "diff_pos": p,
                "token_A": a["ans_id"], "token_B": b["ans_id"],
                "answer_tok_A": tok.decode([a["ans_id"]]),
                "answer_tok_B": tok.decode([b["ans_id"]]),
            }
    raise RuntimeError(
        "No token-aligned parroting pair found. Candidates that parroted: "
        + ", ".join(c["v"] for c in parroting)
        + ". Widen CANDIDATE_VALUES or adjust the template.")


# ---------------------------------------------------------------------------
# Part 2: token position map
# ---------------------------------------------------------------------------

def build_token_map(lm, pair, H_A, H_B) -> dict:
    tok = lm.tokenizer
    ia, ib = pair["ids_A"][0].tolist(), pair["ids_B"][0].tolist()
    seq = pair["seq"]
    normA = H_A.norm(dim=-1)  # [n,seq]
    normB = H_B.norm(dim=-1)
    rows = []
    for p in range(seq):
        same = ia[p] == ib[p]
        rows.append({
            "pos": p,
            "id_A": ia[p], "frag_A": tok.decode([ia[p]]),
            "id_B": ib[p], "frag_B": tok.decode([ib[p]]),
            "same": bool(same),
            "role": token_role(tok.decode([ia[p]]), p, seq, pair["diff_pos"]),
            "meanNorm_A": float(normA[:, p].mean().item()),
            "meanNorm_B": float(normB[:, p].mean().item()),
        })
    return {
        "prompt_A": user_msg(pair["vA"]), "prompt_B": user_msg(pair["vB"]),
        "vA": pair["vA"], "vB": pair["vB"], "seq": seq,
        "diff_pos": pair["diff_pos"],
        "answer_token_A": {"id": pair["token_A"], "tok": pair["answer_tok_A"]},
        "answer_token_B": {"id": pair["token_B"], "tok": pair["answer_tok_B"]},
        "tokens": rows,
    }


def token_role(frag: str, pos: int, seq: int, diff_pos: int) -> str:
    f = frag.strip()
    if pos == diff_pos:
        return "fact_value"
    if "<|im_start|>" in frag or "<|im_end|>" in frag or f in ("system", "user", "assistant"):
        return "chat_template"
    if pos >= seq - 3:
        return "assistant_prefix"
    low = f.lower()
    if low in ("dax",):
        return "entity"
    if low in ("color", "colour"):
        return "relation"
    if "?" in frag:
        return "question_end"
    return "content"


# ---------------------------------------------------------------------------
# op builders
# ---------------------------------------------------------------------------

def perturb_op(name, L, P, H_A, hidden, device, dtype):
    if name == "identity":
        return IV.op_identity()
    if name == "ablation":
        return IV.op_zero()
    if name.startswith("rand_a"):
        alpha = float(name.split("rand_a")[1])
        seed = CARTO_SEED ^ (L * 10007 + P * 131 + int(alpha * 1000) * 7)
        d = CL.random_direction(hidden, seed, device, dtype)
        return IV.op_add_scaled_direction(d, alpha)
    raise ValueError(name)


def patch_op(name, L, P, H_A, H_B, hidden, device, dtype):
    h_A = H_A[L, P]
    h_B = H_B[L, P]
    if name == "selfA":                      # A<-A : must be a no-op
        return IV.op_replace(h_A)
    if name == "patchB":                     # A<-B : the semantic patch
        return IV.op_replace(h_B)
    if name == "randnorm":                   # A<- random vector, ||=||h_B||
        seed = CARTO_SEED ^ (L * 911 + P * 2003 + 555)
        d = CL.random_direction(hidden, seed, device, dtype)
        return IV.op_replace(d * h_B.norm())
    raise ValueError(name)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

PERTURB_OPS = ["identity", "ablation", "rand_a0.3", "rand_a1.0"]
PATCH_OPS = ["selfA", "patchB", "randnorm"]


def perturb_cell(lm, ids_A, base_logits_A, H_A, L, P, opname):
    op = perturb_op(opname, L, P, H_A, lm.hidden_size, lm.device, lm.dtype)
    res = CL.run_and_capture_all(lm, ids_A, write=(L, P, op))
    lm_m = CL.logit_metrics(base_logits_A, res.logits, lm.tokenizer)
    inv = CL.causal_invariants(H_A, res.H, L, P)
    prop = CL.propagation_perturb(H_A, res.H, L, P)
    rec = {"op": opname, "L": L, "P": P,
           "norm_before": res.norm_before, "norm_after": res.norm_after}
    rec.update(lm_m)
    rec.update({"inv_" + k: v for k, v in inv.items()})
    rec.update(prop)
    return rec


def patch_cell(lm, ids_A, base_logits_A, base_logits_B, H_A, H_B, C_A, L, P, opname):
    op = patch_op(opname, L, P, H_A, H_B, lm.hidden_size, lm.device, lm.dtype)
    res = CL.run_and_capture_all(lm, ids_A, write=(L, P, op))
    lm_m = CL.logit_metrics(base_logits_A, res.logits, lm.tokenizer)
    ac = CL.answer_contrast(res.logits, TOKEN_A, TOKEN_B)
    inv = CL.causal_invariants(H_A, res.H, L, P)
    psem = CL.propagation_semantic(H_A, H_B, res.H, L, P)
    rec = {"op": opname, "L": L, "P": P,
           "norm_before": res.norm_before, "norm_after": res.norm_after,
           "C": ac["C"], "semantic_transfer": ac["C"] - C_A,
           "p_A": ac["p_A"], "p_B": ac["p_B"], "logp_A": ac["logp_A"], "logp_B": ac["logp_B"],
           "argmax_is_B": lm_m["top1_id"] == TOKEN_B,
           "kl_from_A": lm_m["kl_from_control"],
           "kl_from_B": CL.kl_div(base_logits_B, res.logits)}
    rec.update({k: lm_m[k] for k in ("max_abs_logit_diff", "l2_logit_diff", "top1_id",
                                     "top1_tok", "top1_changed", "topk_overlap", "entropy")})
    rec.update({"inv_" + k: v for k, v in inv.items()})
    rec.update(psem)
    return rec


# module-level answer token ids (set in main once the pair is known)
TOKEN_A = None
TOKEN_B = None


# ---------------------------------------------------------------------------
# atlas: heatmaps + csv + ranked table
# ---------------------------------------------------------------------------

def rows_to_matrix(rows, op, field, n_layers, seq, default=np.nan):
    M2 = np.full((n_layers, seq), default, dtype=float)
    for r in rows:
        if r["op"] == op:
            M2[r["L"], r["P"]] = float(r.get(field, default))
    return M2


def save_heatmap(mat, title, xlabel, ylabel, path, token_frags=None, cmap="viridis", center0=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] * 0.35), max(4, mat.shape[0] * 0.22)))
    kw = {}
    if center0:
        vmax = np.nanmax(np.abs(mat)) or 1.0
        kw = {"vmin": -vmax, "vmax": vmax, "cmap": "coolwarm"}
    else:
        kw = {"cmap": cmap}
    im = ax.imshow(mat, aspect="auto", origin="lower", **kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if token_frags is not None:
        ax.set_xticks(range(len(token_frags)))
        ax.set_xticks(range(len(token_frags)))
        ax.set_xticklabels([t.replace("\n", "\\n") for t in token_frags], rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="pair search + baselines + timing estimate, then stop")
    ap.add_argument("--topk", type=int, default=8, help="raw traces for top-K sites")
    args = ap.parse_args()

    global TOKEN_A, TOKEN_B
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline").mkdir(exist_ok=True)
    t0 = time.time()

    lm = M.load()
    n = lm.num_layers
    print(f"[load] {lm.model_id}: {n} layers, hidden={lm.hidden_size}, dtype={lm.dtype}")

    # ---- Part 4: semantic pair ----
    pair = find_semantic_pair(lm)
    TOKEN_A, TOKEN_B = pair["token_A"], pair["token_B"]
    ids_A, ids_B = pair["ids_A"], pair["ids_B"]
    seq = pair["seq"]
    print(f"[pair] A: dax is {pair['vA']!r} -> {pair['answer_tok_A']!r} (id {TOKEN_A}) | "
          f"B: {pair['vB']!r} -> {pair['answer_tok_B']!r} (id {TOKEN_B}) | "
          f"seq={seq} diff_pos={pair['diff_pos']}")

    # ---- Part 1: baselines (full capture) ----
    baseA = CL.run_and_capture_all(lm, ids_A)
    baseB = CL.run_and_capture_all(lm, ids_B)
    H_A, H_B = baseA.H, baseB.H                 # [n,seq,hidden] on gpu
    base_logits_A, base_logits_B = baseA.logits, baseB.logits
    C_A = CL.answer_contrast(base_logits_A, TOKEN_A, TOKEN_B)["C"]
    C_B = CL.answer_contrast(base_logits_B, TOKEN_A, TOKEN_B)["C"]
    print(f"[baseline] C_A(logit_B - logit_A)={C_A:.3f}  C_B={C_B:.3f}  "
          f"(natural A->B contrast swing = {C_B - C_A:.3f})")

    torch.save(H_A.cpu(), OUT / "baseline" / "H_A.pt")
    torch.save(H_B.cpu(), OUT / "baseline" / "H_B.pt")
    torch.save(base_logits_A.cpu(), OUT / "baseline" / "logits_A.pt")
    torch.save(base_logits_B.cpu(), OUT / "baseline" / "logits_B.pt")

    # ---- Part 2: token map ----
    token_map = build_token_map(lm, pair, H_A, H_B)
    (OUT / "token_map.json").write_text(json.dumps(token_map, indent=2))
    frags = [r["frag_A"] for r in token_map["tokens"]]
    print("[token_map] positions:")
    for r in token_map["tokens"]:
        star = " *DIFF*" if not r["same"] else ""
        print(f"   {r['pos']:2d} {r['role']:16s} A={r['frag_A']!r}{star}")

    # ---- timing estimate (pilot) ----
    n_perturb = n * seq * len(PERTURB_OPS)
    n_patch = n * seq * len(PATCH_OPS)
    ncells_time = 6
    t = time.time()
    for k in range(ncells_time):
        perturb_cell(lm, ids_A, base_logits_A, H_A, k % n, seq - 1, "rand_a1.0")
    per_perturb = (time.time() - t) / ncells_time
    t = time.time()
    for k in range(ncells_time):
        patch_cell(lm, ids_A, base_logits_A, base_logits_B, H_A, H_B, C_A, k % n, seq - 1, "patchB")
    per_patch = (time.time() - t) / ncells_time
    est = n_perturb * per_perturb + n_patch * per_patch
    print(f"[estimate] perturb {n_perturb} cells @ {per_perturb*1000:.0f}ms + "
          f"patch {n_patch} cells @ {per_patch*1000:.0f}ms  => ~{est/60:.1f} min total")

    meta = {
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "model_id": lm.model_id, "dtype": str(lm.dtype)},
        "n_layers": n, "hidden": lm.hidden_size, "seq": seq,
        "pair": {"vA": pair["vA"], "vB": pair["vB"], "diff_pos": pair["diff_pos"],
                 "token_A": TOKEN_A, "token_B": TOKEN_B,
                 "answer_tok_A": pair["answer_tok_A"], "answer_tok_B": pair["answer_tok_B"]},
        "C_A": C_A, "C_B": C_B,
        "n_perturb_cells": n_perturb, "n_patch_cells": n_patch,
        "per_perturb_s": per_perturb, "per_patch_s": per_patch, "est_minutes": est / 60,
        "layer_convention": "H[L] = resid_post of block L = HF hidden_states[L+1]",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    if args.pilot:
        print(f"[pilot] done in {time.time()-t0:.1f}s. Re-run without --pilot for the full sweep.")
        return

    # ---- Part 3: perturbation sweep (resumable) ----
    pmap = OUT / "perturbation_map" / "cells.jsonl"
    done, prows = load_done(pmap)
    print(f"[perturb] resuming: {len(done)}/{n_perturb} cells already done")
    t = time.time()
    for L in range(n):
        for P in range(seq):
            for opname in PERTURB_OPS:
                if (opname, L, P) in done:
                    continue
                rec = perturb_cell(lm, ids_A, base_logits_A, H_A, L, P, opname)
                append_jsonl(pmap, rec)
                prows.append(rec)
        print(f"[perturb] layer {L} done ({time.time()-t:.0f}s elapsed)")

    # ---- Parts 5/6: patch sweep (resumable) ----
    qmap = OUT / "patch_map" / "cells.jsonl"
    doneq, qrows = load_done(qmap)
    print(f"[patch] resuming: {len(doneq)}/{n_patch} cells already done")
    t = time.time()
    for L in range(n):
        for P in range(seq):
            for opname in PATCH_OPS:
                if (opname, L, P) in doneq:
                    continue
                rec = patch_cell(lm, ids_A, base_logits_A, base_logits_B, H_A, H_B, C_A, L, P, opname)
                append_jsonl(qmap, rec)
                qrows.append(rec)
        print(f"[patch] layer {L} done ({time.time()-t:.0f}s elapsed)")

    # ---- regression: pristine baseline must still reproduce ----
    reA = CL.run_and_capture_all(lm, ids_A)
    regr = {
        "H_exact": bool(torch.equal(reA.H, H_A)),
        "H_max_abs_diff": float((reA.H - H_A).abs().max().item()),
        "logits_exact": bool(torch.equal(reA.logits, base_logits_A)),
        "logits_max_abs_diff": float((reA.logits - base_logits_A).abs().max().item()),
    }
    print(f"[regression] pristine baseline reproduces: {regr}")

    # ---- Part 6/7: top-site propagation traces + raw ----
    (OUT / "propagation").mkdir(exist_ok=True)
    (OUT / "selected_raw").mkdir(exist_ok=True)
    patchB_rows = [r for r in qrows if r["op"] == "patchB"]
    top_sem = sorted(patchB_rows, key=lambda r: r["semantic_transfer"], reverse=True)[:args.topk]
    top_disrupt = sorted(patchB_rows, key=lambda r: r["kl_from_A"], reverse=True)[:3]
    selected = {("patchB", r["L"], r["P"]) for r in top_sem} | {("patchB", r["L"], r["P"]) for r in top_disrupt}
    selected |= {("randnorm", top_sem[0]["L"], top_sem[0]["P"])} if top_sem else set()
    traces = []
    for (opn, L, P) in sorted(selected):
        op = patch_op(opn, L, P, H_A, H_B, lm.hidden_size, lm.device, lm.dtype)
        res = CL.run_and_capture_all(lm, ids_A, write=(L, P, op))
        tr = CL.propagation_trace_semantic(H_A, H_B, res.H, P)
        tr.update({"op": opn, "L": L, "P": P})
        traces.append(tr)
        torch.save(res.H.cpu(), OUT / "selected_raw" / f"Hprime_{opn}_L{L}_P{P}.pt")
    (OUT / "propagation" / "traces.json").write_text(json.dumps(traces, indent=2))

    # ---- atlas heatmaps + csv ----
    hm = OUT / "heatmaps"
    hm.mkdir(exist_ok=True)
    def HM(op, field, title, **kw):
        mat = rows_to_matrix(prows + qrows, op, field, n, seq)
        np.savetxt(hm / f"{op}_{field}.csv", mat, delimiter=",")
        save_heatmap(mat, title, "token position", "layer (resid_post)", hm / f"{op}_{field}.png",
                     token_frags=frags, **kw)
    HM("ablation", "kl_from_control", "Ablation  KL(A || ablated)")
    HM("rand_a1.0", "kl_from_control", "Random a=1.0  KL(A || perturbed)")
    HM("rand_a1.0", "top1_changed", "Random a=1.0  top-1 flip")
    HM("patchB", "semantic_transfer", "B->A patch  semantic_transfer", center0=True)
    HM("patchB", "kl_from_A", "B->A patch  KL(A || patched)")
    HM("patchB", "mean_cos_finalpos", "B->A patch  downstream alignment to natural A->B")
    HM("patchB", "argmax_is_B", "B->A patch  argmax becomes token_B")

    # ---- ranked causal-site table ----
    role_by_pos = {r["pos"]: r["role"] for r in token_map["tokens"]}
    frag_by_pos = {r["pos"]: r["frag_A"] for r in token_map["tokens"]}
    ranked = sorted(patchB_rows, key=lambda r: r["semantic_transfer"], reverse=True)
    for r in ranked:
        r["role"] = role_by_pos.get(r["P"], "?")
        r["frag"] = frag_by_pos.get(r["P"], "?")
    write_csv(ranked, OUT / "ranked_semantic_sites.csv",
              ["L", "P", "frag", "role", "semantic_transfer", "kl_from_A", "kl_from_B",
               "p_B", "p_A", "argmax_is_B", "mean_cos_finalpos", "mean_distimprove_finalpos"])

    # ---- summary + report ----
    summary = build_summary(meta, prows, qrows, regr, ranked, token_map)
    (OUT / "cartography_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(OUT / "CARTOGRAPHY_RESULTS.md", summary, meta, token_map, ranked)
    print(f"[done] total {time.time()-t0:.0f}s. Outputs in {OUT}")
    print(f"[verdict] {json.dumps(summary['verdict'], indent=2)}")


def build_summary(meta, prows, qrows, regr, ranked, token_map):
    def rows(op, src):
        return [r for r in src if r["op"] == op]

    ident = rows("identity", prows)
    selfA = rows("selfA", qrows)
    patchB = rows("patchB", qrows)
    randnorm = rows("randnorm", qrows)
    abl = rows("ablation", prows)
    r10 = rows("rand_a1.0", prows)

    # controls
    ident_max_kl = max((r["kl_from_control"] for r in ident), default=0.0)
    ident_inv = all(r["inv_lower_layers_zero"] and r["inv_earlier_positions_zero"] for r in ident)
    ident_delta0 = all(r["rel_delta_site_last_layer"] == 0.0 for r in ident) if ident else False
    selfA_max_kl = max((r["kl_from_A"] for r in selfA), default=0.0)
    causal_inv_all = all(r["inv_earlier_positions_zero"] for r in (prows + qrows))
    lower_inv_all = all(r["inv_lower_layers_zero"] for r in (prows + qrows))

    # best semantic sites
    best = ranked[0] if ranked else None
    # sites where argmax actually becomes B
    flipped_to_B = [r for r in patchB if r.get("argmax_is_B")]
    # generic disruption reference (randnorm) vs semantic (patchB) at matched cells
    def by_cell(lst):
        return {(r["L"], r["P"]): r for r in lst}
    pB, rN = by_cell(patchB), by_cell(randnorm)

    verdict = {
        # cartography soundness
        "identity_is_noop": ident_max_kl < 1e-9 and ident_delta0,
        "selfA_patch_is_noop": selfA_max_kl < 1e-9,
        "causal_earlier_positions_invariant": causal_inv_all,
        "lower_layers_invariant": lower_inv_all,
        "pristine_baseline_regression_exact": regr["H_exact"] and regr["logits_exact"],
        "sites_have_differentiated_effects": (
            float(np.nanstd([r["kl_from_control"] for r in abl])) > 0
            and float(np.nanstd([r["kl_from_control"] for r in r10])) > 0),
        # semantic transfer
        "semantic_transfer_observed": bool(best and best["semantic_transfer"] > 0
                                           and any(r["argmax_is_B"] for r in patchB)),
        "n_sites_argmax_flips_to_B": len(flipped_to_B),
        "max_semantic_transfer": (best["semantic_transfer"] if best else None),
        "natural_contrast_swing": meta["C_B"] - meta["C_A"],
    }
    verdict["cartography_sound"] = bool(
        verdict["identity_is_noop"] and verdict["selfA_patch_is_noop"]
        and verdict["causal_earlier_positions_invariant"] and verdict["lower_layers_invariant"]
        and verdict["pristine_baseline_regression_exact"]
        and verdict["sites_have_differentiated_effects"])

    return {
        "verdict": verdict,
        "regression": regr,
        "controls": {"identity_max_kl": ident_max_kl, "selfA_max_kl": selfA_max_kl,
                     "identity_invariants_ok": ident_inv},
        "top_semantic_sites": [
            {k: r[k] for k in ("L", "P", "semantic_transfer", "kl_from_A", "kl_from_B",
                               "p_B", "p_A", "argmax_is_B", "mean_cos_finalpos",
                               "mean_distimprove_finalpos")}
            for r in ranked[:15]],
        "argmax_flip_sites": [{"L": r["L"], "P": r["P"], "p_B": r["p_B"],
                               "semantic_transfer": r["semantic_transfer"]}
                              for r in flipped_to_B],
    }


def write_report(path, summary, meta, token_map, ranked):
    v = summary["verdict"]
    lines = []
    A = lines.append
    A("# Phase 2A — Residual Causal Cartography (Results)\n")
    A(f"**Model:** {meta['env']['model_id']} · {meta['n_layers']} layers · hidden {meta['hidden']} · "
      f"FP32 · eager · {meta['env']['gpu']}\n")
    A(f"**Layer convention:** `{meta['layer_convention']}`\n")
    A(f"**Semantic pair:** A = *dax is {meta['pair']['vA']}* → answer `{meta['pair']['answer_tok_A']}` "
      f"(id {meta['pair']['token_A']}); B = *dax is {meta['pair']['vB']}* → `{meta['pair']['answer_tok_B']}` "
      f"(id {meta['pair']['token_B']}). One-token difference at position {meta['pair']['diff_pos']}. "
      f"seq={meta['seq']}.\n")
    A(f"Natural answer-contrast swing C_B−C_A = **{summary['verdict']['natural_contrast_swing']:.2f}** "
      f"(C = logit(token_B) − logit(token_A)).\n")

    A("\n## Verdict\n")
    A("| check | result |")
    A("|---|---|")
    for k in ("cartography_sound", "identity_is_noop", "selfA_patch_is_noop",
              "causal_earlier_positions_invariant", "lower_layers_invariant",
              "pristine_baseline_regression_exact", "sites_have_differentiated_effects",
              "semantic_transfer_observed", "n_sites_argmax_flips_to_B", "max_semantic_transfer"):
        A(f"| {k} | {v[k]} |")

    A("\n## Strongest semantic patch sites (B→A)\n")
    A("| rank | layer | pos | token | role | semantic_transfer | kl_from_A | kl_from_B | p_B | argmax→B | align(nat) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked[:15]):
        A(f"| {i+1} | {r['L']} | {r['P']} | `{r['frag'].strip()}` | {r['role']} | "
          f"{r['semantic_transfer']:.2f} | {r['kl_from_A']:.2f} | {r['kl_from_B']:.2f} | "
          f"{r['p_B']:.3f} | {r['argmax_is_B']} | {r['mean_cos_finalpos']:.3f} |")

    A("\n## Files\n")
    A("```")
    A("baseline/H_A.pt H_B.pt logits_A.pt logits_B.pt   # full resid_post stacks + logits")
    A("token_map.json                                   # position table + roles + A/B diff")
    A("perturbation_map/cells.jsonl                     # identity/ablation/random per (L,P)")
    A("patch_map/cells.jsonl                            # selfA/patchB/randnorm per (L,P)")
    A("propagation/traces.json  selected_raw/*.pt       # per-layer traces + raw H' for top sites")
    A("heatmaps/*.png *.csv                             # the atlas")
    A("ranked_semantic_sites.csv  cartography_summary.json")
    A("```")
    A("\n## Reproduce\n")
    A("```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe cartography.py --pilot   # estimate")
    A(".venv\\Scripts\\python.exe cartography.py           # full, resumable")
    A("```")
    A("\n**STOP POINT.** Cartography + patching only. No codec / PCDC / weight changes.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
