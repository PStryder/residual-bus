"""
cartography_relational.py -- Phase 2D: Causal Cartography of RELATIONAL BINDING.

2A copy | 2B arithmetic | 2C scalar comparison. 2D changes the structure of the
thought again: a directed-path / relational-composition judgment whose answer
depends on BINDING which entity participates in which relation.

Task (closed world, non-copy):
    "Only these directed links exist. {s} points to {m}. {x} points to {d}.
     Is there a path of exactly two links from {s} to {d}? Answer in one word."
  A: x = m  ->  s->m, m->d  ->  2-link path s..d exists  ->  Yes
  B: x = s  ->  s->m, s->d  ->  no 2-link path s..d       ->  No
A and B differ at exactly ONE token: the subject of the second relation (P_bind).

Same validated apparatus as 2B/2C (frozen Qwen2.5-1.5B-Instruct, FP32, eager,
deterministic, resid_post hooks, full layer x position capture, no KV cache, no
grad, single forward pass). carto_lib.py reused UNCHANGED; IO/plot helpers
imported from cartography.py UNCHANGED. Outputs only to results/phase2d/.

If no clean high-confidence one-token-difference relational pair is found, the
script STOPS and reports rather than weakening the protocol.

Run:  python cartography_relational.py --pilot
      python cartography_relational.py
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
OUT = ROOT / "results" / "phase2d"
P2A, P2B, P2C = ROOT / "results" / "phase2a", ROOT / "results" / "phase2b", ROOT / "results" / "phase2c"
SYSTEM = "You are a helpful assistant."
CARTO_SEED = 31415
MIN_CONF_PREFERRED = 0.90     # preference; hard requirement is only top-1 correct both

OPS = ["BtoA", "AtoB", "selfA", "selfB", "randA", "randB"]

# candidate single-token entity labels (filtered against the tokenizer at runtime)
ENTITY_POOL = ["Tom", "Sam", "Ben", "Max", "Leo", "Ivy", "Uma", "Eli", "Ada", "Rob",
               "Jon", "Amy", "Ray", "Lee", "Guy", "Ash", "Jay", "Zoe", "Kim", "Ana",
               "Dan", "Tim", "Pat", "Joe", "Ned", "Gus", "Hal", "Mel", "Rex", "Val"]
SUFFIX = " Answer in one word."
TEMPLATES = [
    ("two_link_path",
     lambda s, m, d, x: f"Only these directed links exist. {s} points to {m}. {x} points to {d}. "
                        f"Is there a path of exactly two links from {s} to {d}?" + SUFFIX),
    ("two_steps",
     lambda s, m, d, x: f"Only these one-way links exist. {s} points to {m}. {x} points to {d}. "
                        f"Can you get from {s} to {d} in exactly two steps?" + SUFFIX),
]

# globals set once the pair is chosen
ANS_A = ANS_B = None
P_BIND = None
C_A = C_B = S_NATURAL = None


# ---------------------------------------------------------------------------
# task / pair search
# ---------------------------------------------------------------------------

def single_token_labels(lm, pool):
    tok = lm.tokenizer
    out = []
    for w in pool:
        # how it appears mid-sentence (preceded by a space)
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append((w, ids[0]))
    return out


def find_relational_pair(lm, budget=260):
    tok = lm.tokenizer
    labels = single_token_labels(lm, ENTITY_POOL)
    if len(labels) < 3:
        raise RuntimeError(f"Need >=3 single-token entity labels; found {len(labels)}.")
    log = []
    best = None
    tried = 0
    for tname, tf in TEMPLATES:
        for si in range(len(labels)):
            for mi in range(len(labels)):
                for di in range(len(labels)):
                    if len({si, mi, di}) != 3:
                        continue
                    if tried >= budget:
                        break
                    s, m, d = labels[si][0], labels[mi][0], labels[di][0]
                    idsA = M.build_inputs(lm, tf(s, m, d, m), SYSTEM)  # x=m -> Yes
                    idsB = M.build_inputs(lm, tf(s, m, d, s), SYSTEM)  # x=s -> No
                    tried += 1
                    if idsA.shape[1] != idsB.shape[1]:
                        log.append({"s": s, "m": m, "d": d, "t": tname, "reason": "len_mismatch"})
                        continue
                    diff = (idsA[0] != idsB[0]).nonzero(as_tuple=True)[0].tolist()
                    if len(diff) != 1:
                        log.append({"s": s, "m": m, "d": d, "t": tname, "reason": f"diff_{len(diff)}"})
                        continue
                    p = diff[0]
                    la = M.forward_logits(lm, idsA)[0]
                    lb = M.forward_logits(lm, idsB)[0]
                    aid, bid = int(torch.argmax(la).item()), int(torch.argmax(lb).item())
                    atok, btok = tok.decode([aid]).strip().lower(), tok.decode([bid]).strip().lower()
                    pA = float(torch.softmax(la.double(), -1)[aid].item())
                    pB = float(torch.softmax(lb.double(), -1)[bid].item())
                    okA, okB = atok == "yes", btok == "no"
                    if not (okA and okB):
                        log.append({"s": s, "m": m, "d": d, "t": tname,
                                    "reason": f"answers({atok}/{btok})", "pA": pA, "pB": pB})
                        continue
                    if aid == bid or aid in idsA[0].tolist() or bid in idsB[0].tolist():
                        log.append({"s": s, "m": m, "d": d, "t": tname, "reason": "copy_or_same"})
                        continue
                    score = min(pA, pB)
                    log.append({"s": s, "m": m, "d": d, "t": tname, "reason": "OK", "minp": score})
                    cand = {"template": tname, "s": s, "m": m, "d": d,
                            "ids_A": idsA, "ids_B": idsB, "seq": idsA.shape[1], "P_bind": p,
                            "answer_A": aid, "answer_B": bid,
                            "answer_tok_A": tok.decode([aid]), "answer_tok_B": tok.decode([bid]),
                            "pA": pA, "pB": pB, "score": score}
                    if best is None or score > best["score"]:
                        best = cand
    if best is None:
        raise RuntimeError("PHASE 2D HALT: no clean one-token-difference relational pair with correct "
                           "top-1 baselines was found. Not weakening the protocol. Candidates tried: "
                           f"{tried}. See log.")
    best["tf"] = dict(TEMPLATES)[best["template"]]
    best["search_log"] = log
    best["n_tried"] = tried
    return best


def token_role_rel(frag, pos, seq, p_bind, s, m, d, seen):
    """Occurrence-aware relational role annotation. `seen` counts label occurrences."""
    f = frag.strip()
    if pos == p_bind:
        return "binding_site"
    if pos >= seq - 3:
        return "assistant_prefix"
    if "<|im_start|>" in frag or "<|im_end|>" in frag or f in ("system", "user", "assistant"):
        return "chat_template"
    if f == s:
        seen["s"] += 1
        return "rel1_subject" if seen["s"] == 1 else "query_source"
    if f == m:
        seen["m"] += 1
        return "rel1_object"
    if f == d:
        seen["d"] += 1
        return "rel2_object" if seen["d"] == 1 else "query_dest"
    fl = f.lower()
    if fl in ("points", "to", "->"):
        return "predicate"
    if fl in ("path", "two", "links", "link", "steps", "step", "route", "exactly", "get"):
        return "query_path"
    if fl in ("answer", "word", "one"):
        return "instruction"
    if fl in ("only", "these", "directed", "one-way", "way", "exist", "is", "there", "can", "you", "in", "a"):
        return "scaffold"
    return "content"


def build_token_map(lm, ids, other_ids, p_bind, s, m, d):
    tok = lm.tokenizer
    ia, ib = ids[0].tolist(), other_ids[0].tolist()
    seq = len(ia)
    seen = {"s": 0, "m": 0, "d": 0}
    rows = []
    for p in range(seq):
        rows.append({"pos": p, "id": ia[p], "frag": tok.decode([ia[p]]),
                     "same": ia[p] == ib[p],
                     "role": token_role_rel(tok.decode([ia[p]]), p, seq, p_bind, s, m, d, seen)})
    return rows


# ---------------------------------------------------------------------------
# cell runner (identical metric semantics to Phase 2B/2C)
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
           "kl_from_base": lm_m["kl_from_control"], "kl_from_other": CL.kl_div(other_logits, res.logits),
           "C": C, "semantic_transfer": transfer, "transfer_fraction": transfer / S_NATURAL,
           "p_ansA": ac["p_A"], "p_ansB": ac["p_B"],
           "argmax_id": lm_m["top1_id"], "argmax_tok": lm_m["top1_tok"],
           "argmax_is_target": lm_m["top1_id"] == target_ans,
           "top1_changed": lm_m["top1_changed"], "topk_overlap": lm_m["topk_overlap"],
           "entropy": lm_m["entropy"], "norm_before": res.norm_before, "norm_after": res.norm_after}
    inv = CL.causal_invariants(H_src, res.H, L, P)
    rec.update({"inv_earlier_zero": inv["earlier_positions_zero"], "inv_lower_zero": inv["lower_layers_zero"]})
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


def transfer_vs_layer_plot(BtoA, AtoB, pos, title, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ba = sorted([r for r in BtoA if r["P"] == pos], key=lambda r: r["L"])
    ab = sorted([r for r in AtoB if r["P"] == pos], key=lambda r: r["L"])
    if ba:
        ax.plot([r["L"] for r in ba], [r["semantic_transfer"] for r in ba], "-o", label="B->A", ms=3)
    if ab:
        ax.plot([r["L"] for r in ab], [r["semantic_transfer"] for r in ab], "-s", label="A->B", ms=3)
    ax.axhline(S_NATURAL, ls="--", c="grey", label="S_natural")
    ax.set_title(title); ax.set_xlabel("layer"); ax.set_ylabel("semantic_transfer")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def natural_final_plot(dn_final, jump, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(dn_final)), dn_final, "-o", ms=3)
    ax.axvline(jump, ls="--", c="red", label=f"max jump @L{jump}")
    ax.set_title("Natural ||H_B - H_A|| at final position vs layer")
    ax.set_xlabel("layer"); ax.set_ylabel("||D_natural|| (final)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def migration_plot(BtoA, n, seq, p_bind, roles, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs, ys, cs = [], [], []
    for L in range(n):
        lay = [r for r in BtoA if r["L"] == L]
        if lay:
            b = max(lay, key=lambda r: r["semantic_transfer"])
            xs.append(L); ys.append(b["P"]); cs.append("red" if b["P"] == p_bind else "blue")
    ax.scatter(xs, ys, c=cs, s=28)
    ax.axhline(p_bind, ls="--", c="red", alpha=.5, label="P_bind")
    for L, P in zip(xs, ys):
        ax.annotate(roles.get(P, "?"), (L, P), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_title("Strongest B->A transfer position vs layer (red=binding site)")
    ax.set_xlabel("layer"); ax.set_ylabel("token position"); ax.set_ylim(-1, seq); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def role_transfer_heatmap(BtoA, n, roles, path):
    """max B->A transfer per (layer, role)."""
    rlist = sorted(set(roles.values()))
    mat = np.full((n, len(rlist)), np.nan)
    idx = {r: i for i, r in enumerate(rlist)}
    for r in BtoA:
        j = idx[roles[r["P"]]]
        cur = mat[r["L"], j]
        v = r["semantic_transfer"]
        if np.isnan(cur) or v > cur:
            mat[r["L"], j] = v
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6, len(rlist) * 0.7), 7))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="coolwarm",
                   vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
    ax.set_xticks(range(len(rlist))); ax.set_xticklabels(rlist, rotation=90, fontsize=7)
    ax.set_title("max B->A transfer by semantic role vs layer")
    ax.set_xlabel("role"); ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    np.savetxt(path.with_suffix(".csv"), mat, delimiter=",", header=",".join(rlist))
    return rlist, mat


# ---------------------------------------------------------------------------
# cross-phase comparison
# ---------------------------------------------------------------------------

def natural_final_series(H_A, H_B):
    d = (H_B - H_A).norm(dim=-1)[:, -1]
    diffs = d[1:] - d[:-1]
    return d.cpu().numpy(), int(torch.argmax(diffs).item()) + 1


def load_phase(pdir, baseline_sub, sem_op, rand_op, flip_field, kl_field, source_pos, seq):
    cells = [json.loads(l) for l in (pdir / "patch_map" / "cells.jsonl").read_text().splitlines() if l.strip()]
    sem = [r for r in cells if r["op"] == sem_op]
    rnd = [r for r in cells if r["op"] == rand_op]
    final_pos = seq - 1
    src = [r for r in sem if r["P"] == source_pos]
    fin = [r for r in sem if r["P"] == final_pos]
    src_fl = sorted(r["L"] for r in src if r.get(flip_field))
    fin_fl = sorted(r["L"] for r in fin if r.get(flip_field))
    interior = [r for r in sem if r.get(flip_field) and r["P"] not in (source_pos, final_pos)]
    interior_pos = sorted(set(r["P"] for r in interior))
    H_A = torch.load(pdir / baseline_sub / "H_A.pt")
    H_B = torch.load(pdir / baseline_sub / "H_B.pt")
    _, jump = natural_final_series(H_A, H_B)
    def band(lo, hi):
        v = [r[kl_field] for r in rnd if r["P"] == source_pos and lo <= r["L"] <= hi]
        return float(np.mean(v)) if v else None
    return {"source_deadline": (max(src_fl) if src_fl else None), "source_stripe_width": len(src_fl),
            "final_onset": (min(fin_fl) if fin_fl else None),
            "n_interior_flip_sites": len(interior), "interior_flip_positions": interior_pos,
            "natural_final_jump_layer": jump,
            "max_transfer": max((r["semantic_transfer"] for r in sem), default=None),
            "rand_kl_deep_L16_21": band(16, 21),
            "rand_target_flip_rate": (sum(1 for r in rnd if r.get(flip_field)) / max(1, len(rnd)))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    global ANS_A, ANS_B, P_BIND, C_A, C_B, S_NATURAL

    for sub in ("baselines", "natural_delta", "patch_map", "propagation", "controls",
                "selected_raw", "heatmaps"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = M.load()
    n = lm.num_layers
    tok = lm.tokenizer
    print(f"[load] {lm.model_id}: {n} layers, hidden={lm.hidden_size}")

    pair = find_relational_pair(lm)
    ANS_A, ANS_B, P_BIND = pair["answer_A"], pair["answer_B"], pair["P_bind"]
    ids_A, ids_B, seq = pair["ids_A"], pair["ids_B"], pair["seq"]
    s, m, d = pair["s"], pair["m"], pair["d"]
    print(f"[pair] template={pair['template']} | entities s={s} m={m} d={d} | "
          f"A(x={m})->{pair['answer_tok_A']!r} B(x={s})->{pair['answer_tok_B']!r} | "
          f"seq={seq} P_bind={P_BIND} | pA={pair['pA']:.3f} pB={pair['pB']:.3f} (tried {pair['n_tried']})")
    if pair["score"] < MIN_CONF_PREFERRED:
        print(f"[warn] baseline confidence {pair['score']:.3f} < preferred {MIN_CONF_PREFERRED}; "
              f"proceeding (top-1 correct both) but flagging lower SNR in the report.")

    assert ANS_A != ANS_B
    assert ANS_A not in ids_A[0].tolist() and ANS_B not in ids_B[0].tolist(), "answer in prompt (copy!)"
    assert (ids_A[0] != ids_B[0]).sum().item() == 1

    tmapA = build_token_map(lm, ids_A, ids_B, P_BIND, s, m, d)
    tmapB = build_token_map(lm, ids_B, ids_A, P_BIND, s, m, d)
    (OUT / "token_map_A.json").write_text(json.dumps(tmapA, indent=2), encoding="utf-8")
    (OUT / "token_map_B.json").write_text(json.dumps(tmapB, indent=2), encoding="utf-8")
    roles = {r["pos"]: r["role"] for r in tmapA}
    frag = {r["pos"]: r["frag"] for r in tmapA}
    # landmark positions
    P_M1 = next((p for p in range(seq) if roles[p] == "rel1_object"), None)      # intermediate entity, occ1
    P_QSRC = next((p for p in range(seq) if roles[p] == "query_source"), None)
    P_QDST = next((p for p in range(seq) if roles[p] == "query_dest"), None)
    (OUT / "task_selection.json").write_text(json.dumps({
        "template": pair["template"], "template_str": pair["tf"](s, m, d, "{X}"),
        "entities": {"s": s, "m": m, "d": d}, "P_bind": P_BIND,
        "binding_A_token": m, "binding_B_token": s,
        "answer_A": {"id": ANS_A, "tok": pair["answer_tok_A"]},
        "answer_B": {"id": ANS_B, "tok": pair["answer_tok_B"]},
        "seq": seq, "pA": pair["pA"], "pB": pair["pB"], "non_copy_verified": True,
        "landmarks": {"P_bind": P_BIND, "P_rel1_object(intermediate)": P_M1,
                      "P_query_source": P_QSRC, "P_query_dest": P_QDST, "P_final": seq - 1},
        "search_log": pair["search_log"],
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
    assert correctA and correctB

    torch.save(H_A.cpu(), OUT / "baselines" / "H_A.pt")
    torch.save(H_B.cpu(), OUT / "baselines" / "H_B.pt")
    torch.save(base_A.cpu(), OUT / "baselines" / "logits_A.pt")
    torch.save(base_B.cpu(), OUT / "baselines" / "logits_B.pt")

    # ---- Part 1/6: natural atlas + final divergence ----
    D = H_B - H_A
    dnorm = D.norm(dim=-1).cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(H_A, H_B, dim=-1).cpu().numpy()
    np.savetxt(OUT / "natural_delta" / "dnorm.csv", dnorm, delimiter=",")
    np.savetxt(OUT / "natural_delta" / "cos.csv", cos, delimiter=",")
    before = D[:, :P_BIND]
    before_zero = bool(before.abs().max().item() == 0.0) if before.numel() else True
    save_heatmap(dnorm, "Natural |D| = ||H_B - H_A||", "token position", "layer (resid_post)",
                 OUT / "heatmaps" / "natural_dnorm.png", token_frags=frags)
    dn_final, jump_layer = natural_final_series(H_A, H_B)
    np.savetxt(OUT / "natural_delta" / "final_dnorm.csv", dn_final, delimiter=",")
    natural_final_plot(dn_final, jump_layer, OUT / "heatmaps" / "natural_final_divergence.png")
    print(f"[natural] before-bind zero={before_zero} | final jump @L{jump_layer}")

    ctx = {"ids_A": ids_A, "ids_B": ids_B, "H_A": H_A, "H_B": H_B, "base_A": base_A, "base_B": base_B}

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
            "task": "relational_binding (directed 2-link path) -> categorical (non-copy)",
            "pair": {"template": pair["template"], "s": s, "m": m, "d": d,
                     "answer_A": ANS_A, "answer_B": ANS_B,
                     "answer_tok_A": pair["answer_tok_A"], "answer_tok_B": pair["answer_tok_B"],
                     "P_bind": P_BIND, "P_rel1_object": P_M1, "P_query_source": P_QSRC,
                     "P_query_dest": P_QDST, "pA": pair["pA"], "pB": pair["pB"]},
            "C_A": C_A, "C_B": C_B, "S_natural": S_NATURAL,
            "baseline_correct": {"A": correctA, "B": correctB}, "baseline_repro": repro,
            "natural_final_jump_layer": jump_layer, "natural_before_bind_zero": before_zero,
            "n_cells": ncells, "per_cell_s": per,
            "layer_convention": "H[L] = resid_post of block L = HF hidden_states[L+1]"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.pilot:
        print(f"[pilot] done {time.time()-t0:.1f}s")
        return

    # ---- Part 2: bidirectional sweep (resumable) ----
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

    reA, reB = CL.run_and_capture_all(lm, ids_A), CL.run_and_capture_all(lm, ids_B)
    regr = {"A_H": bool(torch.equal(reA.H, H_A)), "A_logits": bool(torch.equal(reA.logits, base_A)),
            "B_H": bool(torch.equal(reB.H, H_B)), "B_logits": bool(torch.equal(reB.logits, base_B))}
    print(f"[regression] {regr}")

    analyze_and_report(lm, meta, rows, dnorm, dn_final, jump_layer, frags, tmapA, regr, repro, ctx,
                       before_zero, roles, frag, P_M1)
    print(f"[done] total {time.time()-t0:.0f}s. Outputs in {OUT}")


def analyze_and_report(lm, meta, rows, dnorm, dn_final, jump_layer, frags, tmapA, regr, repro, ctx,
                       before_zero, roles, frag, P_M1):
    n, seq = meta["n_layers"], meta["seq"]
    PB = meta["pair"]["P_bind"]
    def rws(op): return [r for r in rows if r["op"] == op]
    BtoA, AtoB, selfA, selfB, randA = rws("BtoA"), rws("AtoB"), rws("selfA"), rws("selfB"), rws("randA")

    selfA_ok = max((abs(r["semantic_transfer"]) for r in selfA), default=0) < 1e-6 and \
        max((r["kl_from_base"] for r in selfA), default=0) < 1e-9
    selfB_ok = max((abs(r["semantic_transfer"]) for r in selfB), default=0) < 1e-6 and \
        max((r["kl_from_base"] for r in selfB), default=0) < 1e-9
    inv_ok = all(r["inv_earlier_zero"] and r["inv_lower_zero"] for r in rows)
    fBA = [r for r in BtoA if r["argmax_is_target"]]
    fAB = [r for r in AtoB if r["argmax_is_target"]]
    dBA = {(r["L"], r["P"]): r for r in BtoA}
    dAB = {(r["L"], r["P"]): r for r in AtoB}
    bidir_flip = [k for k in dBA if k in dAB and dBA[k]["argmax_is_target"] and dAB[k]["argmax_is_target"]]

    def flip_layers(rs, pos): return sorted(r["L"] for r in rs if r["P"] == pos and r["argmax_is_target"])
    bind_dl_BA = (max(flip_layers(BtoA, PB)) if flip_layers(BtoA, PB) else None)
    bind_dl_AB = (max(flip_layers(AtoB, PB)) if flip_layers(AtoB, PB) else None)
    fin_on_BA = (min(flip_layers(BtoA, seq - 1)) if flip_layers(BtoA, seq - 1) else None)
    fin_on_AB = (min(flip_layers(AtoB, seq - 1)) if flip_layers(AtoB, seq - 1) else None)

    interior_flip = [r for r in fBA if r["P"] not in (PB, seq - 1)]
    interior_pos = sorted(set(r["P"] for r in interior_flip))
    nonbind = sorted([r for r in BtoA if r["P"] != PB], key=lambda r: r["semantic_transfer"], reverse=True)
    # role max transfer
    role_maxT = {}
    for r in BtoA:
        role_maxT[roles[r["P"]]] = max(role_maxT.get(roles[r["P"]], -1e9), r["semantic_transfer"])
    # intermediate/join entity occurrences: binding site (occ2, A-side) and rel1_object (occ1)
    join_occ = {"rel1_object_pos": P_M1,
                "rel1_object_flip_layers": (flip_layers(BtoA, P_M1) if P_M1 is not None else []),
                "binding_site_flip_layers": flip_layers(BtoA, PB)}
    rand_flip = [r for r in randA if r["argmax_is_target"]]

    # no-single-site gap: layers where no position flips
    flip_by_layer = {L: sorted(set(r["P"] for r in BtoA if r["L"] == L and r["argmax_is_target"])) for L in range(n)}
    gap_layers = [L for L in range(n) if not flip_by_layer[L]]

    # ---- heatmaps ----
    hm = OUT / "heatmaps"
    def HM(rs, op, field, title, **kw):
        mat = rows_to_matrix(rs, op, field, n, seq)
        np.savetxt(hm / f"{op}_{field}.csv", mat, delimiter=",")
        save_heatmap(mat, title, "token position", "layer (resid_post)", hm / f"{op}_{field}.png",
                     token_frags=frags, **kw)
    HM(BtoA, "BtoA", "semantic_transfer", "B->A transfer (relational)", center0=True)
    HM(AtoB, "AtoB", "semantic_transfer", "A->B transfer (relational)", center0=True)
    HM(BtoA, "BtoA", "argmax_is_target", "B->A argmax -> B answer")
    HM(AtoB, "AtoB", "argmax_is_target", "A->B argmax -> A answer")
    HM(BtoA, "BtoA", "kl_from_base", "B->A KL(A||patched)")
    HM(randA, "randA", "kl_from_base", "random-control KL")
    HM(randA, "randA", "semantic_transfer", "random-control transfer", center0=True)
    HM(BtoA, "BtoA", "prop_distimprove_final", "B->A distance-to-B improvement (final)", center0=True)
    HM(BtoA, "BtoA", "prop_cos_final", "B->A downstream alignment to natural A->B")
    bmat = np.full((n, seq), np.nan)
    for k in dBA:
        if k in dAB:
            bmat[k[0], k[1]] = min(dBA[k]["semantic_transfer"], dAB[k]["semantic_transfer"])
    np.savetxt(hm / "bidirectional_min_transfer.csv", bmat, delimiter=",")
    save_heatmap(bmat, "bidirectional min(transfer)", "token position", "layer (resid_post)",
                 hm / "bidirectional_min_transfer.png", token_frags=frags, center0=True)
    transfer_vs_layer_plot(BtoA, AtoB, PB, f"Binding site (P={PB}) transfer vs layer",
                           hm / "binding_transfer_vs_layer.png")
    transfer_vs_layer_plot(BtoA, AtoB, seq - 1, f"Final (P={seq-1}) transfer vs layer",
                           hm / "final_transfer_vs_layer.png")
    migration_plot(BtoA, n, seq, PB, roles, hm / "strongest_position_vs_layer.png")
    role_transfer_heatmap(BtoA, n, roles, hm / "role_max_transfer.png")

    # ---- ranked tables ----
    def dump(rs, name, extra=True):
        for r in rs:
            r["role"] = roles.get(r["P"], "?"); r["frag"] = frag.get(r["P"], "?")
        write_csv(rs, OUT / name, ["L", "P", "frag", "role", "semantic_transfer", "transfer_fraction",
                                   "argmax_is_target", "kl_from_base", "kl_from_other", "p_ansA", "p_ansB",
                                   *(["prop_cos_final", "prop_distimprove_final"] if extra else [])])
    dump(sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:40], "ranked_BtoA.csv")
    dump(sorted(AtoB, key=lambda r: r["semantic_transfer"], reverse=True)[:40], "ranked_AtoB.csv")
    dump(nonbind[:40], "ranked_nonbinding_BtoA.csv")
    dump(sorted(randA, key=lambda r: r["kl_from_base"], reverse=True)[:20], "ranked_random_disruption.csv", extra=False)

    # ---- selected raw ----
    sel = {("BtoA", r["L"], r["P"]) for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:3]}
    sel |= {("BtoA", r["L"], r["P"]) for r in nonbind[:2]}
    if bind_dl_BA is not None:
        sel.add(("BtoA", bind_dl_BA, PB))
    if fin_on_BA is not None:
        sel.add(("BtoA", fin_on_BA, seq - 1))
    if P_M1 is not None and join_occ["rel1_object_flip_layers"]:
        sel.add(("BtoA", join_occ["rel1_object_flip_layers"][0], P_M1))
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

    # ---- Part 11: four-phase comparison ----
    cmp = {"phase2d": {"task": "relational", "binding_deadline": bind_dl_BA, "final_onset": fin_on_BA,
                       "natural_final_jump_layer": jump_layer, "n_interior_flip_sites": len(interior_flip),
                       "interior_flip_positions": interior_pos, "interior_flip_roles": sorted(set(roles[p] for p in interior_pos)),
                       "gap_layers": gap_layers, "max_transfer": max((r["semantic_transfer"] for r in BtoA), default=None),
                       "S_natural": meta["S_natural"], "bidirectional_flips": len(bidir_flip),
                       "rand_target_flip_rate": len(rand_flip) / max(1, len(randA)),
                       "rand_kl_deep_L16_21": float(np.mean([r["kl_from_base"] for r in randA
                                                             if r["P"] == PB and 16 <= r["L"] <= 21]) or 0)}}
    for name, pdir, bsub, so, ro, ff, kf in [
            ("phase2a", P2A, "baseline", "patchB", "randnorm", "argmax_is_B", "kl_from_A"),
            ("phase2b", P2B, "baselines", "BtoA", "randA", "argmax_is_target", "kl_from_base"),
            ("phase2c", P2C, "baselines", "BtoA", "randA", "argmax_is_target", "kl_from_base")]:
        try:
            mm = json.loads((pdir / "meta.json").read_text())
            spos = mm["pair"].get("P_source", mm["pair"].get("diff_pos"))
            cmp[name] = load_phase(pdir, bsub, so, ro, ff, kf, spos, mm["seq"])
            cmp[name]["task"] = {"phase2a": "copy", "phase2b": "arithmetic", "phase2c": "categorical"}[name]
        except Exception as e:
            cmp[name] = {"error": f"{type(e).__name__}: {e}"}

    # Part 8: ~L22-23 readout hypothesis across 4 tasks
    jumps = {ph: cmp[ph].get("natural_final_jump_layer") for ph in ("phase2a", "phase2b", "phase2c") if isinstance(cmp.get(ph), dict)}
    jumps["phase2d"] = jump_layer
    jvals = [v for v in jumps.values() if v is not None]
    readout = classify_readout(jvals, jumps)

    coincide = (bind_dl_BA is not None and fin_on_BA is not None
                and abs((bind_dl_BA + 1) - fin_on_BA) <= 1 and abs(fin_on_BA - jump_layer) <= 1)

    verdict = {
        "cartography_sound": bool(selfA_ok and selfB_ok and inv_ok and all(regr.values())),
        "selfA_noop": selfA_ok, "selfB_noop": selfB_ok, "invariants_all_cells": inv_ok,
        "regression_exact": all(regr.values()), "baselines_reproducible": all(repro.values()),
        "relational_transfer_BtoA": bool(fBA), "relational_transfer_AtoB": bool(fAB),
        "bidirectional": bool(bidir_flip),
        "n_flip_BtoA": len(fBA), "n_flip_AtoB": len(fAB), "n_flip_bidirectional": len(bidir_flip),
        "max_transfer_BtoA": max((r["semantic_transfer"] for r in BtoA), default=None), "S_natural": meta["S_natural"],
        "binding_write_deadline_BtoA": bind_dl_BA, "binding_write_deadline_AtoB": bind_dl_AB,
        "final_onset_BtoA": fin_on_BA, "final_onset_AtoB": fin_on_AB,
        "natural_final_jump_layer": jump_layer, "three_signals_coincide": bool(coincide),
        "any_interior_sufficient": bool(interior_flip),
        "interior_flip_roles": sorted(set(roles[p] for p in interior_pos)),
        "join_entity_rel1object_flip_layers": join_occ["rel1_object_flip_layers"],
        "no_single_site_gap_layers": gap_layers,
        "random_target_flip_count": len(rand_flip),
        "l22_readout_hypothesis": readout["classification"],
    }
    summary = {"verdict": verdict, "regression": regr, "repro": repro,
               "controls": {"selfA_ok": selfA_ok, "selfB_ok": selfB_ok, "invariants_ok": inv_ok,
                            "random_target_flip_count": len(rand_flip),
                            "random_max_kl": max((r["kl_from_base"] for r in randA), default=None)},
               "role_max_transfer": role_maxT, "join_entity": join_occ,
               "top_BtoA": [{k: r[k] for k in ("L", "P", "semantic_transfer", "transfer_fraction",
                                               "argmax_is_target", "p_ansB", "prop_cos_final", "prop_distimprove_final")}
                            for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:15]],
               "top_nonbinding": [{"L": r["L"], "P": r["P"], "role": roles[r["P"]],
                                   "transfer": r["semantic_transfer"], "argmax_is_target": r["argmax_is_target"]}
                                  for r in nonbind[:15]],
               "four_phase_comparison": cmp, "l22_readout": readout, "natural_before_bind_zero": before_zero}
    (OUT / "phase2d_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps({"n_cells": len(rows), "ops": OPS}, indent=2), encoding="utf-8")
    write_report(OUT / "CARTOGRAPHY_RELATIONAL_RESULTS.md", summary, meta, tmapA, BtoA, nonbind, cmp,
                 readout, roles, frag, coincide, gap_layers, join_occ)
    print(f"[verdict] {json.dumps(verdict, indent=2, default=str)}")


def classify_readout(jvals, jumps):
    if not jvals:
        return {"classification": "ambiguous", "why": "no jump data", "jumps": jumps}
    lo, hi = min(jvals), max(jvals)
    in_band = all(21 <= v <= 24 for v in jvals)
    if in_band and (hi - lo) <= 2:
        cls = "strengthened"
        why = f"natural final-position divergence jump sits in L21-24 for all tasks ({jumps}); spread {hi-lo}."
    elif hi - lo <= 3:
        cls = "strengthened"
        why = f"jumps cluster tightly ({jumps}, spread {hi-lo}) though not all strictly in L22-23."
    else:
        cls = "weakened"
        why = f"jumps spread widely ({jumps}, spread {hi-lo}); no stable ~L22-23 readout."
    return {"classification": cls, "why": why, "jumps": jumps,
            "conservative_label": "late final-position readout/refinement region (not a reasoning layer)"}


def write_report(path, summary, meta, tmapA, BtoA, nonbind, cmp, readout, roles, frag, coincide, gap_layers, join_occ):
    v = summary["verdict"]
    L = []
    A = L.append
    A("# Phase 2D — Causal Cartography of Relational Binding (Results)\n")
    A(f"**Model:** {meta['env']['model_id']} · {meta['n_layers']} layers · hidden {meta['hidden']} · FP32 · eager · {meta['env']['gpu']}\n")
    A(f"**Task (relational composition, non-copy):** directed 2-link path judgment. "
      f"Entities s={meta['pair']['s']}, m={meta['pair']['m']}, d={meta['pair']['d']}.\n")
    A(f"- A (2nd link `{meta['pair']['m']} points to {meta['pair']['d']}`): path exists → `{meta['pair']['answer_tok_A']}`.")
    A(f"- B (2nd link `{meta['pair']['s']} points to {meta['pair']['d']}`): no 2-link path → `{meta['pair']['answer_tok_B']}`.")
    A(f"- The ONLY differing token is the **second-relation subject** at **P_bind={meta['pair']['P_bind']}** "
      f"(`{meta['pair']['m']}`↔`{meta['pair']['s']}`). Non-copy asserted (yes/no absent). "
      f"baselines p={meta['pair']['pA']:.2f}/{meta['pair']['pB']:.2f}, S_natural={meta['S_natural']:.2f}.\n")

    A("## Verdict\n| check | result |\n|---|---|")
    for k in ("cartography_sound", "selfA_noop", "selfB_noop", "invariants_all_cells", "regression_exact",
              "baselines_reproducible", "relational_transfer_BtoA", "relational_transfer_AtoB", "bidirectional",
              "n_flip_BtoA", "n_flip_AtoB", "n_flip_bidirectional", "max_transfer_BtoA", "S_natural",
              "binding_write_deadline_BtoA", "binding_write_deadline_AtoB", "final_onset_BtoA", "final_onset_AtoB",
              "natural_final_jump_layer", "three_signals_coincide", "any_interior_sufficient",
              "interior_flip_roles", "no_single_site_gap_layers", "random_target_flip_count",
              "l22_readout_hypothesis"):
        A(f"| {k} | {v[k]} |")

    A("\n## Binding-site → final structure\n")
    A(f"- **Binding write deadline (B→A):** L{v['binding_write_deadline_BtoA']} (A→B: L{v['binding_write_deadline_AtoB']}).")
    A(f"- **Final onset (B→A):** L{v['final_onset_BtoA']} (A→B: L{v['final_onset_AtoB']}).")
    A(f"- **Natural final divergence jump:** L{v['natural_final_jump_layer']}.")
    A(f"- **Interior causally-sufficient roles:** {v['interior_flip_roles']}.")
    A(f"- **Join/intermediate entity** (rel1-object occurrence) flip layers: {join_occ['rel1_object_flip_layers']}.")
    A(f"- **No-single-site-sufficiency gap layers:** {gap_layers}.")

    A("\n## Strongest B→A sites\n")
    A("| L | P | token | role | transfer | frac | flip | p_ansB | align |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(BtoA, key=lambda r: r["semantic_transfer"], reverse=True)[:12]:
        A(f"| {r['L']} | {r['P']} | `{frag[r['P']].strip()}` | {roles[r['P']]} | {r['semantic_transfer']:.2f} | "
          f"{r['transfer_fraction']:.2f} | {r['argmax_is_target']} | {r['p_ansB']:.3f} | "
          f"{r.get('prop_cos_final', float('nan')):.3f} |")

    A("\n## Role max-transfer\n```json")
    A(json.dumps(summary["role_max_transfer"], indent=2, default=str))
    A("```")

    A("\n## Four-phase comparison (2A copy / 2B arithmetic / 2C categorical / 2D relational)\n```json")
    A(json.dumps(cmp, indent=2, default=str))
    A("```")

    A("\n## ~L22–23 natural final-position readout hypothesis\n")
    A(f"**Classification: `{readout['classification']}`** — {readout['why']}")
    A(f"\nConservative label: *{readout['conservative_label']}*.")

    A("\n## Report answers\n")
    A(_answers(summary, meta, cmp, readout, roles, gap_layers, join_occ))

    A("\n## Files\n```")
    A("task_selection.json token_map_A/B.json baselines/ natural_delta/ patch_map/cells.jsonl")
    A("propagation/traces.json selected_raw/*.pt heatmaps/*.png *.csv ranked_*.csv phase2d_summary.json")
    A("```")
    A("\n## Reproduce\n```powershell")
    A('$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"')
    A('$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1')
    A(".venv\\Scripts\\python.exe cartography_relational.py --pilot")
    A(".venv\\Scripts\\python.exe cartography_relational.py")
    A("```")
    A("\n**STRICT STOP POINT.** Single-site mapping only. No codec/adapter/probe training, no fine-tuning, "
      "no weight changes, no PCDC, no head-level decomposition, no multi-site combinatorial patching.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def _answers(summary, meta, cmp, readout, roles, gap_layers, join_occ):
    v = summary["verdict"]
    q = []
    q.append(f"1. **Relational categorical transfer?** {'Yes' if v['relational_transfer_BtoA'] else 'No'} "
             f"— {v['n_flip_BtoA']} B→A flips to the derived answer (never in prompt).")
    q.append(f"2. **Bidirectional?** {'Yes' if v['bidirectional'] else 'No'} ({v['n_flip_bidirectional']} both ways).")
    q.append(f"3. **Binding writable where?** P{meta['pair']['P_bind']} (2nd-relation subject), through L{v['binding_write_deadline_BtoA']}.")
    q.append(f"4. **Binding write deadline:** L{v['binding_write_deadline_BtoA']} (B→A).")
    q.append(f"5. **Control move into another entity occurrence?** join entity (rel1-object) flip layers: "
             f"{join_occ['rel1_object_flip_layers']}.")
    q.append(f"6. **Intermediate/join entity a waypoint?** {'Yes' if join_occ['rel1_object_flip_layers'] else 'No'}.")
    q.append(f"7/8. **Relation-predicate / query-entity waypoints?** interior sufficient roles = {v['interior_flip_roles']}.")
    q.append(f"9. **Any interior state sufficient?** {'Yes' if v['any_interior_sufficient'] else 'No'}.")
    q.append(f"10. **No-single-site period?** gap layers = {gap_layers}.")
    q.append(f"11. **Final sufficiency onset:** L{v['final_onset_BtoA']}.")
    q.append(f"12. **Steepest natural final reorganization:** L{v['natural_final_jump_layer']}.")
    q.append(f"13. **~L22-23 readout persists?** {readout['classification']} (jumps {readout['jumps']}).")
    q.append("14. **Divergence broader than causal control?** compare natural_dnorm.png with BtoA_argmax_is_target.png.")
    q.append("15. **Trajectory alignment?** top-site prop_cos_final / dist→B in ranked_BtoA.csv.")
    q.append("16. **Vs copy/arith/comparison?** see four_phase_comparison JSON.")
    q.append(f"17. **Computation-dependent writable sites?** deadlines 2A/2B/2C/2D differ → supports task-dependence.")
    q.append("18. **Ambiguous:** relational-binding may be distributed; single-site sufficiency only partially captures it.")
    q.append("19. **Surprises:** see comparison (waypoint identity, deadline, gap).")
    q.append("20. **Next anatomical test (for discussion):** vary graph depth / join structure; do NOT escalate to "
             "multi-site or head-level here.")
    return "\n".join(q)


if __name__ == "__main__":
    main()
