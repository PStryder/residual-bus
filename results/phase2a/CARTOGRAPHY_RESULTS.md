# Phase 2A — Residual Causal Cartography (Results)

**Model:** Qwen2.5-1.5B-Instruct (local) · 28 layers · hidden 1536 · FP32 · eager · RTX 4080
**Convention:** `H[L] = resid_post of decoder block L = HF hidden_states[L+1]` (hidden_states[0] = embeddings). A WriteHook at layer L edits `resid_post[L]`; blocks L+1.. reprocess it. Single forward pass, **no KV cache**, frozen model, `no_grad`.
**Cells run:** 8428 (4816 perturbation + 3612 patch), full layer×position sweep, resumable.

## Semantic pair (Part 4)

Token-aligned minimal pair, one-token fact difference (assertions passed):

- **A:** *"Fact: the dax is **blue**. …"* → answer `blue` (id 12203)
- **B:** *"Fact: the dax is **green**. …"* → answer `green` (id 13250)
- Differ only at **position 20** (` blue`/` green`); seq = 43; answer is a verbatim **copy** of the fact value.
- Baseline answer-contrast `C = logit(green) − logit(blue)`: **C_A = −14.97**, **C_B = +12.47** → natural A→B swing **27.44**. Clean, well-separated.

## Verdict

| check | result |
|---|---|
| **cartography_sound** | ✅ **true** |
| identity write is a true no-op | ✅ max KL 0.0, Δ bitwise 0 |
| A←A patch (`selfA`) is a true no-op | ✅ max \|semantic_transfer\| 0.0, max KL 0.0 |
| causal invariant: earlier positions unchanged | ✅ all 8428 cells (bitwise) |
| lower layers unchanged | ✅ all 8428 cells (bitwise) |
| pristine baseline regression after sweep | ✅ **bitwise exact** (H and logits) |
| sites have differentiated effects | ✅ |
| **semantic_transfer_observed** | ✅ **true** |
| sites where argmax flips A→`green` | **28** |
| max semantic_transfer | **27.54** (≈ the full natural swing 27.44) |

A negative semantic result would still have been valid; it was strongly positive. No thresholds were massaged — the argmax flip and p_B=1.0 are unambiguous.

## The socket (strongest causal sites)

Patching the single **fact-value token (pos 20)** with B's residual flips A's answer to `green` (**p_B = 1.000**, argmax→B) at **every layer L = 0 … 21**, with semantic_transfer ≈ 27.4 (matching, and at L8 slightly exceeding at 27.54, the full natural swing). It is **partial at L22** (p_B 0.77) and **collapses at L ≥ 23** (p_B ≈ 0). See `heatmaps/patchB_semantic_transfer.png` — one solid stripe at pos 20, layers 0–21.

| rank | layer | pos | token | role | sem_transfer | p_B | argmax→B | align(nat) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 20 | `blue` | fact_value | 27.54 | 1.000 | ✅ | 0.78 |
| 2 | 7 | 20 | `blue` | fact_value | 27.49 | 1.000 | ✅ | 0.78 |
| 3 | 4 | 20 | `blue` | fact_value | 27.48 | 1.000 | ✅ | 0.80 |
| … | 0–21 | 20 | `blue` | fact_value | 27.4–26.9 | 1.000 | ✅ | 0.73–0.94 |
| — | 27 | 42 | `\n` | assistant_prefix | 27.44 | 1.000 | ✅ | 0.00 |

Two qualitatively different sockets:

1. **Fact-value port (pos 20, layers 0–~21)** — a *wide receptive band*. Because the answer is a copy of the fact, the color identity lives in that token's residual and is read out late; patching it anywhere before the read-out reroutes the answer. **Causal deadline ≈ L22**: after the copy has happened, patching is too late.
2. **Output-substitution port (final position 42, last layers only)** — replacing the final residual at the last layer just swaps the logits directly (`align = 0.00`, no downstream layers). Causal but trivial: output substitution, not semantic routing. (This confirms the Phase 1 finding that mid-layer *final-token* replacement is inert — it only bites at the very last layer.)

## Semantic transfer vs generic disruption (the key discrimination)

At the fact-value token, `patchB` (semantic) vs `randnorm` (norm-matched random replacement, same ‖·‖):

| depth | patchB sem_transfer | randnorm sem_transfer | patchB KL(A‖·) | randnorm KL(A‖·) |
|---|---|---|---|---|
| L0–11 | ~27.4 (p_B=1.0, flips) | ~10–14 (no flip) | ~12 (stable) | 2–5 |
| L14–21 | ~27 (p_B=1.0, flips) | ~10 (no flip) | ~12 (stable) | **0.5–1.4 (decays)** |

Two discriminators, both decisive:
- **Directionality:** only `patchB` drives p_B→1.0 and flips the argmax; norm-matched random noise nudges the contrast (destroying "blue" evidence relatively favors "green") but never flips it.
- **Durability:** `patchB`'s output effect is depth-stable (KL ≈ 12 at all layers), while generic disruption **decays with depth** (KL 4.5 → 0.5) — RMSNorm and the model's robustness wash out off-manifold noise, but the semantic patch persists because it rides the model's own routing.

This is exactly the "was it disrupted vs did it move toward B" distinction the brief demanded, and it comes out cleanly on the side of genuine semantic movement.

## Propagation toward the natural B state (Part 6)

For fact-value patches, the induced downstream delta at the **final (answer) position** aligns with the natural trajectory `D_natural = H_B − H_A`:

- **cosine(D_induced, D_natural) at final pos:** mean **0.76–0.94** over downstream layers (earliest injection L0–L4 aligns best, 0.87–0.94).
- **distance-to-B improvement:** patching reduces `‖H_A − H_B‖ → ‖H'_A − H_B‖` by **~50–65%** at the final position, averaged over downstream layers.

So patching one internal state does not merely change the answer — it pushes later computation **along the model's own A→B manifold** toward the natural B state. Full per-layer traces for the top sites are in `propagation/traces.json`; raw `H'` stacks in `selected_raw/`.

## Perturbation sensitivity map (Part 3)

- **Most ablation-sensitive layers:** 3–8 (peak **L7**). Early-middle blocks are where zeroing a residual most disrupts the output.
- **Most random-sensitive layers:** 7, 6, 2, 14, 9.
- **Most sensitive positions** (mean ablation KL over layers): **final position 42 (10.9)** ≫ ` word` (2.6) > **fact_value 20 (1.9)** > sink token 0 (1.7) > ` lowercase` (1.2). The final position dominates (it feeds logits directly); among content tokens the fact value and the instruction words carry the most causal weight; pure chat-template tokens are largely inert **except** the attention-sink token 0.
- **Locality:** every intervention stayed local — positions strictly *before* the edited token were **bitwise unchanged** at all layers (causal-attention invariant), verified on all 8428 cells.

## Answers to the 10 questions

1. **Most intervention-sensitive layers:** early-middle, layers **3–8** (ablation), peaking at L7.
2. **Most causally important positions:** the **final/answer position**, then the **fact-value token**, the instruction words (` word`, ` lowercase`), and the **attention-sink token 0**.
3. **Content vs structural tokens:** content tokens are far more causally useful than chat-template tokens — with two exceptions that carry weight for mechanical reasons: the attention-sink (pos 0) and the final assistant-prefix position (feeds logits).
4. **Local or spreading:** interventions are **strictly local backwards** (earlier tokens never change) and spread **forward** modestly; a fact-value edit propagates to the answer position with high alignment. `spread_frac` and `rel_delta_finalpos` quantify this per cell.
5. **Persist / amplify / rotate / disappear:** **semantic** patches at the fact-value port **persist** (depth-stable KL) and **amplify onto the answer** (alignment rises toward output); **generic random** perturbations **disappear** with depth (RMSNorm washout); ablation is most disruptive at layers 3–8.
6. **Can activation patching transfer the semantic distinction B→A?** **Yes, unambiguously** — 28 single-site patches flip A's answer to `green`, p_B = 1.000, with no prompt-B text ever entering A (activation-only).
7. **Which sites move A toward B most?** The **fact-value token (pos 20) across layers 0–21** (semantic_transfer ≈ 27.4, argmax flip); plus the trivial final-position/last-layer output port.
8. **Do successful patches push toward the natural B trajectory or only disrupt?** **Toward the trajectory** — final-position cosine 0.76–0.94 to `D_natural`, distance-to-B down ~50–65%. Distinct from `randnorm`, which disrupts and decays.
9. **Broad receptive regions (future memory ports):** the **fact-carrying content token over a wide early-to-mid layer band (≈ L0–21)** is a broad, forgiving injection port, bounded by a **causal read-out deadline (~L22)**. The final-position/last-layer port is a trivial output override, not a routing port.
10. **Logical next experiment:** (a) repeat the cartography for a fact that must be **bound/inferred, not copied** (e.g. a comparative or a 2-hop fact) to see whether the causal region shifts **deeper and more distributed** — testing whether "copy ports" generalize; (b) identify the **mechanism of the L22 deadline** (which attention head copies fact→answer) via head-level patching; (c) toward the codec: probe the **minimal subspace of `H_B[20] − H_A[20]`** that carries the color, i.e. whether a *low-rank learned vector* (not a copied residual) injected at the port reproduces the flip. (c) is the bridge to PCDC write-back — **for discussion, not this phase.**

## What worked / failed / surprised

- **Worked:** all controls (identity, A←A, causal invariants, bitwise regression); token-aligned pair search; the full atlas; unambiguous semantic transfer with a clean semantic-vs-disruption separation.
- **Failed:** nothing scientific. One cosmetic bug (Windows cp1252 vs `→` in the report writer) — fixed with UTF-8; the sweep/data were unaffected (regression already confirmed bitwise-exact before it).
- **Surprised:** (1) the fact-value port is causally potent across an **enormous layer band (0–21)**, not just mid layers — the color identity is available and re-routable very early and stays routable until the copy. (2) A single-token patch can **exceed** the full natural contrast swing (27.54 > 27.44) — direct injection at the port slightly overshoots the natural B state. (3) The sharp **causal deadline at L22** is a crisp, mechanistic signature of the fact→answer copy.

## Honest scope caveat

The clean, wide causal region reflects that **this task is a verbatim copy** — the answer token *is* the fact-value token, so its causal structure is unusually concentrated and forgiving. The cartography **method** is general (it makes no copy assumption), but this specific **semantic result** is a best case. A fact requiring inference or binding would likely show a **narrower, deeper, more distributed** causal region — which is precisely why question 10(a) is the right next probe before designing any general memory plug.

## Files

```
baseline/H_A.pt H_B.pt logits_A.pt logits_B.pt   # full [28,43,1536] resid_post stacks + logits
token_map.json                                   # position table, roles, A/B diff, per-pos norms
meta.json                                        # env, pair, layer convention, timing estimate
perturbation_map/cells.jsonl                     # identity/ablation/rand_a0.3/rand_a1.0 per (L,P)
patch_map/cells.jsonl                            # selfA/patchB/randnorm per (L,P)
propagation/traces.json                          # per-layer final-pos traces for top sites
selected_raw/Hprime_*.pt                         # raw H' stacks for top semantic + control sites
heatmaps/*.png *.csv                             # the atlas (7 maps)
ranked_semantic_sites.csv                        # every patchB site ranked by semantic_transfer
cartography_summary.json                         # machine-readable verdict + top sites
```

Heatmaps: `patchB_semantic_transfer`, `patchB_argmax_is_B`, `patchB_kl_from_A`,
`patchB_mean_cos_finalpos` (trajectory alignment), `ablation_kl_from_control`,
`rand_a1.0_kl_from_control`, `rand_a1.0_top1_changed`.

## Reproduce

```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe cartography.py --pilot   # pair search + baselines + runtime estimate
.venv\Scripts\python.exe cartography.py           # full, resumable sweep + atlas + report
```

---

**STRICT STOP POINT.** Cartography + activation-patching analysis only. No PCDC
integration, no adapter/codec training, no memory injection, no weight changes.
Natural A/B residual differences were computed for analysis but **not** turned
into a generalized injection system. *Map the socket before we design the plug.*
Awaiting discussion before the next phase.
