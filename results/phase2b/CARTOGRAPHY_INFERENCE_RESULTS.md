# Phase 2B — Causal Cartography of DERIVED Information (Results)

**Model:** Qwen2.5-1.5B-Instruct (local) · 28 layers · hidden 1536 · FP32 · eager · RTX 4080
**Convention:** `H[L] = resid_post of block L = HF hidden_states[L+1]`. Single forward pass, no KV cache, frozen, `no_grad`. Same apparatus as Phase 2A; only the task changed.
**Cells:** 8568 = 6 ops × 28 layers × 51 positions (BtoA, AtoB, selfA, selfB, randA, randB), resumable.

## Task — a non-copy arithmetic transform

`"The dax starts with X stones. It receives 4 more stones. How many stones does the dax have now? Reply with just the number."` → **X+4**.

- **A:** X=1 → answer `5` (id 20). **B:** X=2 → answer `6` (id 21). One-token difference at **P_source = 20**; operand `4` at P=26; seq = 51.
- **Non-copy asserted:** the answer token never appears in its own prompt (searched + verified). Both baselines produce the correct derived answer with **p = 1.000**.
- Baselines **bitwise-reproducible**; natural A/B difference is **bitwise-zero before P_source** (causal mask holds).
- Contrast `C = logit(6) − logit(5)`: C_A = −14.36, C_B = +16.31, **S_natural = 30.67**.

## Verdict

| check | result |
|---|---|
| **cartography_sound** | ✅ **true** |
| selfA / selfB patch no-op | ✅ / ✅ (max \|transfer\| 0, max KL 0) |
| causal invariants (all 8568 cells) | ✅ earlier positions & lower layers bitwise unchanged |
| pristine regression after sweep (A & B) | ✅ **bitwise exact** |
| baselines reproducible | ✅ |
| **derived transfer B→A** | ✅ 28 sites flip A→`6` (the derived answer) |
| **derived transfer A→B** | ✅ 28 sites flip B→`5` |
| **bidirectional** | ✅ **28 sites flip in BOTH directions** |
| max transfer (B→A / A→B) | 30.82 / 30.71 (≈ S_natural 30.67) |
| source write deadline | **L21** (both directions) |

No thresholds were tuned. The argmax flips to an answer token that is absent from the prompt, so this is genuine transfer of *computed* information, not routing of a supplied value.

## The headline: a sharp source→final handoff at L22 (derived-state commitment)

The causal control of the derived answer occupies **two disjoint regimes with a clean boundary at layer 22**, and **nothing in between carries it**:

| regime | layers | causal site | effect |
|---|---|---|---|
| **source-writable** | **0 – 21** | source token (P20) | patching flips the derived answer, transfer ≈ S_natural, **bidirectional** |
| **committed** | **22 – 27** | final position (P50) | patching flips the derived answer; source token is **causally dead** |

Three independent signals put the boundary at exactly **L22**:

1. **Source-token cliff** (`source_transfer_vs_layer.png`): B→A and A→B transfer both sit at ~30.7 through L21, then **drop to 0.00 at L22** and stay dead. Near-perfect bidirectional symmetry.
2. **Final-position onset:** patching the final residual is **inert through L21** (`kl_from_base = 0.00`, no flip) then **snaps to full transfer at L22** (kl 16.5, p_answer = 1.000). The `kl=0` at L20–21 rules out the trivial "the last few layers just preserve the final residual" explanation — the answer is genuinely *absent* from the final residual until L22.
3. **Natural divergence (no patching):** `‖H_B − H_A‖` at the final position is ~0.03 early, ~10 at L20–21, then **jumps to 43.7 at L22** → 74 → 97 → … → 136 at L27. B's computed answer physically enters the final-position residual at L22.

All three coincide: **the source token stops mattering exactly where the derived answer is written into the final position.** That is a direct, three-way-corroborated observation of a **derived-state commitment point at L22**.

`strongest_position_vs_layer.png` shows the handoff as a hard jump: strongest B→A site = P20 for L0–21, then P50 for L22–27, with no intermediate positions ever winning.

## Migration? Only source→final; interior positions are causally inert

The brief asked whether causal influence migrates into *relational/content/question* positions. **It does not.** Patching interior tokens never flips the derived answer:

- operand `4` (P26): max transfer **0.94** (no flip) across all layers
- question tokens (` How`, ` many`, ` now`, `?`): max transfer **≤ 0.03** (no flip)
- **NON-source, NON-final flip sites: zero.** Every one of the 28 B→A flips is at either P20 (source) or P50 (final).

Note a genuine nuance: the *natural* A/B divergence **does** spread across the sequence (operand diverges from L1, question tokens from L6–9), so the source's influence is present at many positions — but that divergence is **not causally sufficient**: injecting it doesn't move the answer. Divergence ≠ causal control. The derived answer is handed off **directly source→final**, with no intermediate causal way-station.

## Semantic transfer vs generic disruption (control target-specificity)

`randA` (norm-matched random replacement) vs `patchB` at the source token:

- **argmax→derived-target flips:** patch **28/28** vs random **1/1428** (a lone chance hit under a large perturbation).
- random max KL 19.4 (strong disruption) but it does **not** produce the specific derived answer.
- The `semantic_transfer` *contrast* metric can read moderately high for random at the source (max 18.5 — destroying the "5" evidence relatively favors "6"), which is exactly why **argmax→target**, not the contrast alone, is the decisive discriminator for a derived task. Generic disruption ≠ semantic transfer.

## Downstream trajectory alignment (Part 9/10)

Top source-token patches move A's downstream computation along the natural A→B manifold (from `traces.json` / `BtoA_prop_*` heatmaps): final-position `cosine(D_induced, D_natural)` is high and distance-to-B improves where the patch takes. The A→B direction is symmetric (the source-token plot overlays almost perfectly), i.e. sites are **bidirectionally** effective rather than exploiting a one-token asymmetry.

## Phase 2A (copy) vs Phase 2B (derive) — quantified

| quantity | 2A (copy) | 2B (derive) |
|---|---|---|
| source write deadline | L22 | **L21** (≈ same) |
| source flip-layers | 23 | 22 |
| non-source flip sites | 5 (final pos, late) | 6 (final pos, L22–27) |
| max transfer / S_natural | 27.54 / 27.44 | 30.82 / 30.67 |
| random KL @source shallow (L0–5) | 4.14 | 5.57 |
| random KL @source **deep (L16–21)** | **0.68** | **4.56** |
| bidirectional tested | no (copy, one-directional) | **yes, 28/28** |

**What's the same:** the *topology* — a wide source stripe (L0–21) plus a final-position late band — and the ~L21–22 source deadline. The write deadline appears to be a **position/architecture property, not a function of task difficulty** (it barely moved despite the task now requiring computation).

**What's different (and meaningful):**
1. In 2A the final-position band trivially carries the *copied* answer token; in 2B it carries a **computed** answer, and the natural-‖D‖ jump at L22 shows that answer being **written** into the final position — a real commitment event, not a copy.
2. The 2B handoff is **sharp and complementary** (source dies exactly where final activates); the answer is derived, so this is genuinely "computed-state commitment."
3. **Random disruption at the source persists far deeper in the derived task** (deep-layer KL 4.56 vs 0.68). Interpretation: in the arithmetic task the source value is still being *actively used* (re-read for computation) through mid layers, so corrupting it keeps mattering; in the copy task it is passively waiting to be copied and off-manifold noise washes out by depth. This is a clean quantitative fingerprint of "computation" vs "routing."
4. Bidirectional symmetry is strong and explicit in 2B.

## Answers to the 13 questions

1. **Derived (non-copy) transfer?** **Yes** — 28 B→A sites flip A to `6` and 28 A→B sites flip B to `5`; the answer token is never in the prompt.
2. **Bidirectional?** **Yes** — the same 28 (L,P) sites flip in both directions; source-token curves overlay.
3. **Where is the source fact writable?** At **P20**, layers **0–21**, transfer ≈ S_natural.
4. **Source write deadline?** **L21** (transfer cliffs to 0 at L22).
5. **Does influence migrate off the source token?** **Yes, but only to the final position** (P50) at L22+. Not to interior/relational tokens.
6. **Which later positions become causal?** Only the **final/answer position**; operand and question positions are causally inert (max transfer ≤ 0.94, no flips).
7. **Separate derived representation?** **Yes** — the derived answer is written into the final-position residual at L22 (natural ‖D‖ jumps 10→44; final-position patching goes inert→full at L22). It is a distinct, later state, not the source representation.
8. **Derived-state commitment region?** **Yes, L22–27**, onset sharply at L22, coincident with the source deadline.
9. **Downstream trajectory alignment?** Successful source patches move downstream states along the natural A↔B trajectory (high final-pos cosine, distance-to-target improves), symmetric both ways.
10. **Vs Phase 2A?** Same topology and deadline; 2B adds a genuine computed-state commitment, sharp complementary handoff, deeper-persisting source disruption, and demonstrated bidirectionality (table above).
11. **Most interesting sites for further study:** the **L21→L22 boundary at P20 and P50** (the handoff), and the attention that writes source→final at L22.
12. **Ambiguous:** the *mechanism* of the L22 write (which head/component) is not resolved here (out of scope — no head-level decomposition); the small-integer single-token arithmetic may be idiosyncratically clean; whether the deadline shifts for larger operands / multi-token answers is untested.
13. **Surprises:** (a) the source deadline barely moved vs the copy task (≈L21 vs L22) — it looks architectural, not task-driven; (b) the handoff is a **hard cliff**, not a gradual migration; (c) **no** interior/relational token is causally sufficient — it is a direct source→final handoff; (d) random disruption at the source persists ~7× deeper in the derived task, a crisp "computation vs routing" signature.

## What the evidence supports — and what it does not

- **Supported:** activation patching transfers a *derived* (never-present) answer, bidirectionally; a sharp source-write deadline (L21) coincident with a derived-state commitment onset (L22) triple-corroborated by source-cliff, final-onset, and natural-divergence; target-specificity vs random controls; trajectory alignment.
- **Not supported / not claimed:** distributed causal representation across interior tokens (absent); any specific *mechanism* for the L22 write (not investigated); generality beyond this single-token small-integer addition pair (untested). Divergence at interior positions exists but is not causally sufficient.

## Files

```
task_selection.json  token_map_A.json token_map_B.json     # task, search log, token tables
baselines/H_A.pt H_B.pt logits_A.pt logits_B.pt            # full [28,51,1536] stacks + logits
natural_delta/{dnorm,rel,cos}.csv                          # natural A/B divergence atlas
patch_map/cells.jsonl                                       # all 8568 cells (6 ops)
propagation/traces.json  selected_raw/Hprime_*.pt          # traces + raw H' for key sites
heatmaps/*.png *.csv                                        # atlas (incl. source_transfer_vs_layer,
                                                            #  strongest_position_vs_layer, natural_dnorm)
ranked_{BtoA,AtoB,bidirectional,nonsource_BtoA,random_disruption}.csv
phase2b_summary.json  metrics.json
```

## Reproduce

```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe cartography_infer.py --pilot   # pair search + baselines + estimate
.venv\Scripts\python.exe cartography_infer.py           # full bidirectional sweep + atlas + report
```

---

**STRICT STOP POINT.** Mapping pass only. No codec, no adapter/probe training, no
fine-tuning, no weight changes, no PCDC integration, no head-level mechanistic
decomposition, no arbitrary/absent-fact injection. Natural A/B differences were
computed for analysis only. *Map the transformation — then stop.* Awaiting discussion.
