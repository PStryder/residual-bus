# Phase 2C — Causal Cartography of Categorical Inference (Results)

**Model:** Qwen2.5-1.5B-Instruct (local) · 28 layers · hidden 1536 · FP32 · eager · RTX 4080
**Convention:** `H[L] = resid_post of block L = HF hidden_states[L+1]`. Single forward pass, no KV cache, frozen, `no_grad`. Same apparatus as 2A/2B (carto_lib.py reused unchanged); only the task changed.
**Cells:** 5376 = 6 ops × 28 layers × 32 positions (BtoA, AtoB, selfA, selfB, randA, randB), resumable.

## Task — numeric premise → categorical answer (non-copy)

`"Is X greater than 4? Answer in one word."` → **Yes/No** (a different *kind* of value than the numeric source).

- **A:** X=1 → answer `No` (id 2753). **B:** X=5 → answer `Yes` (id 9454). One-token difference at **P_source = 16**; threshold `4` at **P=20**; seq = 32.
- **Non-copy asserted:** neither `Yes` nor `No` appears in the prompt. Baselines confident: **p = 1.000 / 0.999**. Bitwise-reproducible; natural A/B difference bitwise-zero before P_source.
- Contrast `C = logit(Yes) − logit(No)`: C_A = −13.55, C_B = +15.42, **S_natural = 28.97**.

## Verdict

| check | result |
|---|---|
| **cartography_sound** | ✅ **true** |
| selfA / selfB no-op | ✅ / ✅ |
| causal invariants (all 5376 cells) | ✅ bitwise |
| pristine regression (A & B) | ✅ bitwise exact |
| baselines reproducible | ✅ |
| **categorical transfer B→A / A→B** | ✅ **26 / 28 flips** |
| **bidirectional** | ✅ **26 sites both ways** |
| max transfer | 29.12 (≈ S_natural 28.97) |
| source write deadline | **L11** (both directions) |
| final-position onset | **L17** (B→A) / L16 (A→B) |
| natural final-divergence jump | **L22** |
| three signals coincide | **✗ (spread over 11 layers)** |
| interior site causally sufficient | ✅ **yes (threshold token, L12–14)** |
| random target-flip rate | 3 / 896 = 0.003 |
| **L21/L22 boundary hypothesis** | **weakened** |

Numeric→categorical transfer is real and bidirectional. But the *anatomy* is different from 2B, and that difference is the main result.

## Headline: a three-stage causal relay (not a sharp two-region handoff)

`strongest_position_vs_layer.png` is a clean staircase. Causal control of the categorical answer moves through **three** sites:

| stage | layers | causal site (role) | behavior |
|---|---|---|---|
| **1. source** | **0 – 11** | P16 `1`/`5` (source_value) | patching flips the category; transfer 29 → 15 (decaying) |
| **2. threshold waypoint** | **12 – 14** | P20 `4` (threshold operand) | **NEW** interior site: after the source dies, patching the *threshold* residual flips the answer (transfer ~26) |
| **(diffuse gap)** | 15 – 16 | — | no single position is sufficient; the answer is "in flight" |
| **3. final** | **17 – 27** | P31 (assistant_prefix / final) | commits; transfer 16 → 29, p_answer → 0.999 |

Contrast with **2B (arithmetic)**, which was a *direct* source→final handoff at ~L21/L22 with **zero** interior sites. Categorical comparison exposes an **interior causal waypoint** — the threshold operand `4` — exactly the outcome the brief flagged as possible ("source → comparison/relation token → final") and did **not** appear for copy or addition.

Three independent facts define the relay:
1. **Source cliff at L11** (`source_transfer_vs_layer.png`): B→A and A→B both ride ~29 through L11 then **drop to ~0 at L12** — near-perfect symmetry. Source write deadline **L11**.
2. **Threshold waypoint L12–14:** the only interior flips in the whole sweep are at P20, layers 12–14 (transfer 24–27, p_answer 0.999). Corroborated by the natural atlas: `‖H_B−H_A‖` at the threshold token steps up right there (L12 24.6 → L14 30.3).
3. **Final onset L17** (`final_transfer_vs_layer.png`): final-position patching is inert through L16 (kl 0), flips from **L17** (p_answer 0.928 → 0.999).

The **source deadline (L11), final onset (L17), and natural final-divergence jump (L22) are spread across 11 layers** — they do **not** coincide, unlike 2B where all three sat at ~L22.

## Semantic transfer vs generic disruption

Even cleaner target-specificity than 2B: `randA` (norm-matched random replacement) flips the category **3/896 times** (rate 0.003; the 3 are edge cells — the attention-sink token and one threshold cell at the L11 transition), vs 26 real B→A flips. Random disruption at the source **washes out entirely by deep layers** (KL 5.5e-5 at L16–21) — consistent with the source being causally dead after L11. Semantic direction, not perturbation magnitude, drives the flip.

## Three-phase comparison (2A copy / 2B arithmetic / 2C categorical)

| quantity | 2A copy | 2B arithmetic | 2C categorical |
|---|---|---|---|
| source write deadline | L22 | L21 | **L11** |
| source stripe width (flip layers) | 23 | 22 | **12** |
| final-position onset | L23 | L22 | **L17** |
| **interior causally-sufficient sites** | **0** | **0** | **3 (threshold, L12–14)** |
| source-death ↔ final-onset coincide | yes (sharp) | yes (sharp) | **no (spread 11L)** |
| **natural final-divergence jump** | **L23** | **L22** | **L22** |
| max transfer / S_natural | 27.5 / 27.4 | 30.8 / 30.7 | 29.1 / 29.0 |
| bidirectional | (not tested) | 28/28 | 26/26 |
| random deep-KL @source (L16–21) | 0.68 | 4.56 | **≈0 (5e-5)** |
| random target-flip rate | 0.0 | 0.0007 | 0.003 |

Two things move together and two things stay put:
- **What changed with task:** the source write deadline (L22 → L21 → **L11**) and the *existence of an interior waypoint* (none → none → **threshold operand**). The sharp coincident handoff of 2A/2B **did not reproduce**; categorical comparison discharges the source early and relays the result through the comparison operand before committing at the final position.
- **What stayed put:** the **natural final-position divergence jump at ~L22–23 in all three tasks**, and max transfer ≈ S_natural with clean target-specificity in all three.

## L21/L22 boundary hypothesis — classification: **WEAKENED**

The 2B hypothesis was: *"~L21/L22 is a general transition where source-position control converts into final-position answer-state control."* Phase 2C **weakens** it:

- In 2C the source→final conversion is **complete by L17** and is **routed through an interior waypoint (L12–14)**; the L21/L22 layers are **not** where source control converts to final control here. Source deadline moved to **L11**.
- So the specific claim "L21/L22 = the causal source→final handoff" is **task-dependent, not architectural** — it held for copy and addition but not for comparison.

A **narrower, weaker invariant does survive** and should be stated conservatively: across all three tasks the **steepest natural change in the final-position residual occurs at ~L22–23** (`natural_final_divergence.png`). That is a **candidate final-position "commitment/read-out" region** — but it is decoupled from source-writability (which varies L11–L22 by task), and this phase does **not** establish any mechanism for it. It is not a "reasoning layer"; it is, at most, a task-invariant *final-position read-out layer* whose causal role is only partially characterized.

## Answers to the report questions

1. **Numeric→categorical transfer?** Yes — 26 B→A / 28 A→B sites flip the category to the correct opposite answer (never in prompt).
2. **Bidirectional?** Yes — 26 sites both ways; source-token curves overlay.
3. **Source writable where?** P16, layers 0–11.
4. **Source write deadline?** **L11** (both directions) — much earlier than 2A/2B.
5. **Final-position onset?** **L17** (B→A) / L16 (A→B).
6. **Natural final-divergence transition?** Jump at **L22**.
7. **Do the three coincide?** **No** — L11 / L17 / L22, spread over 11 layers.
8. **Interior sites sufficient?** **Yes** — the threshold operand `4` (P20) at L12–14. A genuine causal waypoint.
9. **Migrate or jump?** **Multi-hop migration:** source → threshold operand → (diffuse) → final. A three-stage staircase.
10. **Target-specific vs random?** Yes — random flips 3/896; real flips are localized to source/threshold/final.
11. **Downstream trajectory alignment?** Top source/threshold patches move downstream states toward the natural target trajectory (prop_cos_final / dist→B in ranked_BtoA.csv); symmetric both directions.
12. **Vs 2A/2B?** Earlier source deadline, an interior waypoint (new), non-coincident signals — see the table. The stable feature is the ~L22 natural final-divergence jump.
13. **Boundary hypothesis?** **Weakened** (see above); a narrower L22 final-read-out invariant survives.
14. **Ambiguous:** whether the earlier L11 deadline is driven by task simplicity vs prompt length/structure (both plausible; untested); the *mechanism* of the threshold waypoint and the L22 read-out (not investigated — out of scope); single-token small-integer idiosyncrasy.
15. **Surprises:** (a) an interior causal waypoint appeared for the first time (the threshold operand) — comparison genuinely routes through its operand; (b) the source deadline nearly halved (L11 vs L21) while (c) the natural final-divergence jump stayed pinned at ~L22 across all three tasks — i.e. source-writability and final-read-out are **decoupled** and only the latter is task-invariant.
16. **Next anatomical test (for discussion, NOT executed):** to disambiguate "task-simplicity vs prompt-structure" for the L11 deadline, hold the surface form fixed and vary only comparison difficulty; and to test whether the threshold-waypoint generalizes, vary which operand is the shared/threshold token. Mechanism-level questions (which component performs the L~22 final read-out) are explicitly deferred.

## What the evidence supports — and what it does not

- **Supported:** numeric→categorical activation transfer, bidirectional, target-specific; a **three-stage causal relay** with a real **interior waypoint** (threshold operand); an **early source deadline (L11)**; a **task-invariant ~L22 natural final-position divergence jump**.
- **Not supported / not claimed:** the 2B "L21/L22 source→final handoff" as an architectural constant (weakened); any *mechanism* for the threshold waypoint or the L22 read-out (uninvestigated); generality beyond this single-token small-integer comparison. The natural divergence at interior positions again exists without being causally sufficient except at the specific threshold-waypoint band.

## Files

```
task_selection.json (+ search log) token_map_A/B.json   # task, non-copy proof, token tables
baselines/H_A.pt H_B.pt logits_A.pt logits_B.pt          # full [28,32,1536] stacks + logits
natural_delta/{dnorm,cos,final_dnorm}.csv                # natural A/B divergence atlas
patch_map/cells.jsonl                                     # all 5376 cells (6 ops)
propagation/traces.json selected_raw/Hprime_*.pt         # traces + raw H' for key sites
heatmaps/*.png *.csv                                      # atlas: source_/final_transfer_vs_layer,
                                                          #  strongest_position_vs_layer, natural_final_divergence, ...
ranked_{BtoA,AtoB,nonsource_BtoA,random_disruption}.csv
phase2c_summary.json  metrics.json
```

## Reproduce

```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe cartography_categorical.py --pilot   # pair search + baselines + estimate
.venv\Scripts\python.exe cartography_categorical.py           # full bidirectional sweep + atlas + report
```

---

**STRICT STOP POINT.** Mapping pass only. No codec / adapter / probe / classifier
training, no fine-tuning, no weight changes, no PCDC integration, no head-level
decomposition. The threshold-waypoint and the L22 read-out are documented as real
observations; their mechanisms are deliberately left uninvestigated. Awaiting discussion.
