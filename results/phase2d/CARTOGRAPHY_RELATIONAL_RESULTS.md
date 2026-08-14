# Phase 2D — Causal Cartography of Relational Binding (Results)

**Status: HALTED AT TASK SELECTION (a valid scientific outcome).**

**Model:** Qwen2.5-1.5B-Instruct (local) · FP32 · eager · deterministic · frozen. Apparatus unchanged from 2A/2B/2C; `carto_lib.py` and `cartography.py` reused unmodified; new entry point `cartography_relational.py`. **No sweep was run** — there is no valid task to map. Phases 1/2A/2B/2C untouched.

## What happened

The protocol required a directed 2-link path-composition task with a clean, high-confidence, one-token-difference A/B pair whose ground-truth answer flips (A→Yes, B→No) with the binding change:

```
Only these directed links exist. {s} points to {m}. {x} points to {d}.
Is there a path of exactly two links from {s} to {d}? Answer in one word.
   A: x = m  → s→m→d           → Yes
   B: x = s  → s→m, s→d (no 2-hop) → No
```

The in-entrypoint search tried **260** entity triples and found **0** valid pairs. A follow-up diagnostic over **5 phrasings × 24 entity triples (120 configs)** confirmed the cause: **the model is binding-insensitive.**

| template | valid pairs (p≥0.9) | binding-sensitive-correct (any conf) | **insensitive: A==B answer** |
|---|---|---|---|
| path_exact | 0 | 0 / 24 | 12 / 24 |
| reach_2steps | 0 | 3 / 24 (all p≈0.3) | 21 / 24 |
| arrow_2hop | 0 | 0 / 24 | **24 / 24** |
| follows | 0 | 0 / 24 | **24 / 24** |
| gives | 0 | 0 / 24 | **24 / 24** |

The model gives the **same** Yes/No answer to A and B — it anchors on surface phrasing (answers "Yes" to everything under one wording, "No" under another) and does **not** compute `endpoint(rel1) == startpoint(rel2)` and compose the path. The single template with any correct-polarity hits (`reach_2steps`, 3/24) produced them at **p ≈ 0.30** — chance level, not reasoning.

Example (`path_exact`, s=Tom m=Sam d=Ben), next-token top-5:
- **A** (x=Sam, ground truth **Yes**): `Yes` 0.517, `No` 0.336, `no` 0.075, `yes` 0.071 — weak, and
- **B** (x=Tom, ground truth **No**): `Yes` 0.666, `No` 0.203 — **wrong** (says Yes), i.e. identical polarity to A.

## Decision — HALT, do not weaken the microscope

Per the protocol ("*If no clean high-confidence one-token-difference task can be found, STOP and report that rather than weakening the protocol*" and "*Do not change the microscope just because the specimen became harder*"), the confidence bar was **not** lowered. Running the exhaustive patch sweep on a task the model cannot perform would produce a causal map of **noise**, not of relational binding — the natural A/B difference would not encode a derived-answer distinction, so "semantic transfer" would be undefined.

This is an explicitly permitted result: *"Positive semantic transfer is not required… If relational information appears distributed and single-site patching cannot capture it, document that and stop."* Here the situation is even upstream of that: the **computation itself is absent** from the model's single-pass behavior.

## The finding (and why it matters for the arc)

**Cartography can only chart computations the model actually performs.** 2A/2B/2C were mappable because the model does copy, addition, and scalar comparison confidently (baseline p ≈ 1.0). 2-hop relational composition is **beyond this 1.5B model's reliable single-forward-pass capability**, so there is no relational computation whose causal anatomy could be written into or read out of the residual stream.

| phase | task | baseline p(correct) | model performs it? | mapped? |
|---|---|---|---|---|
| 2A | semantic copy | ~1.000 | yes | ✅ |
| 2B | arithmetic derivation | ~1.000 | yes | ✅ |
| 2C | scalar comparison → category | 1.000 / 0.999 | yes | ✅ |
| **2D** | **relational 2-hop composition** | **≤ ~0.35, binding-insensitive** | **no** | **✗ (halt)** |

This **extends** the Phase-2 thesis. 2A→2C showed that *writable causal sites depend on the computation*. 2D adds the stronger precondition: **a computation must exist in the model before its causal anatomy can be mapped at all.** The "writable semantic site is computation-dependent" hypothesis is therefore consistent with 2D — trivially and importantly, an absent computation has no writable site.

## The ~L22–23 readout hypothesis — NOT testable in 2D

Across 2A (L23) / 2B (L22) / 2C (L22) the steepest natural final-position divergence sat at ~L22–23. **2D cannot test this:** with no binding-sensitive pair, the natural A/B final-position divergence does **not** encode a derived-answer difference, so its depth profile is not comparable to the prior phases. The hypothesis remains **as of 2C (clustered across three tasks)**, untested by 2D. It should not be reported as strengthened or weakened here.

## Report questions (answered honestly given the halt)

1. **Relationally-derived transfer occur?** Not measured — no valid task (model cannot perform the composition).
2–16. **Causal structure (deadline / waypoints / join entity / gaps / final onset / trajectory):** N/A — no sweep was run, because there is no reliable relational computation to map.
17. **Computation-dependent writable sites?** Consistent and extended: an absent computation has no writable site.
18. **Ambiguous:** whether a larger model, multi-token entities, or chain-of-thought / multi-pass would make the task tractable and expose a mappable relational anatomy — untested **by design** (the microscope, including single-pass and model choice, was held fixed).
19. **Surprised?** The model is near-completely binding-**insensitive** (24/24 identical answers under 3 of 5 phrasings), not merely low-accuracy. It performs *no* relational composition; its output is driven by a phrasing prior.
20. **Next anatomical test (for discussion, NOT executed):** either (a) select a relational task **within** the 1.5B model's single-pass capability (e.g. 1-hop attribute/coreference binding) to map genuine binding, or (b) deliberately change **one** controlled variable (model size, or allow chain-of-thought / multi-pass) — a scoped decision for discussion, not an automatic escalation.

## Success criterion

Phase 2D's criterion was: *cartography remains sound AND we obtain a reproducible causal atlas for a genuine relational task; positive transfer not required.* Part A holds (apparatus sound, controls unchanged, prior phases untouched). Part B could not be satisfied because **the model does not perform the task** — established by a **reproducible** 120-config diagnostic. The scientifically correct action was to halt and report, which is what this document does.

## Files

```
task_selection_diagnostic.json   # 5 templates x 24 triples: binding-insensitivity evidence + examples
phase2d_summary.json             # machine-readable halt record + four-phase task tractability
CARTOGRAPHY_RELATIONAL_RESULTS.md
(baselines/ natural_delta/ patch_map/ ... created but empty — no sweep was run)
```

## Reproduce

```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe cartography_relational.py --pilot   # -> HALT: no clean relational pair
```

---

**STRICT STOP POINT.** Halted at task selection. No confidence-bar weakening, no
multi-site/combinatorial patching, no head-level decomposition, no model change,
no codec/adapter/probe/PCDC. The right next move is a **discussion** about whether
to map a simpler (model-tractable) relational binding or to change one controlled
variable deliberately. Awaiting that discussion.
