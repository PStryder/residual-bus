# Phase 1 Results — Writable Residual Activation Bus

**Model:** Qwen2.5-1.5B-Instruct (local ModelScope copy) · 28 layers · hidden=1536
**Dtype:** FP32 · **Attn:** eager · **Device:** RTX 4080 (CUDA 12.4, torch 2.6.0)
**Prompt A:** "What is the capital of France? Answer in one word." (31 tokens, control top-1 = `Paris`)
**Sweep layers:** 11, 14, 16, 19 (≈40–70% depth) · **Intervention position:** final token only
**Test:** single forward pass, next-token logits (no generation, no KV cache)

## Verdict — SUCCESS

> "We can add a controlled vector to an intermediate residual-stream activation
> of a frozen local LLM and demonstrate a reproducible causal change in
> downstream logits." — **experimentally TRUE.**

| gate | result |
|------|--------|
| reproducible_capture | ✅ bitwise exact (max_abs_diff 0.0, cosine 1.0) across two identical passes |
| alpha0_is_true_noop | ✅ identity write → KL 0.0, top-1 unchanged, overlap 1.0 at all layers |
| random_sweep_monotonic_all_layers | ✅ KL non-decreasing with alpha at all 4 layers |
| direction_change_strong_all_layers | ✅ ablation / max-random KL ≥ 1.0 at all layers |
| top1_flips_under_strong_intervention | ✅ argmax flips under strong direction change at all layers |

## Experiment A — capture + reproducibility

Two identical forward passes, mid-layer residuals captured via `ReadHook` on the
block output (`resid_post`). All four layers reproduced **bitwise** (exact_equal
True, max_abs_diff 0.0). Logits also bitwise identical.

Per-layer captured activation `[31, 1536]`:

| layer | full-tensor norm | mean | std | min | max | last-token norm |
|-------|------|------|-----|-----|-----|-----------------|
| 11 | 11053 | 0.062 | 50.7 | −6627 | +5275 | 45.5 |
| 14 | 11052 | 0.057 | 50.6 | −6628 | +5277 | 51.5 |
| 16 | 11048 | 0.056 | 50.6 | −6630 | +5279 | 60.9 |
| 19 | 11050 | 0.052 | 50.6 | −6636 | +5283 | 66.6 |

**Note — massive activations.** The full-tensor norm (~11,053) is dominated by a
few outlier dimensions on the sink/first token (min −6627 / max +5275), a known
"massive activation" phenomenon. The *last-token* residual norm is only ~46–67.
This is exactly why interventions are scaled relative to the **per-position**
`‖h‖`, not a global norm.

## Experiment B — dose-dependent causal interventions

KL = KL(P_control ‖ P_intervened) at the final-token next-token distribution.

Random sweep `h → h + alpha·‖h‖·dir` (fixed random unit direction, seed 1234) —
clean monotonic dose-response:

| layer | a=0.1 | a=0.3 | a=1.0 | a=3.0 | top-1 @ a=3.0 |
|-------|-------|-------|-------|-------|----------------|
| 11 | 0.0000 | 0.0005 | 0.437 | 20.22 | `Paris`→`D` |
| 14 | 0.0000 | 0.0001 | 0.028 | 12.23 | `Paris`→`:` |
| 16 | 0.0000 | 0.0003 | 6.589 | 14.95 | `Paris`→`.` |
| 19 | 0.0000 | 0.0003 | 0.299 | 16.38 | `Paris`→`You` |

Ablation `h → 0` (final token): KL 3.7–20.4, argmax flips at every layer.

## Diagnostics (expected-weak; NOT success gates)

- **Pure scaling `h → (1+alpha)·h`** is weak even at 4× (self_a3.0 KL: L11 1.95,
  L14 0.46, L16 0.28, L19 0.05) despite norms ballooning to ~200–270. Cause:
  pre-norm **RMSNorm is scale-invariant**, so a pure magnitude change on the
  residual is largely normalized away downstream. A steering signal must change
  **direction**, not magnitude.
- **Replacement with prompt B's final-token residual** is near-inert
  (KL 0.0001–0.0013). Cause: prompts A and B share the same chat-template suffix,
  so their **final structural token** carries almost identical mid-layer state
  (norm 67→67). Question-specific content lives at earlier content-token
  positions, not the final `<...>model\n` token.

### Implication for the next phase (semantic delta injection)

`delta = h_B − h_A` taken at the **final token** would be tiny and near-useless.
Derive semantic deltas from **content-token positions** (and/or later layers),
and prefer **direction-changing** injection over magnitude scaling.

## Files

```
expA/experiment_a.json      per-layer stats + reproducibility
expA/activations_run{1,2}.pt {layer: [seq,hidden]} captured, two runs
expA/logits_run1.pt          next-token logits [vocab]
expB/experiment_b.json       per-layer, per-intervention metrics
expB/next_token_logits.pt    {control, layer{i}_{intervention}: [vocab]}
summary.json                 env + machine-readable verdict + per-layer KL
```

## Reproduce

```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe experiment.py
```

**STOP POINT.** Success criterion met. Do not proceed to semantic delta
injection without discussion (per project scope guard).
