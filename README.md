# residual-bus

**A minimal, framework-free study of a *writable* residual-stream interface in a frozen local LLM — from "can we write to the bus at all?" to "is a constructed steering direction a reusable, shared latent primitive?"**

Everything here runs on one consumer GPU against a frozen **Qwen2.5-1.5B-Instruct**, in plain PyTorch + Hugging Face Transformers with raw forward hooks. No training, no fine-tuning, no interpretability framework. Deliberately tiny code; heavily instrumented experiments; adversarial self-checks.

---

## Why this exists

A prior project (**PCDC**, a predictive-coding sidecar for a frozen LLM) stalled on one problem: it could *read* the model's internal representations and produce structured signals, but there was no reliable way to **write** a useful signal *back* into the model's computation. Projecting sidecar state through the LM head failed because it wasn't aligned to the residual-stream manifold.

The [safety-research/introspection-mechanisms](https://github.com/safety-research/introspection-mechanisms) work showed causal read/write intervention on intermediate activations. `residual-bus` is the ground-up investigation of that interface: **before designing a "plug" (a memory→latent writer), map the "socket."**

> **Map the socket before you design the plug.**

This repo is that mapping, taken far enough to answer: *does a reusable latent write primitive actually exist?*

---

## TL;DR of the findings

| phase | question | verdict |
|---|---|---|
| **1** | Is the residual stream writable at all? | ✅ Yes — reproducible, dose-dependent causal control; **direction** matters, magnitude is washed out by RMSNorm |
| **2A** | Where can a *copied* fact be causally rewritten? | Wide receptive band at the fact token; sharp causal **deadline ~L22** |
| **2B** | A *derived* (arithmetic) answer? | Bidirectional transfer; sharp source→final handoff at ~L22 |
| **2C** | A *categorical* (comparison) answer? | Different topology — early source deadline (L11), an interior **operand waypoint**, distributed handoff |
| **2D** | A *relational* (2-hop) answer? | **HALT** — the 1.5B model can't do the task, so there's no computation to map (reported honestly, not forced) |
| **3A** | Does an *additive* constructed delta write back? | ✅ **PASS** — sub-replacement flips at α≈0.5; semantic ≫ random; cartography predicts where injection works/fails |
| **3C** | Does an *averaged* direction generalize to held-out prompts? | ✅ **STRONG** — 100% held-out flip; specific; direction (not raw magnitude) generalizes |
| **3N** | Is 3C better than chance under a fair null? | ✅ **STRONG REJECTION** of the null — 0/5000 matched-noise trials reproduce it; on-manifold wrong-info deltas (cos≈0.4–0.5) still produce zero effect |
| **3LB** | Is the effect specific to the *selected* layer? | ✅ **ROBUST** — works across the whole receptive band (L4–L18), fails at the post-deadline site |
| **3XL** | Do different layers share a semantic coordinate system? | ✅ **BROAD-PORTABILITY (receptivity-dominated)** — any receptive direction works at any receptive layer; the gating factor is **site receptivity**, not the layer where the direction was built |

**Bottom line:** within this frozen system, a single constructed direction — placed at any *receptive* site — steers unseen contexts toward an intended concept, cleanly, specifically, and beyond any fair null. The socket is real and unusually well-characterized. Building the plug (a learned memory→direction map) is deliberately **not** done here.

---

## The frozen apparatus (identical across all phases)

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, weights frozen (`eval`, `requires_grad_(False)`).
- **Precision:** **FP32** (removes both quantization *and* bf16-rounding confounds; VRAM is not the constraint).
- **Attention:** `eager` (transparent compute path — no fused kernels between blocks).
- **Determinism:** fixed seeds, TF32 off, cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`. Results are bitwise-reproducible on this hardware.
- **Reads/writes:** raw PyTorch `register_forward_hook` on decoder blocks. **One layer convention:** `H[L] = resid_post of block L = HF hidden_states[L+1]` (so `hidden_states[0]` = embeddings). Read and write hooks are separate context-managed objects with guaranteed cleanup.
- **No KV cache**, single forward pass — the cleanest possible causal readout.
- **Hardware used:** RTX 4080 (16 GB), CUDA 12.4, PyTorch 2.6.

---

## Repository layout

### Core library (reused unchanged across phases)
| file | role |
|---|---|
| `model.py` | model-agnostic loader (FP32, eager, deterministic, frozen); locates decoder layers for Qwen/Llama/Gemma |
| `intervention.py` | `ReadHook` / `WriteHook` context managers + delta builders (zero, norm-relative random, add-self, replace) |
| `carto_lib.py` | full-stack capture engine + metrics (KL, entropy, answer-contrast, causal invariants, propagation) |

### Phase entry points
| file | phase | what it runs |
|---|---|---|
| `experiment.py` | **1** | writable-bus validation (capture + reproducibility; dose-dependent causal interventions) |
| `cartography.py` | **2A** | causal cartography of a copy task |
| `cartography_infer.py` | **2B** | cartography of derived (arithmetic) information |
| `cartography_categorical.py` | **2C** | cartography of categorical inference |
| `cartography_relational.py` | **2D** | relational binding (halts at task selection — model can't do it) |
| `phase3_additive.py` | **3A** | additive write-back `h + α·D`; dose-response; cartography-as-oracle |
| `phase3_generalization.py` | **3C** | held-out generalization of an averaged direction `D_mean` |
| `phase3_null.py` | **3N** | adversarial empirical null test (preregistered) |
| `phase3_layerband.py` | **3LB** | layer-band robustness (is L8 a cherry-pick?) |
| `phase3_xlayer.py` | **3XL** | cross-layer portability matrix |

### Results (`results/`)
Each phase writes a self-contained folder with a machine-readable `*_summary.json`, a human-readable `*_RESULTS.md`, per-cell JSONL, CSVs, and plots. Start with the `*_RESULTS.md` in each:

```
results/
  expA/ expB/ summary.json          # Phase 1
  phase2a/CARTOGRAPHY_RESULTS.md
  phase2b/CARTOGRAPHY_INFERENCE_RESULTS.md
  phase2c/CARTOGRAPHY_CATEGORICAL_RESULTS.md
  phase2d/CARTOGRAPHY_RELATIONAL_RESULTS.md      # the honest HALT
  phase3a/PHASE3A_RESULTS.md
  phase3c/PHASE3C_GENERALIZATION_RESULTS.md
  phase3n/NULL_DESIGN.md  PHASE3N_NULL_RESULTS.md   # preregistration + result
  phase3lb/PHASE3LB_RESULTS.md
  phase3xl/DESIGN.md  PHASE3XL_RESULTS.md
```

> **What's committed vs regenerable.** All code, all reports/metrics/CSVs/plots, and the tiny reusable **`D_mean` direction vectors** are committed. The large raw activation tensors (baseline stacks, full logits, selected `H'` snapshots — ~234 MB, all reproducible from the scripts) and the model weights / venv are **git-ignored**. Everything can be regenerated deterministically.

---

## The phases in a bit more detail

**Phase 1 — the bus is writable.** Adding a controlled vector to a mid-layer residual produces a reproducible, dose-dependent change in next-token logits. α=0 is an exact no-op; a norm-relative random direction gives a clean monotonic dose-response. Key lesson that shapes everything after: **pure magnitude scaling is weak** (pre-norm RMSNorm is scale-invariant) — a steering signal must change *direction*.

**Phase 2 (A–D) — causal cartography.** Exhaustive layer×position activation patching maps *where* a distinction is causally writable, and how that changes with the *kind* of computation. Copy (2A) and arithmetic (2B) give a sharp source→final handoff near L22; categorical comparison (2C) instead routes through an interior operand waypoint with an earlier deadline. The relational task (2D) **halted honestly**: the model doesn't reliably perform 2-hop path composition, so there is no computation to map — cartography can only chart computations the model actually runs.

**Phase 3 (A/C/N/LB/XL) — from transplant to constructed write-back.** The ladder from "replay a real activation" toward "a reusable primitive": additive delta (3A) → averaged direction generalizing to held-out prompts (3C) → survives the strongest fair null (3N) → holds across the receptive layer band (3LB) → the band shares one coordinate system gated by site receptivity (3XL). The `D_mean` directions are constructed with **no training** — just deterministic averaging of natural activation differences.

---

## Reproduce

### 1. Environment
```powershell
cd residual-bus
uv venv --python 3.11
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

### 2. Get the model
HuggingFace CDN was IP-rate-limiting during development; the model was pulled from **ModelScope** instead (Qwen is Alibaba's, hosted there natively):
```powershell
uv pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
```
(Any local copy of the same checkpoint works — HF `Qwen/Qwen2.5-1.5B-Instruct` if your network allows it.)

### 3. Run any phase
All scripts read the model path from an env var and run offline:
```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1

python experiment.py                       # Phase 1
python cartography.py                       # Phase 2A   (add --pilot for a dry run + runtime estimate)
python phase3_additive.py                   # Phase 3A
python phase3_generalization.py             # Phase 3C
python phase3_null.py                        # Phase 3N   (adversarial null; ~30 min)
python phase3_layerband.py                   # Phase 3LB
python phase3_xlayer.py                       # Phase 3XL
```
Exhaustive sweeps are resumable (per-cell JSONL) and print a runtime estimate from a small pilot before the full pass.

---

## Methodological stance

This project tried hard *not* to fool itself:

- **Predeclared success gates and interpretation classes** before looking at results; corrections to a metric are documented *before* re-evaluation, never tuned to force a pass.
- **Controls everywhere:** α=0 / identity / self-patch no-ops; norm-matched random directions; causal-mask invariants checked on every cell; bitwise pristine-baseline regression after every sweep.
- **Adversarial validation:** Phase 3N is a preregistered null test whose explicit job is to *kill* the Phase 3 interpretation with the strongest fair null (including on-manifold real-delta comparators that share ~half the target direction). It didn't die.
- **Honest halting:** Phase 2D stopped at task selection rather than run cartography on a task the model can't perform. A negative/HALT is reported as a result.
- **Stated caveats:** the strong results are **conditional on a frozen pipeline** (layer/gain/task chosen a priori), a **small held-out set (N=6)**, and a **single concept (blue→green) in one task family**. This maps a real effect *within this system*; it is not a universality claim, and no learned transform was tested.

---

## What this does *not* establish

- That the effect generalizes across models, tasks, or arbitrary concepts (untested).
- Any *learned* memory→direction mapping (a **codec** / adapter) — deliberately out of scope here.
- No PCDC integration, no compression/PCA/probes, no attention-head decomposition.

The natural next step (a **Phase 4** conversation, not started in this repo) is compression/low-rank of `D_mean` and then a learned memory→direction map — the actual on-ramp back to PCDC write-back.

---

## Related

- **PCDC** — the predictive-coding sidecar that motivated this: <https://github.com/PStryder/PCDC>
- **introspection-mechanisms** — the causal read/write interface that inspired the approach: <https://github.com/safety-research/introspection-mechanisms>

## License & attribution

Code in this repository is released under the MIT License (see `LICENSE`). The model weights are **not** included and are governed by their own license (Qwen2.5 / Tongyi Qianwen). Built and analyzed with Claude Code (Anthropic).
