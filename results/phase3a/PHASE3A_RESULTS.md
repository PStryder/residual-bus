# Phase 3A — Exact Additive Semantic Write-Back (Results)

**Model:** F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct · FP32 · eager · frozen · single forward pass. carto_lib reused unchanged; tasks reused from Phase 2A/2C.

**Intervention:** `h <- h + alpha*D`, `D = h_B - h_A`, at (layer, P_source). Additive (not replacement) — the PCDC-shaped write.

**Primary task (2A copy):** blue→green, P_source=20, answers `blue`/`green`, S_natural=28.91.

## Verdict: **PASS**

| gate | result |
|---|---|
| gate_sanity_alpha1_reproduces_replacement | True |
| gate_semantic_beats_random | True |
| gate_dose_response_coherent | True |
| gate_receptive_beats_postdeadline | True |
| gate_subreplacement_regime(5a_or_5b) | True |
| subreplacement_flip_alpha(5a) | 0.5 |
| semantic_max_targetprob | 0.9999548797012192 |
| random_max_targetprob | 0.00029543449849407714 |
| recommended_site_L | 8 |
| recommended_alpha | 0.5 |

> **Oracle gate metric (corrected):** argmax_is_target + P(target)>=0.5 (strong); never-flips + P(target)<0.10 + KL<0.5 (post-deadline). Corrected from transfer_fraction (Phase 2C: contrast drifts under perturbation, not decisive). Post-deadline L24: never flips any α = True, P(target)@α1 = 0.001, KL@α1 = 0.00.


## alpha=1 algebraic sanity (L8)

- additive α=1 vs exact replacement: logit cosine **1.000000**, KL **3.44e-14**, argmax equal **True**, site-state cosine **1.000000**. (Confirms `h_A + (h_B-h_A) = h_B`.)

## Dose-response (strong site L8, A→B, semantic)

| alpha | argmax | flip→target | P(target) | transfer | frac | cos→h_B |
|---|---|---|---|---|---|---|
| 0.00 | `blue` | False | 0.000 | 0.00 | 0.00 | 0.734 |
| 0.10 | `blue` | False | 0.000 | 1.67 | 0.06 | 0.780 |
| 0.25 | `blue` | False | 0.000 | 5.34 | 0.18 | 0.845 |
| 0.50 | `green` | True | 0.645 | 15.10 | 0.52 | 0.933 |
| 0.75 | `green` | True | 1.000 | 24.33 | 0.84 | 0.984 |
| 1.00 | `green` | True | 1.000 | 29.05 | 1.00 | 1.000 |
| 1.25 | `green` | True | 1.000 | 31.09 | 1.08 | 0.988 |
| 1.50 | `green` | True | 1.000 | 31.75 | 1.10 | 0.960 |
| 2.00 | `green` | True | 1.000 | 30.88 | 1.07 | 0.887 |

## Cartography as oracle (same semantic D, α=1)

| site | role | flip→target | transfer | frac |
|---|---|---|---|---|
| L8 | strong | True | 29.05 | 1.00 |
| L21 | near_deadline | True | 26.56 | 0.92 |
| L24 | post_deadline | False | 6.94 | 0.24 |

## Secondary corroboration (2C comparison): PASS

S_natural=28.97; same additive form tested at a different-topology task (source deadline ~L11). See phase3a_summary.json.


## Report answers (3A)

1. **α=1 reproduces replacement?** Yes (logit cosine 1.00000, KL 3.4e-14).
2. **Coherent dose-response?** True.
3. **Useful sub-replacement α?** flip at α=0.5; big-fraction<1 exists=True.
4. **Semantic beats random?** True (max P(target): semantic 1.000 vs random 0.000).
5. **Receptive site beats post-deadline?** True.
6. **Best predeclared candidate for 3C:** site L8, alpha=0.5.
7. **Surprises:** see dose-response table (where the flip occurs relative to α=1).
8. **Classification:** **PASS** → advance to 3C = True.

## Reproduce
```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe phase3_additive.py
```

**STOP unless PASS.** Only a PASS verdict advances to Phase 3C.
