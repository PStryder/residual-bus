# Phase 3C — Held-Out Generalization of a Constructed Direction (Results)

**Model:** F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct · FP32 · frozen · single forward pass. carto_lib reused unchanged.

**Policy (fixed from 3A):** inject at layer **L8**, additive `h + alpha*D_mean`, primary **alpha=0.5**, reference alpha=1.0. No per-example tuning.

**D_mean** built from TRAIN pairs' `h_green - h_blue` at L8; injected into HELD-OUT prompts. No training, no probes.

## Stage 1 — within-template, held-out ENTITIES

- 20 validated blue→green pairs; train 14 / held-out test 6.
- train entities: ['wug', 'cat', 'ring', 'book', 'hat', 'cup', 'star', 'fish', 'pen', 'car', 'box', 'leaf', 'coin', 'key']; **held-out**: ['tom', 'sam', 'lamp', 'tree', 'dog', 'dax'].
- D_mean cos(dir vs raw)=1.000, mean individual ‖D‖=34.1, raw mean ‖D‖=31.1.

| classification | primary α=0.5 | reference α=1.0 |
|---|---|---|
| **within-template** | **STRONG_POSITIVE** | STRONG_POSITIVE |

**Held-out flip rate by condition** (α=0.5 unless α=1.0 shown):

| condition | flip rate | mean P(green) | median transfer frac | mean cos→exact D |
|---|---|---|---|---|
| baseline@0.0 | 0.00 | 0.000 | 0.00 | 0.898 |
| random1@0.5 | 0.00 | 0.000 | 0.03 | 0.040 |
| unrelated_single_D@0.5 | 0.83 | 0.627 | 0.50 | 0.863 |
| D_mean_raw@0.5 | 0.33 | 0.504 | 0.48 | 0.898 |
| D_mean_dir@0.5 | 1.00 | 0.879 | 0.55 | 0.898 |
| D_mean_dir@1.0 | 1.00 | 1.000 | 1.03 | 0.898 |
| oracle_exact_D@0.5 | 0.67 | 0.611 | 0.52 | 1.000 |
| oracle_exact_D@1.0 | 1.00 | 1.000 | 1.00 | 1.000 |

## Specificity (green-direction injected into UNRELATED prompts)

| probe@α | base argmax | injected argmax | changed | P(green) base→inj | KL |
|---|---|---|---|---|---|
| arithmetic@0.5 | Yes | Yes | False | 0.000→0.000 | 0.00 |
| arithmetic@1.0 | Yes | Yes | False | 0.000→0.000 | 0.00 |
| capital@0.5 | par | par | False | 0.000→0.000 | 0.00 |
| capital@1.0 | par | par | False | 0.000→0.000 | 0.00 |
| comparison@0.5 | Yes | Yes | False | 0.000→0.000 | 0.00 |
| comparison@1.0 | Yes | Yes | False | 0.000→0.000 | 0.00 |
| unrelated_color@0.5 | red | red | False | 0.000→0.000 | 0.00 |
| unrelated_color@1.0 | red | red | False | 0.000→0.000 | 0.00 |

## Site portability (same D_mean, strong L8 vs post-deadline L24)

| alpha | L8 flip rate | L24 flip rate |
|---|---|---|
| 0.5 | 1.00 | 0.00 |
| 1.0 | 1.00 | 0.00 |

## Stage 2 — cross-phrasing, held-out TEMPLATES

- held-out templates: ['fact']; train templates: ['painted'].
- **cross-template classification:** α=0.5 **PARTIAL_POSITIVE**, α=1.0 STRONG_POSITIVE.
  - D_mean_dir@0.5: flip 0.25, P(green) 0.234, cos→exact 0.834
  - D_mean_dir@1.0: flip 1.00, P(green) 1.000, cos→exact 0.834
  - random1@0.5: flip 0.00, P(green) 0.000, cos→exact 0.038
  - oracle_exact_D@1.0: flip 1.00, P(green) 1.000, cos→exact 1.000

## Answers (3C)

9. **Generalizes to held-out entities?** within-template = STRONG_POSITIVE (D_mean flip 1.00 vs random 0.00).
10. **Generalizes to held-out phrasings?** cross-template = ('PARTIAL_POSITIVE', 'STRONG_POSITIVE').
11. **Beats random?** flip 1.00 vs 0.00.
12. **Beats unrelated single delta?** 1.00 vs 0.83.
13. **How close to oracle?** oracle flip 0.67; mean cos(D_mean,exact)=0.898.
14. **Target-specific?** see specificity table (P(green) movement on unrelated prompts).
15. **Leaves unrelated intact?** see specificity argmax-changed / KL.
16. **Cartography predicts site?** see portability (L8 vs L24 flip rate).
17. **Reusable object best described as:** context-general (within task) — see stage1/stage2.
18. **Implication for a memory→latent translator:** a fixed averaged direction DOES steer unseen contexts → a reusable latent write primitive is plausible.
19. **Unsupported:** anything beyond the tested task/scale; compression/probe/learned routes (not run).
20. **Phase 4 (for discussion):** compression/low-rank of D_mean, then a learned memory→direction map.

## Reproduce
```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe phase3_generalization.py
```

**STRICT STOP after Phase 3C.** No compression/PCA/probe/learned map/PCDC — later decisions.
