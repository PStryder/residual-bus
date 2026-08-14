# Phase 3XL — Cross-Layer Semantic Portability (Results)

**Model:** F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct · frozen Phase 3 apparatus. See DESIGN.md (preregistration).

**Anchor:** diagonal L8 reproduces flip 1.00, S 0.879.

## Verdict: **BROAD-PORTABILITY (receptivity-dominated)**


**Held-out flip — targetnorm, α=1.0** (rows=source D_S, cols=target L_T):

| S\T | L4 | L8 | L12 | L15 | L18 | L24 |
|---|---|---|---|---|---|---|
| **D4** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D8** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D12** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D15** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D18** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D24** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

**Held-out flip — targetnorm, α=0.5** (rows=source D_S, cols=target L_T):

| S\T | L4 | L8 | L12 | L15 | L18 | L24 |
|---|---|---|---|---|---|---|
| **D4** | 0.67 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 |
| **D8** | 1.00 | 1.00 | 0.83 | 0.67 | 0.83 | 0.00 |
| **D12** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D15** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D18** | 1.00 | 1.00 | 0.67 | 0.67 | 1.00 | 0.00 |
| **D24** | 1.00 | 1.00 | 0.17 | 0.33 | 0.67 | 0.00 |

**Held-out flip — raw, α=1.0** (rows=source D_S, cols=target L_T):

| S\T | L4 | L8 | L12 | L15 | L18 | L24 |
|---|---|---|---|---|---|---|
| **D4** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D8** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D12** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D15** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D18** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| **D24** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

## Key findings

- **Receptive off-diagonal portability** (targetnorm α1): 100% of 20 cross-receptive cells reach flip≥0.8.
- **Target-norm rescue of raw failures** (receptive off-diag): 0 cells.
- **L24 as target accepts any receptive direction?** False (into-L24 targetnorm flips: {'D4->L24': 0.0, 'D8->L24': 0.0, 'D12->L24': 0.0, 'D15->L24': 0.0, 'D18->L24': 0.0}).
- **D24 (dead-site payload) works at a receptive layer?** True (D24→receptive targetnorm flips: {'D24->L4': 1.0, 'D24->L8': 1.0, 'D24->L12': 1.0, 'D24->L15': 1.0, 'D24->L18': 1.0}).
- **cosine ↔ portability correlation** (off-diagonal): 0.2769184969439378.
- **max portability asymmetry** |flip(S→T)−flip(T→S)|: 1.00.
- **cross-layer null** max flip (permute/sign/isotropic): 0.00.

## Specificity on strong off-diagonal cells

| S→T | spec meanP(green) | specific? |
|---|---|---|
| D4→L8 | 0.000 | True |
| D4→L12 | 0.000 | True |
| D4→L15 | 0.000 | True |
| D4→L18 | 0.000 | True |
| D8→L4 | 0.000 | True |
| D8→L12 | 0.000 | True |
| D8→L15 | 0.000 | True |
| D8→L18 | 0.000 | True |
| D12→L4 | 0.000 | True |
| D12→L8 | 0.000 | True |
| D12→L15 | 0.000 | True |
| D12→L18 | 0.000 | True |
| D15→L4 | 0.000 | True |
| D15→L8 | 0.000 | True |
| D15→L12 | 0.000 | True |
| D15→L18 | 0.000 | True |
| D18→L4 | 0.000 | True |
| D18→L8 | 0.000 | True |
| D18→L12 | 0.000 | True |
| D18→L15 | 0.000 | True |
| D24→L4 | 0.000 | True |
| D24→L8 | 0.000 | True |
| D24→L12 | 0.000 | True |
| D24→L15 | 0.000 | True |
| D24→L18 | 0.000 | True |

(diagonal L8 spec meanP(green) = 0.000)


## Native norms & cosine

norms: L4=37.5, L8=34.1, L12=33.4, L15=33.1, L18=33.7, L24=75.3

## Answers

1. **Diagonal reproduces 3LB?** Yes (L8 anchor flip 1.00, S 0.879).
2. **Move directions between receptive layers?** 100% of cross-receptive cells portable (targetnorm α1).
3. **Falloff with layer distance?** see plots/distance_vs_portability.png.
4. **Target-norm rescues raw?** 0 raw failures rescued.
5. **Geometric similarity D4..D24?** see cosine matrix (high within band, L4/L24 outliers).
6. **Cosine predicts write-back?** correlation 0.2769184969439378.
7. **Symmetric?** max asymmetry 1.00.
8. **Early→late vs reverse?** see asymmetry matrix.
9. **D24 works at a receptive site?** True.
10. **Any early direction makes L24 writable?** False.
11. **Null reproduces cross-layer transfer?** max null flip 0.00 (no).
12. **Strong off-diagonal cells context-specific?** see specificity table.
13. **Best interpretation:** BROAD-PORTABILITY (receptivity-dominated).
14. **Codec implication:** a shared receptive-band basis exists → one direction may serve multiple layers (fewer per-layer codecs).
15. **Does NOT establish:** universality (frozen pipeline, N=6, single concept/task); no learned transform tested.
16. STOP.

## Reproduce
```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe phase3_xlayer.py
```

**STRICT STOP.** No cross-layer transform/Procrustes/probe/PCA/compression/PCDC/Phase 4.
