# Phase 3N — Empirical Null Test of Constructed Write-Back (Results)

**Model:** F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct · frozen Phase 3 config. See NULL_DESIGN.md (preregistration).

**Frozen:** inject L8, α=0.5 primary, D_mean sha1 `b9b7ce81f5e38110` norm 34.06, held-out ['tom', 'sam', 'lamp', 'tree', 'dog', 'dax'].

**Primary statistic S = mean held-out P(green) @α0.5. Observed D_mean S = 0.879** (flip 1.00, specificity meanP(green) 0.0000).

## Verdict: **STRONG_REJECTION**

| null family | role | mean S | q99 | max S | k≥S_sem | empirical p=(k+1)/(N+1) |
|---|---|---|---|---|---|---|
| isotropic | weak baseline | 0.0000 | 0.0000 | 0.0000 | 0 | **5.00e-04** (N=2000) |
| permuted | structure-preserving | 0.0000 | 0.0000 | 0.0000 | 0 | **5.00e-04** (N=2000) |
| sign | sign-flip | 0.0000 | 0.0000 | 0.0002 | 0 | **9.99e-04** (N=1000) |

## Empirical STRONG null (real wrong-info deltas)

| delta | held-out green flip | green S | specificity meanP(green) | (own-COLOR flip: does the wrong-color delta steer to ITS color?) |
|---|---|---|---|---|
| othercolor:red | 0.00 | 0.000 | 0.000 | 1.00 |
| othercolor:yellow | 0.00 | 0.000 | 0.000 | 0.17 |
| othercolor:black | 0.00 | 0.000 | 0.000 | 0.50 |
| othercolor:white | 0.00 | 0.000 | 0.000 | 0.83 |
| othercolor:brown | 0.00 | 0.000 | 0.000 | 0.00 |
| othercolor:purple | 0.00 | 0.000 | 0.000 | 0.17 |
| othercolor:orange | 0.00 | 0.000 | 0.000 | 0.17 |
| othercolor:pink | 0.00 | 0.000 | 0.000 | 0.00 |
| othercolor:gray | 0.00 | 0.000 | 0.000 | 0.00 |
| othercolor:gold | 0.00 | 0.000 | 0.000 | 0.00 |
| crosstask:arithmetic_2B | 0.00 | 0.000 | 0.000 | — |
| crosstask:comparison_2C | 0.00 | 0.000 | 0.000 | — |

Empirical max green-S = 0.00012 (essentially 0), #≥S_sem = 0, other-color OWN-color mean flip = 0.28, **max GREEN flip by any wrong-color delta = 0.00**.

> **Interpretation of the strong null.** Not one of the 10 other-color deltas nor the 2 cross-task deltas produced *any* green on held-out prompts (green-S = 0.000 for all), **despite cos(delta, D_mean) ≈ 0.40–0.52** for the color deltas (they share a color-attribute subspace). Meanwhile each wrong-color delta steers held-out prompts toward **its own** color (red→red 1.00, white→white 0.83, black→black 0.50; weaker colors lower, mean 0.28). So the mechanism is general — a real `blue→X` delta writes X — and `D_mean` is specifically the *green* direction. Being a real, ~half-aligned color direction is **not** sufficient to produce green; only the green direction does. This is the decisive on-manifold refutation of "any plausible color-ish vector would work."

## Composite (transfer ≥5/6 AND specificity <0.05) reproduced by noise

| family | count / N |
|---|---|
| isotropic | 0 / 2000 |
| permuted | 0 / 2000 |
| sign | 0 / 1000 |

## Anatomical square (mean held-out P(green))

| | receptive L8 | dead L24 |
|---|---|---|
| **semantic** | 0.879 | 0.000 |
| **noise (mean)** | 0.000 | 0.000 |

## Non-null directional controls

- `neg_Dmean@0.5`: flip 0.00, S 0.000
- `Dmean@deadL24@0.5`: flip 0.00, S 0.000
- `Dmean@1.0`: flip 1.00, S 1.000

## Cross-phrasing null (frozen 3C stage2)

- alpha0.5: semantic flip 0.12 S 0.236 | isotropic p=1.25e-03, permuted p=1.25e-03
- alpha1.0: semantic flip 1.00 S 1.000 | isotropic p=1.25e-03, permuted p=1.25e-03

## Answers

1. **Null:** a matched non-semantic vector, same frozen conditions, is as likely as D_mean to produce target-specific held-out green transfer.
2. **Modified design?** Yes — isotropic demoted to weak baseline; empirical real-delta null added as PRIMARY strong null; fragile covariance-matched null removed (rank-deficient with ~N samples). See NULL_DESIGN.md.
3. **Fairest/strongest null:** the empirical other-color / cross-task deltas (on-manifold, real, wrong information).
4. **Isotropic meaningful?** Weak sanity check only (off-manifold). p_iso=5.00e-04.
5–6. **Where does D_mean fall / p-value:** S_sem=0.879; empirical p — iso 5.00e-04, permuted 5.00e-04, sign 9.99e-04.
7–9. **Noise ANY vs TARGET flip / generalization:** noise mean held-out green-S (receptive) = 0.000; composite semantic-like reproduced by noise = 0.
10. **Chance reproduce transfer AND specificity?** no.
11. **Shuffled/sign vectors retain effect?** permuted mean S=0.0000, sign mean S=0.0000.
12. **Unrelated real directions produce target?** empirical max green-S=0.00011954258537002258, #≥S_sem=0; wrong-color deltas steer to their OWN target (mean flip 0.2833333333333333).
13. **Accidental success ~ cosine alignment?** see plots/cosine_vs_S.png.
14. **Site gating survives?** semantic L8 0.88 vs dead L24 0.00; noise L8 0.00. Gating intact=True.
15. **Cross-phrasing extreme vs null?** see cross-phrasing section.
16. **Selection caveats:** L8/α/task frozen from prior phases (uncorrected forking path); held-out N=6; one task family. p-value is CONDITIONAL on this pipeline; external validity narrow (stated in NULL_DESIGN.md §7-8).
17. **Classification: STRONG_REJECTION.**
18. **Surviving claims:** additive direction produces target-specific, site-gated, specific held-out transfer far beyond matched nulls (subject to §16 caveats).
19. **Weakened/withdrawn:** any implicit universality — result is conditional on the frozen pipeline and small held-out set; magnitude/gain dependence remains.
20. STOP.

## Reproduce
```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe phase3_null.py
```

**STRICT STOP after Phase 3N.** No compression/PCA/probe/codec/PCDC/Phase 4.
