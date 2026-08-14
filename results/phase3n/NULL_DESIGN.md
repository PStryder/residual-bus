# Phase 3N — Null Design & Preregistration (written BEFORE running nulls)

Adversarial validation of the Phase 3C constructed-direction result. Goal: try to
**kill** the semantic interpretation with the strongest *fair* null, not confirm it.

## 1. The claim that requires a null

Phase 3C: a constructed direction `D_mean = mean(h_green − h_blue)` at layer 8,
scaled to typical ‖D‖≈34.06 and injected additively (`h + α·D_mean`) at the
color-fact token, made **held-out** prompts answer *green*: **100% flip** on 6
held-out entities at α=0.5 (mean P(green)=0.879), 100% at α=1.0; norm-matched
isotropic random = 0%; specificity intact; works at receptive L8, not post-deadline L24.

## 2. Null vs alternative (refined)

- **H0 (null):** a vector with comparable *non-semantic* statistical properties,
  injected under identical frozen conditions, is as likely as `D_mean` to produce
  the observed **target-specific** held-out behavior (green flips + specificity).
- **H1 (alternative):** the *direction/organization* of `D_mean` carries
  information that produces green-specific held-out transfer far beyond matched nulls.

## 3–4. Is isotropic norm-matched noise a fair null? (critique)

**No — it is a necessary sanity baseline but a WEAK null, and I say so up front.**
These residual streams are strongly anisotropic (massive-activation dims / attention
sinks, documented Phase 1). An isotropic unit vector is ~orthogonal to any
meaningful direction (Phase 3C: cos(random, D_mean)≈0.04, effect 0%). Beating it
proves only "D_mean beats an off-manifold poke." The scientifically decisive nulls
must be **on- or near-manifold** and **structure/energy-preserving**:

- **Coordinate-permuted `D_mean`** — preserves exact ‖·‖ and the full coordinate
  magnitude multiset; destroys *which coordinate carries what*. Strong.
- **Sign-randomized `D_mean`** — preserves per-coordinate magnitudes, flips signs.
  Caveat: under heavy anisotropy, flipping a few dominant coordinates may be either
  very destructive or land near a real axis; treated as a mid-strength null.
- **Empirical real-delta null (my nominated PRIMARY STRONG null)** — real averaged
  deltas at L8 that carry **the wrong information**: (i) *other-color* transforms
  (blue→red, →yellow, →black, …) built by the identical process, and (ii) *cross-task*
  deltas (2B arithmetic, 2C comparison). These are genuine model-state directions;
  the only thing they lack is "blue→green". If `D_mean` produces green where these
  do not, semantic content — not "being a plausible direction" — is the cause.

## 5. One R_j reused across all held-out prompts — AGREE

The claim is that **one reusable direction** generalizes. So each null trial =
**one** R_j applied to **all** held-out prompts; the trial score is the aggregate.
Independent-per-prompt noise is a *different, easier* task and is run only as a
clearly-labeled **secondary diagnostic**.

## 6. Metrics — resolution problem, and fix

`D_mean` flip rate is **saturated at 100%** on 6 prompts, so flip rate alone can't
separate `D_mean` from a hypothetical noise that also hits 100%. Therefore:

- **Primary (continuous) statistic S = mean held-out P(target=green) at α=0.5.**
  Resolution-bearing; `D_mean` value = **0.879** (frozen reference).
- **Co-primary = held-out target-flip rate at α=0.5** (the headline; `D_mean`=1.0).
- **Composite "semantic-like" statistic (preregistered):** a vector counts only if
  it BOTH (a) flips ≥ 5/6 held-out to green AND (b) keeps specificity — mean
  P(green) on the 4 unrelated prompts < 0.05 (frozen threshold; `D_mean`≈0).
  This asks the *strong* question: how often does chance produce transfer **and**
  contextual specificity together?

All chosen **before** viewing null results.

## 7–8. Selection / multiple-comparison honesty (critique)

- **No leakage in the split:** `D_mean` built on 14 train entities; evaluated on the
  frozen 6 held-out. Both `D_mean` and every R_j are scored on the *same* 6.
- **But the pipeline was selected by prior phases:** injection layer **L8** (from 2A
  cartography), gain **α** (from 3A), task family **color** (from 2A), position rule,
  and metric lineage were all chosen earlier. The null test **holds these fixed and
  does NOT correct for that forking-path selection.** The defensible claim is
  therefore *conditional*: "given this frozen pipeline, how unusual is `D_mean`'s
  performance vs matched nulls on this fixed 6-prompt held-out set." **Not** a claim
  about arbitrary prompts, layers, or tasks. Prompt-N is small (6) and there is one
  task family — external validity is narrow and stated as such.

## 9. Stronger null to add — YES (empirical real-delta null, see §3–4).

The other-color null is the most informative: each other-color delta should push
toward **its own** color, not green — simultaneously a fair strong null *and* a
specificity probe.

## 10. Null to remove / modify — the covariance-matched null (authorized correction)

The proposed **covariance-matched Gaussian null (family D) is REMOVED.**
Justification (pre-results): a 1536×1536 covariance estimated from only ~20–40
empirical deltas is severely rank-deficient (rank ≤ 40 ≪ 1536); sampling from it is
dominated by estimation artifacts and would be an *unaudittable, possibly misleading*
null (arbitrarily weak or strong). The **empirical real-delta null achieves the same
on-manifold goal** (respecting anisotropy) without fragile estimation, so it
substitutes for D. This is the only substantive change to the proposed design.

## Frozen null families (final)

| family | construction | role |
|---|---|---|
| `isotropic` | unit Gaussian × ‖D_mean‖ | required WEAK baseline, N=2000 |
| `permuted` | random coord permutation of D_mean | strong structure-preserving, N=2000 |
| `sign` | ±1 per coord × |D_mean| | mid-strength, N=2000 |
| `empirical_othercolor` | mean(h_c − h_blue), c∈{red,yellow,…}, scaled to ‖D_mean‖ | **primary STRONG** on-manifold |
| `empirical_crosstask` | 2B arithmetic & 2C comparison deltas at L8, scaled | on-manifold, unrelated semantics |

Equal injection budget throughout: every R_j scaled to **‖D_mean‖=34.057**, injected
at **L8**, position = each prompt's color-fact token, **α ∈ {0.5 (primary), 1.0}**,
identical forward path. No per-trial tuning.

## Meaningful non-null controls (evaluated separately, NOT in the noise distribution)

- `−D_mean` (negated semantic direction) — semantically meaningful, not chance.
- `D_mean` at **dead site L24** — anatomical gating, not chance.
- other-color deltas' effect on **their own** target — mechanism-generality check.

## Secondary diagnostics (labeled)

- independent-per-prompt isotropic noise (different task than reuse).
- cosine(R_j, D_mean) and cosine(R_j, held-out exact D) vs performance — does
  *accidental* success require *accidental alignment*? (analyzed after, never used to filter.)
- anatomical 2×2 square (semantic/noise × receptive/dead), noise at dead site subsampled.
- cross-phrasing null (held-out `fact` template, frozen from 3C: partial@0.5, strong@1.0).

## Empirical p-value & reporting

Primary S = mean held-out P(green)@α0.5. `S_sem = 0.879` (frozen). For each family,
`k = #{S_j ≥ S_sem}`, `p = (k+1)/(N+1)` (add-one; never "p=0"). Report null
mean/median/sd, quantiles {50,90,95,99,99.9}%, max, and `S_sem` percentile — **per
family**. Also flip-rate p and composite rate. Empirical distribution IS the result;
no normality assumed.

## Reproduce-or-halt gate

Load `D_mean` from `results/phase3c/D_mean_stage1.pt` (sha1 `b9b7ce81…`), rebuild the
frozen 6 held-out pairs, re-evaluate `D_mean@0.5`. If flip rate ≠ 1.0 or mean
P(green) not ≈0.879 (±0.02), **HALT** (config drift) before any null runs.

## Interpretation classes (preregistered)

- **STRONG REJECTION:** S_sem beyond all fair-null tails (empirical p ≤ ~1/(N+1)),
  including the empirical real-delta null; composite rarely/never reproduced;
  specificity + site-gating intact.
- **MODERATE REJECTION:** S_sem significantly exceeds nulls but some family produces
  nontrivial target-specific behavior.
- **NULL NOT REJECTED:** matched (esp. empirical) nulls frequently reproduce
  comparable target-specific held-out transfer.
- **AMBIGUOUS:** null demonstrably unfair / N inadequate / selection issues dominate.
- **NEGATIVE-ARTIFACT:** a stronger fair null explains most/all of the Phase 3 effect.

A null/negative result will be reported plainly — that is this phase's purpose.
