# Phase 3XL — Cross-Layer Semantic Portability: Design & Preregistration

Written BEFORE observing the portability matrix. Reuses the frozen Phase 3
apparatus and Phase 3LB directions. Writes only to results/phase3xl/.

## Question

Do the independently-constructed blue→green mean directions at different layers
live in a **shared** semantic coordinate system, or are they **layer-private**?
Take `D_S` (built at source layer S), inject at target layer T (`h_T + α·D_S`),
sweep the full 6×6 source×target matrix over the Phase 3LB layers
{4,8,12,15,18,24} (receptive 4–18; L24 = post-deadline dead control).

## Confirmed before running

- **Directions reproducibly constructible:** `D_mean[L]` = dir-normalized mean of
  `h_green−h_blue` at L over the frozen 14 train entities (Phase 3LB method).
  Saved to `matrices/D_mean_L{L}.pt` this run.
- **Same frozen held-out set** (6 entities: tom, sam, lamp, tree, dog, dax),
  same contrast (green vs blue), same target tokens, same specificity prompts,
  same position rule (per-prompt color-fact token).
- **Native norms (‖D_L‖):** L4 37.5, L8 34.1, L12 33.4, L15 33.1, L18 33.7,
  **L24 75.3**. → Within the receptive band norms are similar (±~7%), so raw and
  target-norm conditions will nearly coincide there; they diverge materially only
  for **D24-as-source** (norm ~2× larger) and **L24-as-target**. This is exactly
  the confound the two-condition design isolates.
- **Cross-layer cosine (pre-results):** high within band, proximity-graded
  (cos(D12,D15)=0.985, cos(D8,D12)=0.937), L4 the outlier (cos(D4,D18)=0.717);
  D24 moderately aligned to receptive (0.64–0.80).

## Minimal-correction review

The two proposed scaling conditions are well-formed. **Raw:** `h_T + α·D_S`
(‖·‖=‖D_S‖). **Target-norm-matched:** `h_T + α·(D_S/‖D_S‖)·‖D_T‖` where ‖D_T‖ is
the frozen native norm at T; identical target-norm convention for all held-out
examples, no per-example tuning. Note the diagonal (S==T) is identical under both
conditions and must reproduce Phase 3LB (**halt gate**). **No correction needed**;
I only record that raw≈target-norm for receptive×receptive cells and that the
conditions are decisive specifically for D24/L24.

## Two conditions (as specified)

- **A. Raw portability** — can the literal S-vector be read at T?
- **B. Target-norm-matched** — direction of S, magnitude of T; separates *wrong
  magnitude* from *wrong coordinates*. This is the primary *coordinate* test.

Both at **α ∈ {0.5, 1.0}**.

## Primary predeclared metric

Held-out **target (green) flip rate** + **mean P(green)** (same as Phase 3LB).
A cross-layer cell is "portable" iff held-out flip ≥ 0.8 at α≤1.0 AND (for strong
cells) specificity preserved (mean P(green) on unrelated < 0.05).

## Predeclared primary questions

1. Off-diagonal portability substantially above null?
2. Local (neighbor) vs broad (band-wide)?
3. Does target-norm scaling rescue raw failures?
4. Does cosine(D_S,D_T) predict portability? (correlation over off-diagonal cells)
5. Is portability asymmetric with depth? (`port(S→T) − port(T→S)`)
6. Can D24 (dead-site payload) work when moved INTO a receptive layer?
7. Can any receptive direction make L24 writable? (expected: no)

## Compact null controls (not a re-run of 3N)

For sources {D8, D12}, at targets {native, one non-native receptive L15, dead L24},
under target-norm scaling at α=0.5: coordinate-permutation, sign-randomization,
isotropic norm-matched noise, deterministic seeds, N=150 each. Distinguishes
semantic cross-layer compatibility from generic perturbability.

## Specificity

Run the 4 unrelated prompts on: all strong off-diagonal cells (flip≥0.8),
a representative diagonal cell, a representative null. Portable ⇒ green on
relevant prompts AND unrelated prompts substantially unchanged.

## Interpretation classes (classify, don't force)

LAYER-PRIVATE · LOCAL-PORTABILITY · BROAD-PORTABILITY · ASYMMETRIC-PORTABILITY ·
RECEPTIVITY-DOMINATED · MIXED/AMBIGUOUS. A null (layer-private) result is useful.

## Caveats (carried forward, unchanged)

Conditional on the frozen pipeline; held-out N=6; single concept (blue→green);
single task family. This maps portability *within this system*, not universally.
Per the stop point: no cross-layer transform / Procrustes / probe / PCA /
compression / PCDC / Phase 4 — document the matrix and stop.
