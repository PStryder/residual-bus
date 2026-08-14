# Phase 3LB — Layer-Band Robustness of Constructed Write-Back (Results)

**Model:** F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct · frozen Phase 3 config (same train/held-out split, α, controls).

**Purpose:** test whether the Phase 3 write-back effect is specific to the *selected* layer L8 (forking-path caveat) or holds across the color task's cartographic receptive band.

**Anchor:** D_mean rebuilt at L8 reproduces flip 1.00, S 0.879.

## Verdict: **ROBUST**  (receptive layers passing, excl. L8: [4, 12, 15, 18])

| layer | role | flip@0.5 | flip@1.0 | S@0.5 | spec@0.5 | null max flip | null p | cos→exact |
|---|---|---|---|---|---|---|---|---|
| L4 | receptive | 0.67 | 1.00 | 0.719 | 0.000 | 0.00 | 2.00e-03 | 0.964 |
| L8 | receptive | 1.00 | 1.00 | 0.879 | 0.000 | 0.00 | 2.00e-03 | 0.898 |
| L12 | receptive | 1.00 | 1.00 | 0.783 | 0.000 | 0.00 | 2.00e-03 | 0.898 |
| L15 | receptive | 1.00 | 1.00 | 0.752 | 0.000 | 0.00 | 2.00e-03 | 0.894 |
| L18 | receptive | 1.00 | 1.00 | 0.766 | 0.000 | 0.00 | 2.00e-03 | 0.893 |
| L24 | post_deadline | 0.00 | 0.00 | 0.000 | 0.000 | 0.00 | 2.00e-03 | 0.902 |

**Criterion (predeclared):** L8 not a cherry-pick if >=2 other receptive layers give held-out flip>=0.8 @a<=1.0, null flip~0, spec<0.05; dead layer fails


## Reading

- Receptive-band layers where the constructed direction generalizes (excl. L8): **[4, 12, 15, 18]** (4 of 4).
- Post-deadline L24 fails as predicted: True.
- Isotropic null at every layer stays at ~0 flip (equal injection budget), so the per-layer effect is not chance.
- Interpretation: the L8 result is **not** a cherry-pick — the write-back primitive works across the receptive band, exactly where the cartography said it should, and fails at the post-deadline site.

## Plots
- `plots/flip_vs_layer.png` — semantic vs null flip rate across layers.
- `plots/S_and_cos_vs_layer.png` — mean P(green) and D_mean↔exact alignment across layers.

## Reproduce
```powershell
$env:RESIDUAL_BUS_MODEL="F:/HexyLab/residual-bus/models/Qwen2.5-1.5B-Instruct"
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe phase3_layerband.py
```

**STRICT STOP.** Robustness check only. No compression/probe/codec/PCDC/Phase 4.
