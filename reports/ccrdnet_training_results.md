# AgriCCRDNet-v0 baseline training on the CCRDNet cotton dataset

Run `ccrd_a1_b3_256_ce_seed42` — spec Task 4 (paper-style baseline, spec §39).
Trained and evaluated 2026-08-31.

## Data

- Zenodo 15194034 (CC-BY-4.0), downloaded 2026-08-30, md5-verified
  (`acdb89f12ad86794183b447f02d1dd62`), audited in
  [cotton_dataset_audit.md](cotton_dataset_audit.md).
- Conversion: black→OTHER, white(vegetation)→STRUCTURE, red(central crop
  row)→NAV_BAND via `tools/build_navigation_targets.py`; ~17 px band at
  640×480, kept as annotated.
- Split: official train 400 → train 320 / val 80 (contiguous last-id block, no
  random frame leakage), official test 400 untouched.

## Setup

| item | value |
|---|---|
| model | `perception/ccrdnet` reproduction, DSC + ASPP(2/4/6/8, last 3 skips) |
| scale | 33,694 params / 37.61 M MACs @256 (paper: 33,621 / 38.226 M) |
| input | 640×480 → stretch 256×256 RGB (paper baseline P0) |
| loss / optim | CrossEntropy / Adam 2e-4, batch 4 (paper) |
| augmentation | hflip, brightness/contrast, gamma, mild blur |
| schedule | max 300 epochs, early stop on val line-IoU (patience 40) → stopped at 92, best epoch 52 |
| hardware | RTX 4070 SUPER (`.venv-train`, torch 2.13.0+cu126), ~2 s/epoch |
| artifacts | `runs/ccrd_a1_b3_256_ce_seed42/` (config, metrics.csv, best/last.pt) |

## Test results (official 400-frame test split, best.pt @ epoch 52)

Postprocess = paper baseline: largest connected component + least-squares.
Line IoU rendered at 6 px width @256 (≈ the annotated band width after resize).

| metric | mean | median | P95 | paper (their defs) |
|---|---:|---:|---:|---:|
| angle error (deg) | **0.780** | 0.400 | 3.055 | 1.13 |
| lateral error near y=0.8H (px @256) | 1.79 | 0.87 | 6.59 | — |
| lateral error far y=0.45H (px @256) | 0.96 | 0.59 | 3.32 | — |
| line IoU | 0.726 | 0.756 | — | 0.7837 |
| NAV_BAND mask IoU | 0.676 | 0.738 | — | — |

- detection rate: **400/400** (a line was always produced)
- line accuracy (AE≤5° AND lateral_near≤10 px): **97.5 %** (paper 95.57 %,
  different criterion)

Val-split numbers are notably worse (best val line IoU 0.417, AE ~2.2°): the
contiguous val block is drawn from the hardest tail of the train recordings —
one more reason not to trust random-split numbers.

## Reading

- The reproduction reaches the paper's reported navigation quality on the
  official test split (AE even lower; line IoU 0.726 vs 0.784 with our
  narrower 6 px rendering — the metric is very sensitive to band width).
- Failure cases (`ccrd_a1_b3_256_ce_seed42/failure_cases/`) are dominated by
  **dense canopy with an angled central row**: prediction stays near-vertical
  while GT leans, up to ~7.8° AE / ~23 px lateral. This mirrors the paper's own
  dense-row difficulty.
- Warning per spec §16: those failures keep confidence ≈ 0.7 —
  **high-confidence wrong lines exist**. Confidence calibration work (span/fit
  scores alone are not enough) is needed before control use.
- Nothing here validates the Agri-X semantics: this dataset supervises the
  crop-row *center*, not the traversable furrow. Camera V2 evaluation
  (spec Task 6) remains the real gate.

## Files

```text
reports/ccrd_a1_b3_256_ce_seed42/
├── metrics.json            aggregate test metrics
├── metrics_frames.csv      per-frame AE / lateral / IoU / confidence
├── summary.md              auto-generated summary + verdict
├── overlays/               24 sampled test overlays (GT green, pred red)
├── failure_cases/          16 worst frames
├── preview_grid.jpg        8-tile quick look
└── failures_grid.jpg       4 worst dense-canopy failures
runs/ccrd_a1_b3_256_ce_seed42/   (gitignored) checkpoints, curves, config
```

## Next steps (spec order)

1. Ablations A/B/C (2-class vs 3-class, DSC/ASPP on-off, component selection,
   LS vs TLS) — spec Task 5.
2. Fixed Camera V2 eval set + furrow-semantics target rebuild — spec Task 6.
3. ONNX export + Pi 4 latency benchmark — spec Task 7.

**Verdict: KEEP** — the CCRDNet reproduction learns the correct corridor on
held-out cotton data at paper-level navigation accuracy; proceed to ablations
and real-domain evaluation.

## CRDLD multi-furrow experiment (crdld_furrow_v1)

See ../crdld_furrow_v1/: CRDLD v2.1 (all rows labeled as lines) converted to
STRUCTURE=all row lines + NAV_BAND=furrow midline bands between adjacent rows
(tools/build_crdld_targets.py). Trained same architecture, weighted CE 1,4,4,
best @116. Test (430 frames, bottom-center furrow selection):
median AE 1.26 deg / median lateral 3.4 px / line accuracy 69%, det 100%.
Hard-shadow, stubble, dense-weed frames self-report conf<=0.08 (gated out).
