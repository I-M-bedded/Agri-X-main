# cotton_pilot (47-frame RealSense) evaluation & fine-tuning

Evaluated 2026-08-31 with the cotton-trained baseline
(`runs/ccrd_a1_b3_256_ce_seed42`), then fine-tuned per instruction because the
zero-shot result was poor.

## Data

`data/cotton_pilot/trajectory{01,02,03}` — 47 RealSense colour frames
(640×480, 2026-05-13, three recordings), YOLO-seg polygons. Only two classes
are actually used: `hilera` (row/ridge strips) and `maleza` (weeds); the
declared `surco`/`suelo`/`HILERA` classes have zero instances. **This is the
Agri-X driving geometry**: camera travels inside the furrow, sky fills the top
~45 %, early-season mostly-bare ridges, low sun / harsh shadows.

### Derived targets (`tools/build_pilot_targets.py`)

No furrow or central-row class exists, so NAV_BAND was derived: walking rows
bottom-up, the corridor centre is tracked as the midpoint between the nearest
hilera edges left/right of the running centre; a 17 px band is drawn along it.
hilera→STRUCTURE, maleza→OTHER. 5/47 frames yield no band (headland, farm
road, one black underexposed frame) and are excluded by construction.

**GT caveat:** trajectory01/02 label only 2–4 of the visible rows, so the
derived corridor sometimes brackets an unlabeled ridge or selects the furrow
adjacent to the true camera furrow. GT is consistent per trajectory but not
guaranteed to be *the* driven furrow. Montages: `data/cotton_pilot/prepared/targets_all_{0,1}.jpg`.

Splits: fine-tune train = traj01+03 minus last 3 each (28), val = those 6,
held-out test = **trajectory02 (13 frames, never trained on)**.

## Results (held-out trajectory02, largest-CC + LS postprocess)

| model | AE mean° | AE med° | lat near px | line IoU | acc(AE≤5°∧lat≤10px) | det |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot cotton baseline (47 frames) | 20.9 | 21.3 | 23.0 | 0.040 | 0.00 | 0.83 |
| ft v1: CE, lr 1e-4, patience 30 | — collapsed: predicts no NAV_BAND (val IoU 0.05) | | | | | |
| ft v2: **weighted CE 1,1,8**, lr 2e-4, 430 ep | 10.7 | **4.9** | 13.1 | 0.195 | **0.38** | 1.00 |
| ft v3: v2 + rotation/shift augment, 590 ep | 9.3 | 6.2 | 12.0 | 0.188 | 0.38 | 1.00 |

- Zero-shot transfer fails completely — expected: unseen domain (sky, bare
  ridges, low sun) **and** flipped semantics (cotton target = crop-row centre,
  pilot target = furrow corridor).
- v1 collapse cause: NAV_BAND is ~0.8 % of pixels here (≈⅓ of cotton) and only
  7 iterations/epoch; unweighted CE drops the class entirely. Weighted CE fixed
  it; that is why the deviation from the paper's plain CE is kept.
- v2 vs v3 differences are noise at n=13; postprocess variants
  (bottom-centre/scored, TLS) were also swept and do not beat the baseline
  postprocess. **v2 is the reported checkpoint** (`runs/pilot_ft_v2/best.pt`).

### The useful pattern: confidence separates the failures

Per-frame (v2, test): 7 frames have AE ≤ 5.2°, lateral ≤ 11 px; the 4 gross
failures (AE 20–38°, adjacent/mirrored corridor) all have confidence ≤ 0.34
while good frames sit at 0.40–0.70. Gating at **confidence ≥ 0.4 accepts 7/13
frames, all of them good** (max AE 5.2°). Unlike the cotton dense-canopy
failures, confidence is informative in this domain — a temporal filter (spec
§21) plus this gate is the plausible path to usable guidance.

## Reading

1. 47 frames (28 trainable) is simply not enough to learn corridor selection;
   the dominant failure is wrong-corridor lean, and part of the measured error
   is **GT ambiguity from sparsely labeled rows**, not model error.
2. Fine-tuning recovers the geometry quickly (median AE ≈ 5° from a 0 %-accuracy
   start), which supports the transfer-learning route.
3. Overlays: `reports/pilot_ft_v2/overlays/` + `test_grid.jpg`;
   zero-shot: `reports/pilot_zeroshot/`.

**Verdict: INCONCLUSIVE** — promising (median AE ~5°, confidence-gated subset
clean) but the pilot set is too small and its GT too ambiguous for acceptance.
Before more model work: label *all* rows (or at least both rows flanking the
driven furrow) in every frame, add frames (target 150+, spec §26), and prefer
an explicit furrow/`surco` polygon so NAV_BAND needs no derivation.
