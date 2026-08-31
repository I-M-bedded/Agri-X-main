# CCRDNet (cotton-row) dataset audit — Zenodo 15194034

Spec Task 1 deliverable (spec §4, §36). Audited 2026-08-30 before any training.

## Source

- **Record:** https://zenodo.org/records/15194034 — "CCRDNet: dataset and video results"
  (companion dataset of Zheng & Wang, Frontiers in Plant Science 2026,
  doi:10.3389/fpls.2026.1744637)
- **License:** CC-BY-4.0
- **Archive:** single zip, 6,402,470,283 bytes, md5 `acdb89f12ad86794183b447f02d1dd62`
  (verified after download)

## Contents

```text
dataset/original_rgb      7,367 frames 640x480 PNG (full video-frame pool)
dataset/original_label    7,367 colour labels
dataset/original_binary   7,367 binary vegetation masks
dataset/train/{rgb,line,binary}   400 frames each  (official train split)
dataset/test/{rgb,line,binary}    400 frames each  (official test split)
video_results/            4 result MP4s (paper demos, not training data)
```

Train and test ids do not overlap. The 800 official frames are the annotated
subset used for the paper's experiments; `original_*` is the larger raw pool.

## Annotation format (verified pixel-exact, not inferred from filenames)

`line/` (and `original_label/`) are 8-bit BGR PNGs with **exactly three
colours** — 0 off-palette pixels across all 800 official frames:

| colour (BGR) | meaning | Agri-X class |
|---|---|---|
| (0,0,0) black | background / soil | 0 OTHER |
| (255,255,255) white | vegetation (all crop rows) | 1 STRUCTURE |
| (0,0,255) red | **central crop row band** | 2 NAV_BAND |

- The red band is a constant-width painted line: median width ≈ **17 px at
  640×480** (paper says 15 px; measured 16–17), spanning nearly the full image
  height in most frames.
- 100 % of red pixels lie inside the binary vegetation mask — the target is
  drawn **on the central crop row**, not in the inter-row furrow.
- Every one of the 800 frames contains a NAV_BAND region.
- `binary/` is a channel-identical white-on-black mask of all vegetation
  (central row included); redundant with the colour label for our purposes.

## Semantics warning (spec §1, §44)

The supervised target is the **crop-row center**, i.e. the row the machine
straddles in the paper's scenario. Agri-X drives *between* ridges, so this
dataset trains the reproduction baseline only; the learned prior must be
re-validated (and the target re-defined as the traversable furrow) on real
Camera V2 data before any runtime use. This audit does **not** certify
STRUCTURE as a collision mask.

## Diversity observed in the 36-frame montage (`cotton_dataset_audit/samples_overlay.jpg`)

- growth stages from bare seedlings to dense canopy
- soil states: dry, tilled, wet, plastic-mulch strips with standing water
- strong photographer shadow in many frames (camera held by a walking person)
- lighting from overcast to harsh sun; some near-monochrome soil scenes
- viewpoint is forward-looking and roughly row-aligned throughout; no headland
  or row-entry views seen

## Split / leakage

Official split is 400 train / 400 test with disjoint ids; frame ids are
sequential video frames, so **random** sub-splitting would leak near-duplicate
frames. Our val split therefore takes the last 20 % of the sorted train ids as
a contiguous block (`build_navigation_targets.py --val-fraction 0.2`):
train 320 / val 80 / test 400. Residual risk: block boundaries are still
adjacent frames, and the paper does not document which recordings compose each
split — treat val as weakly independent.

## Conversion

`farm_robot/tools/build_navigation_targets.py` maps the three palette colours
to class ids by nearest colour (exact here — 0 off-palette pixels), writes
uint8 masks + manifests to `data/ccrdnet/prepared/`. NAV_BAND width is kept as
annotated (~17 px at 640×480 ≈ 7 px after 256×256 stretch); Line-IoU rendering
width defaults to 6 px at 256 accordingly.

**Gate decision: PASS** — label semantics are unambiguous and the conversion is
deterministic; proceed to baseline training.
