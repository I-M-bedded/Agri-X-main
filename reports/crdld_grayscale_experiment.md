# CRDLD grayscale-camera experiment

Date: 2026-09-01

## Verdict

Do not replace the RGB camera/model with the tested grayscale models. The best
grayscale result retained only 23% of the RGB line-IoU and reduced line
accuracy from 69.3% to 21.9%, while the true one-channel network reduced MACs
by only 1.83%. CRDLD's plant/soil colour contrast is useful signal for this
model.

This does not prove that a physical monochrome sensor can never work. CRDLD
contains RGB images converted to luminance, not raw frames from the target
monochrome sensor (whose spectral response, dynamic range, and IR sensitivity
can differ). A field-collected monochrome set is required before a final
hardware decision.

## Controlled setup

- Data: CRDLD prepared split, train 1,250 / val 250 / untouched test 430.
- Target: 3 classes (OTHER / STRUCTURE / NAV_BAND), bottom-centre furrow
  component selection.
- Common settings: 256x256, seed 42, batch 4, weighted CE 1:4:4, Adam,
  identical photometric/geometric augmentation policy.
- RGB baseline: maximum 200 epochs, patience 30, best epoch 116.
- Gray-1ch scratch: maximum 200 epochs, patience 30, best epoch 31.
- Gray-RGB-stem fine-tune: RGB checkpoint warm start, gray input repeated to
  three channels inside the ONNX graph, Adam 5e-5, maximum 100 epochs,
  patience 20, best epoch 27. Epoch 0 was evaluated so a bad fine-tune could
  not silently overwrite the warm-start state.
- Grayscale conversion: OpenCV RGB-to-gray luminance conversion after resize;
  the network receives one channel, matching a monochrome-camera interface.

## Untouched test results

| Model | Input/stem | Line IoU mean | Line accuracy | Angle mean | Lateral near mean | Detection |
|---|---|---:|---:|---:|---:|---:|
| RGB baseline | 3ch / RGB | **0.4164** | **69.30%** | **5.32 deg** | **19.43 px** | 100% |
| Gray scratch | 1ch / 1ch | 0.0839 | 21.63% | 10.96 deg | 38.46 px | 100% |
| Gray fine-tune | 1ch / replicated RGB stem | 0.0972 | 21.86% | 10.54 deg | 34.04 px | 100% |

Line accuracy means angle error <=5 degrees and near-field lateral error <=10
pixels at 256x256. Detection alone is misleading here: every gray frame
produced a line, but most selected lines were wrong.

Relative to RGB, the better grayscale fine-tune loses 76.7% of mean line-IoU
and 47.4 percentage points of line accuracy. Visual failures commonly select
an adjacent corridor after plant/soil colour separation disappears.

## Size, compute, and CPU proxy

| Model | Parameters | MACs @256 | ONNX size | ORT inference median |
|---|---:|---:|---:|---:|
| RGB baseline | 33,694 | 37.61 M | 162.6 KiB | 4.174 ms |
| Gray scratch | 33,652 | 36.92 M | 162.5 KiB | 3.972 ms |
| Gray fine-tune | 33,694 | 37.61 M | 163.1 KiB | 4.284 ms |

Benchmark: AMD Ryzen 5 7600, ONNX Runtime CPU, 3 intra-op threads, sequential
execution, 20 warmups, 7 repeats x 50 runs. This is a same-host comparison,
not a Raspberry Pi 4B latency claim. The direct one-channel model saves 42
parameters and 0.69 M MACs (1.83%); observed latency differences are small
enough that Pi deployment should be measured on-device.

The one-channel ONNX interface does cut input tensor bandwidth by three, but
the segmentation convolutions dominate. The replicated-stem variant preserves
RGB compute and therefore offers no meaningful model-side speedup.

## Reproduction

```powershell
.venv-train/Scripts/python.exe farm_robot/tools/train_ccrdnet.py `
  --data-root data/crdld/prepared --out runs/crdld_furrow_gray_v1 `
  --epochs 200 --patience 30 --workers 4 --class-weights 1,4,4 `
  --component-mode bottom_center --grayscale

.venv-train/Scripts/python.exe farm_robot/tools/train_ccrdnet.py `
  --data-root data/crdld/prepared `
  --out runs/crdld_furrow_gray_rgbstem_ft_v1 `
  --epochs 100 --patience 20 --workers 4 --lr 5e-5 `
  --class-weights 1,4,4 --component-mode bottom_center --grayscale `
  --replicate-gray-to-rgb --init-checkpoint runs/crdld_furrow_v1/best.pt
```

Detailed metrics and overlays are in `reports/crdld_furrow_gray_v1/` and
`reports/crdld_furrow_gray_rgbstem_ft_v1/`. Deployable, parity-checked ONNX
files are under `farm_robot/models/`; checkpoints and training curves are
under `runs/` (both are intentionally gitignored).

## Recommendation

Keep RGB as the primary camera path. If the hardware must be monochrome, do
not spend more compute on this CRDLD luminance conversion alone. First capture
and label a compact target-camera set spanning bare soil and crop-present
conditions, then fine-tune and require at least the current RGB gate on that
fixed field test set. A more realistic efficiency win should come from input
resolution/runtime scheduling or INT8 benchmarking, not deleting colour for a
1.83% MAC reduction.
