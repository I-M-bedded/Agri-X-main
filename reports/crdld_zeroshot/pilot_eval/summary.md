# CCRDNet evaluation summary

- checkpoint: `../runs/crdld_furrow_v1/best.pt` (epoch 116)
- split: test_all, 47 frames (42 with a GT line)
- detection rate: 1.0000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 26.267 | 25.587 | 52.812 |
| lateral near (px) | 54.28 | 65.26 | 82.77 |
| line IoU | 0.0417 | 0.0222 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.0238

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
