# CCRDNet evaluation summary

- checkpoint: `../runs/crdld_furrow_v1/best.pt` (epoch 116)
- split: test, 430 frames (430 with a GT line)
- detection rate: 1.0000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 5.324 | 1.258 | 28.381 |
| lateral near (px) | 19.43 | 3.44 | 99.74 |
| line IoU | 0.4164 | 0.4729 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.6930

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
