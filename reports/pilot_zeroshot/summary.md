# CCRDNet evaluation summary

- checkpoint: `../runs/ccrd_a1_b3_256_ce_seed42/best.pt` (epoch 52)
- split: test_all, 47 frames (42 with a GT line)
- detection rate: 0.8333

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 20.931 | 21.330 | 30.111 |
| lateral near (px) | 22.96 | 20.81 | 37.04 |
| line IoU | 0.0401 | 0.0387 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.0000

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
