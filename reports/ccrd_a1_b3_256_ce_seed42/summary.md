# CCRDNet evaluation summary

- checkpoint: `../runs/ccrd_a1_b3_256_ce_seed42/best.pt` (epoch 52)
- split: test, 400 frames (400 with a GT line)
- detection rate: 1.0000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 0.780 | 0.400 | 3.055 |
| lateral near (px) | 1.79 | 0.87 | 6.59 |
| line IoU | 0.7264 | 0.7556 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.9750

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).

**Verdict: KEEP** — paper-level navigation accuracy on the held-out test split (AE 0.78°, line accuracy 97.5%); dominant failure mode is angled central rows under dense canopy, with confidence staying ~0.7 (calibration work needed before control use). See ../ccrdnet_training_results.md.
