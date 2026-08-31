# -*- coding: utf-8 -*-
"""Convert CRDLD (Crop Row Detection Lincoln Dataset v2.1) into Agri-X masks.

Source: 512x512 JPEGs; labels draw EVERY crop row as a ~6 px white line.
That enables the multi-line furrow formulation the cotton dataset could not
support (it labels only the central row):

    0 OTHER      background
    1 STRUCTURE  all crop-row lines (label > 127, JPEG noise thresholded)
    2 NAV_BAND   furrow midlines: for each image row, the midpoint between
                 every pair of horizontally adjacent row lines whose gap is
                 in [min_gap, max_gap] px, drawn band_width wide

NAV_BAND is multi-instance (one band per furrow).  Training is unchanged;
evaluation and runtime select the driven furrow with the bottom-centre
component rule (postprocess C1).

    python tools/build_crdld_targets.py \
        --source "../data/crdld/Crop Row Detection Lincoln Dataset (CRDLD)/CRDLD_V2.1" \
        --out ../data/crdld/prepared
"""

import argparse
import os
import shutil

import cv2
import numpy as np


def row_runs(line_row: np.ndarray, min_run: int = 1):
    xs = np.nonzero(line_row)[0]
    if len(xs) == 0:
        return []
    breaks = np.nonzero(np.diff(xs) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(xs) - 1]))
    return [(int(xs[s]), int(xs[e])) for s, e in zip(starts, ends)
            if xs[e] - xs[s] + 1 >= min_run]


def build_mask(label: np.ndarray, band_width: int, min_gap: int, max_gap: int):
    rows = (label > 127).astype(np.uint8)
    h, w = rows.shape
    nav = np.zeros((h, w), np.uint8)
    half = band_width / 2.0
    for y in range(h):
        runs = row_runs(rows[y])
        for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
            gap = a1 - b0
            if min_gap <= gap <= max_gap:
                mid = (b0 + a1) / 2.0
                lo = int(round(mid - half))
                hi = int(round(mid + half))
                nav[y, max(0, lo):min(w, hi + 1)] = 1
    mask = np.zeros((h, w), np.uint8)
    mask[rows > 0] = 1
    mask[nav > 0] = 2
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="CRDLD_V2.1 dir with train/val/test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--band-width", type=int, default=12, help="px at 512")
    parser.add_argument("--min-gap", type=int, default=14)
    parser.add_argument("--max-gap", type=int, default=300)
    args = parser.parse_args()

    for sub in ("images", "masks", "manifests"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    class_px = np.zeros(3, np.int64)
    for split in ("train", "val", "test"):
        img_dir = os.path.join(args.source, split, "image")
        lbl_dir = os.path.join(args.source, split, "label")
        names = []
        for f in sorted(os.listdir(img_dir), key=lambda s: int(os.path.splitext(s)[0])):
            stem = os.path.splitext(f)[0]
            label = cv2.imread(os.path.join(lbl_dir, f), cv2.IMREAD_GRAYSCALE)
            if label is None:
                continue
            mask = build_mask(label, args.band_width, args.min_gap, args.max_gap)
            name = f"{split}_{stem}"
            cv2.imwrite(os.path.join(args.out, "masks", name + ".png"), mask)
            dst = os.path.join(args.out, "images", name + ".jpg")
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(img_dir, f), dst)
            names.append(name)
            class_px += np.bincount(mask.ravel(), minlength=3)
        with open(os.path.join(args.out, "manifests", split + ".txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        print(f"{split}: {len(names)}")
    freq = class_px / class_px.sum()
    print("class pixel freq OTHER/STRUCTURE/NAV_BAND:",
          " ".join(f"{v:.4f}" for v in freq))


if __name__ == "__main__":
    main()
