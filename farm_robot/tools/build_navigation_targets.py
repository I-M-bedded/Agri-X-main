# -*- coding: utf-8 -*-
"""Convert the CCRDNet cotton dataset into Agri-X 3-class training masks.

Source (Zenodo 15194034, CC-BY-4.0), 640x480 PNG:

    dataset/{train,test}/rgb/<id>.png      RGB frame
    dataset/{train,test}/line/<id>.png     colour label:
        black (0,0,0)       background
        white (255,255,255) vegetation (non-central crop rows)
        red   BGR (0,0,255) central crop row band (~15 px wide)

Mapping (deterministic, spec section 3):

    black -> 0 OTHER
    white -> 1 STRUCTURE
    red   -> 2 NAV_BAND

Any off-palette pixel (anti-aliasing) is assigned to the nearest of the three
colours; the per-dataset count of such pixels is reported.

The official train split is carved into train/val by *numeric id blocks*
(sorted ids, last VAL_FRACTION as val) rather than randomly, so temporally
adjacent frames do not leak across the split (spec section 23).

    python tools/build_navigation_targets.py \
        --source ../data/ccrdnet/dataset --out ../data/ccrdnet/prepared
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

PALETTE_BGR = np.array(
    [[0, 0, 0], [255, 255, 255], [0, 0, 255]], np.int32
)  # OTHER, STRUCTURE, NAV_BAND


def convert_mask(label_bgr: np.ndarray):
    """Colour label -> class ids; returns (mask, off_palette_pixel_count)."""
    flat = label_bgr.reshape(-1, 3).astype(np.int32)
    dists = ((flat[:, None, :] - PALETTE_BGR[None, :, :]) ** 2).sum(axis=2)
    classes = dists.argmin(axis=1).astype(np.uint8)
    off_palette = int((dists.min(axis=1) > 0).sum())
    return classes.reshape(label_bgr.shape[:2]), off_palette


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="extracted dataset/ dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    for sub in ("images", "masks", "manifests"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    manifests = {"train": [], "val": [], "test": []}
    total_off = 0
    stats = {"frames": 0, "no_nav": []}

    for split in ("train", "test"):
        rgb_dir = os.path.join(args.source, split, "rgb")
        line_dir = os.path.join(args.source, split, "line")
        ids = sorted(
            (os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.endswith(".png")),
            key=lambda s: int(s),
        )
        if split == "train":
            n_val = int(round(len(ids) * args.val_fraction))
            assignment = [("train", ids[: len(ids) - n_val]), ("val", ids[len(ids) - n_val:])]
        else:
            assignment = [("test", ids)]

        for target_split, split_ids in assignment:
            for frame_id in split_ids:
                name = f"{split}_{frame_id}"
                label = cv2.imread(os.path.join(line_dir, frame_id + ".png"),
                                   cv2.IMREAD_COLOR)
                if label is None:
                    print(f"WARN missing label {split}/{frame_id}", file=sys.stderr)
                    continue
                mask, off = convert_mask(label)
                total_off += off
                if not (mask == 2).any():
                    stats["no_nav"].append(name)
                cv2.imwrite(os.path.join(args.out, "masks", name + ".png"), mask)
                src_img = os.path.join(rgb_dir, frame_id + ".png")
                dst_img = os.path.join(args.out, "images", name + ".png")
                if not os.path.exists(dst_img):
                    shutil.copyfile(src_img, dst_img)
                manifests[target_split].append(name)
                stats["frames"] += 1

    for split, names in manifests.items():
        with open(os.path.join(args.out, "manifests", split + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        print(f"{split}: {len(names)} frames")
    print(f"total frames {stats['frames']}, off-palette pixels {total_off}")
    print(f"frames without NAV_BAND: {len(stats['no_nav'])} {stats['no_nav'][:10]}")


if __name__ == "__main__":
    main()
