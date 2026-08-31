# -*- coding: utf-8 -*-
"""Paper-style CCRDNet baseline training (spec section 39).

    python tools/train_ccrdnet.py --data-root ../data/ccrdnet/prepared \
        --out ../runs/ccrd_a1_b3_256_ce_seed42

Baseline follows the paper: 256x256 RGB, 3-class cross entropy, Adam 2e-4,
batch 4.  Validation curves drive checkpoint selection and early stopping
instead of blindly running 500 epochs.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_FARM_ROBOT = os.path.abspath(os.path.join(_HERE, ".."))
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from perception.ccrdnet.config import CCRDNetConfig  # noqa: E402
from perception.ccrdnet.model import CCRDNet, count_macs, count_parameters  # noqa: E402
from perception.ccrdnet.postprocess import PostprocessConfig  # noqa: E402
from tools.ccrdnet_data import CCRDDataset, seed_everything  # noqa: E402
from tools.ccrdnet_metrics import compute_frame_metrics, summarize  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_FARM_ROBOT, text=True
        ).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def validate(model, loader, device, nav_class, postprocess, line_width, criterion):
    model.eval()
    losses = []
    frames = []
    for images, targets, names in loader:
        images = images.to(device)
        targets_d = targets.to(device)
        logits = model(images)
        losses.append(float(criterion(logits, targets_d).item()))
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy().astype(np.uint8)
        nav_probs = probs[:, nav_class].cpu().numpy()
        gts = targets.numpy().astype(np.uint8)
        for i, name in enumerate(names):
            frames.append(
                compute_frame_metrics(
                    name, preds[i], gts[i], nav_class, postprocess, line_width,
                    nav_probability=nav_probs[i],
                )
            )
    return float(np.mean(losses)), summarize(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=40,
                        help="early stop after this many epochs without val line-IoU improvement")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-dsc", action="store_true")
    parser.add_argument("--no-aspp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--geo-augment", action="store_true",
                        help="add small rotation/shift augmentation (identical on masks)")
    parser.add_argument("--line-width", type=int, default=6,
                        help="band width in px at 256 for Line IoU rendering")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--init-checkpoint", default=None,
                        help="warm-start weights from this .pt before training")
    parser.add_argument("--class-weights", default=None,
                        help="comma-separated CE weights, e.g. 1,1,8 (default: unweighted)")
    parser.add_argument("--component-mode", default="largest",
                        choices=["largest", "bottom_center", "scored"],
                        help="NAV_BAND component selection for validation metrics")
    parser.add_argument("--grayscale", action="store_true",
                        help="train a true one-channel model for a monochrome camera")
    parser.add_argument("--replicate-gray-to-rgb", action="store_true",
                        help="repeat 1-channel gray inside the model to reuse an RGB stem")
    args = parser.parse_args()

    if args.replicate_gray_to_rgb and not args.grayscale:
        parser.error("--replicate-gray-to-rgb requires --grayscale")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = CCRDNetConfig(
        in_channels=1 if args.grayscale else 3,
        replicate_grayscale_input=args.replicate_gray_to_rgb,
        use_dsc=not args.no_dsc,
        aspp_skip_count=0 if args.no_aspp else 3,
    )
    model = CCRDNet(config)
    params = count_parameters(model)
    macs = count_macs(model, config.input_size)  # on CPU, before moving to device
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"warm-start from {args.init_checkpoint} (epoch {ckpt.get('epoch')})")
    model = model.to(device)

    os.makedirs(args.out, exist_ok=True)
    run_meta = {
        "argv": sys.argv,
        "args": vars(args),
        "git_commit": git_commit(),
        "device": str(device),
        "parameters": params,
        "macs_256": macs,
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=str)
    print(f"device={device} params={params} macs={macs/1e6:.2f}M")

    train_ds = CCRDDataset(args.data_root, "train", config.input_size,
                           augment=not args.no_augment,
                           geometric_augment=args.geo_augment, seed=args.seed,
                           in_channels=config.in_channels)
    val_ds = CCRDDataset(args.data_root, "val", config.input_size, augment=False,
                         in_channels=config.in_channels)
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    if args.class_weights:
        weights = torch.tensor(
            [float(v) for v in args.class_weights.split(",")], device=device
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    postprocess = PostprocessConfig(component_mode=args.component_mode)

    metrics_path = os.path.join(args.out, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_line_iou", "val_angle_err",
             "val_lat_near", "val_line_acc", "val_det_rate", "seconds"]
        )

    best_iou = -1.0
    best_epoch = -1
    if args.init_checkpoint:
        val_loss, summary = validate(
            model, val_loader, device, config.class_nav_band,
            postprocess, args.line_width, criterion,
        )
        best_iou = summary["line_iou"]["mean"] or 0.0
        best_epoch = 0
        ae = summary["angle_error_deg"]["mean"]
        lat = summary["lateral_near_px"]["mean"]
        acc = summary["line_accuracy"]
        det = summary["detection_rate"]
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [0, "", f"{val_loss:.4f}", f"{best_iou:.4f}",
                 "" if ae is None else f"{ae:.3f}",
                 "" if lat is None else f"{lat:.2f}",
                 "" if acc is None else f"{acc:.4f}",
                 "" if det is None else f"{det:.4f}", "0.0"]
            )
        torch.save({"model": model.state_dict(), "epoch": 0,
                    "val_line_iou": best_iou, "config": run_meta["config"]},
                   os.path.join(args.out, "best.pt"))
        print(
            f"epoch   0 warm-start val loss {val_loss:.4f} lineIoU {best_iou:.4f} "
            f"AE {ae if ae is None else round(ae, 2)} "
            f"lat {lat if lat is None else round(lat, 1)}px acc {acc}",
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))

        if epoch % args.val_every == 0:
            val_loss, summary = validate(
                model, val_loader, device, config.class_nav_band,
                postprocess, args.line_width, criterion,
            )
            iou = summary["line_iou"]["mean"] or 0.0
            ae = summary["angle_error_deg"]["mean"]
            lat = summary["lateral_near_px"]["mean"]
            acc = summary["line_accuracy"]
            det = summary["detection_rate"]
            dt = time.time() - t0
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{iou:.4f}",
                     "" if ae is None else f"{ae:.3f}",
                     "" if lat is None else f"{lat:.2f}",
                     "" if acc is None else f"{acc:.4f}",
                     "" if det is None else f"{det:.4f}", f"{dt:.1f}"]
                )
            print(
                f"epoch {epoch:3d} loss {train_loss:.4f}/{val_loss:.4f} "
                f"lineIoU {iou:.4f} AE {ae if ae is None else round(ae,2)} "
                f"lat {lat if lat is None else round(lat,1)}px acc {acc} ({dt:.0f}s)",
                flush=True,
            )
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": run_meta["config"]},
                       os.path.join(args.out, "last.pt"))
            if iou > best_iou:
                best_iou, best_epoch = iou, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_line_iou": iou, "config": run_meta["config"]},
                           os.path.join(args.out, "best.pt"))
            if epoch - best_epoch >= args.patience:
                print(f"early stop at {epoch} (best line IoU {best_iou:.4f} @ {best_epoch})")
                break

    run_meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["best_val_line_iou"] = best_iou
    run_meta["best_epoch"] = best_epoch
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=str)
    print(f"done: best line IoU {best_iou:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
