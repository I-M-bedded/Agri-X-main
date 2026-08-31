#!/usr/bin/env python3
"""Range-download only the 47 frontal RGB YOLO-seg pairs from Zenodo 20584286."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from remotezip import RemoteZip
except ImportError as exc:  # pragma: no cover - environment guidance
    raise SystemExit("Install the lightweight downloader first: python -m pip install remotezip") from exc


RECORD_URL = "https://zenodo.org/records/20584286"
ARCHIVES = {
    f"trajectory{index:02d}": (
        "https://zenodo.org/api/records/20584286/files/"
        f"COTTON_SEEDLING_DATASET_V1_trajectory{index:02d}.zip/content"
    )
    for index in range(1, 4)
}
EXPECTED_IMAGES = {"trajectory01": 17, "trajectory02": 13, "trajectory03": 17}


def _retry_session() -> requests.Session:
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        status=10,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _record_local_file(output: Path, path: Path, trajectory: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "trajectory": trajectory,
        "archive_member": None,
        "path": path.relative_to(output).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _selected_relative_path(member: str) -> PurePosixPath | None:
    marker = "/annotations/rgb/yolo_seg/"
    if marker not in member or member.endswith("/"):
        return None
    relative = PurePosixPath(member.split(marker, 1)[1])
    if relative.parts[0] in {"images", "labels"}:
        return relative
    if relative.as_posix() in {"data.yaml", "split_manifest.csv"}:
        return relative
    return None


def download(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    trajectory_counts: dict[str, dict[str, int]] = {}

    for trajectory, url in ARCHIVES.items():
        trajectory_root = output / trajectory
        existing_images = sorted((trajectory_root / "images").rglob("*.jpg"))
        existing_labels = sorted((trajectory_root / "labels").rglob("*.txt"))
        if (
            len(existing_images) == EXPECTED_IMAGES[trajectory]
            and len(existing_labels) == EXPECTED_IMAGES[trajectory]
            and {path.stem for path in existing_images} == {path.stem for path in existing_labels}
        ):
            print(
                f"already complete: {trajectory} "
                f"({len(existing_images)} images + {len(existing_labels)} labels)",
                flush=True,
            )
            trajectory_counts[trajectory] = {
                "images": len(existing_images),
                "labels": len(existing_labels),
            }
            for path in sorted(path for path in trajectory_root.rglob("*") if path.is_file()):
                records.append(_record_local_file(output, path, trajectory))
            continue

        print(f"opening remote ZIP: {trajectory}", flush=True)
        with _retry_session() as session, RemoteZip(
            url, session=session, timeout=120
        ) as archive:
            selected = []
            for info in archive.infolist():
                relative = _selected_relative_path(info.filename)
                if relative is not None:
                    selected.append((info, relative))

            images = [item for item in selected if item[1].parts[0] == "images"]
            labels = [item for item in selected if item[1].parts[0] == "labels"]
            image_stems = {item[1].stem for item in images}
            label_stems = {item[1].stem for item in labels}
            if image_stems != label_stems:
                raise RuntimeError(
                    f"{trajectory}: image/label mismatch: "
                    f"images_only={sorted(image_stems - label_stems)}, "
                    f"labels_only={sorted(label_stems - image_stems)}"
                )

            trajectory_counts[trajectory] = {
                "images": len(images),
                "labels": len(labels),
            }
            print(
                f"  selected {len(images)} images + {len(labels)} labels",
                flush=True,
            )

            for position, (info, relative) in enumerate(selected, start=1):
                destination = output / trajectory / Path(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and destination.stat().st_size == info.file_size:
                    data = destination.read_bytes()
                    state = "kept"
                else:
                    data = archive.read(info.filename)
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                    state = "downloaded"
                records.append(
                    {
                        "trajectory": trajectory,
                        "archive_member": info.filename,
                        "path": destination.relative_to(output).as_posix(),
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                print(
                    f"  [{position:02d}/{len(selected):02d}] {state}: {relative}",
                    flush=True,
                )

    total_images = sum(counts["images"] for counts in trajectory_counts.values())
    total_labels = sum(counts["labels"] for counts in trajectory_counts.values())
    if (total_images, total_labels) != (47, 47):
        raise RuntimeError(
            f"Expected 47 paired RGB samples, found {total_images} images/{total_labels} labels"
        )

    manifest: dict[str, object] = {
        "source": RECORD_URL,
        "doi": "10.5281/zenodo.20584286",
        "scope": "annotations/rgb/yolo_seg only",
        "trajectory_counts": trajectory_counts,
        "total_images": total_images,
        "total_labels": total_labels,
        "files": records,
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/cotton_pilot"))
    args = parser.parse_args()
    result = download(args.output)
    print(
        f"complete: {result['total_images']} images + {result['total_labels']} labels "
        f"under {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
