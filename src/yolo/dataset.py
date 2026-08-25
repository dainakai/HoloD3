from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path is outside the self-contained workspace: {resolved}") from exc
    return resolved


def repo_relative(path: Path) -> str:
    absolute = path if path.is_absolute() else REPO_ROOT / path
    return absolute.absolute().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_yolo_objects(label_path: Path) -> int:
    count = 0
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: expected 5 fields")
        cls, xc, yc, width, height = (float(value) for value in parts)
        if int(cls) != cls or cls < 0:
            raise ValueError(f"Invalid class id at {label_path}:{line_number}: {parts[0]}")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"Out-of-range YOLO box at {label_path}:{line_number}")
        count += 1
    return count


@dataclass(frozen=True)
class DatasetItem:
    source: str
    source_root: str
    source_split: str
    split: str
    image: str
    label: str
    image_sha256: str
    label_sha256: str
    object_count: int


@dataclass(frozen=True)
class DuplicateItem:
    source: str
    source_split: str
    image: str
    image_sha256: str
    label_sha256: str
    kept_image: str
    kept_label_sha256: str
    kept_split: str
    split_conflict: bool
    label_conflict: bool


def _split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path] | None:
    aliases = ("valid", "val", "validation") if split == "val" else ("train",)
    for alias in aliases:
        image_dir = dataset_root / alias / "images"
        label_dir = dataset_root / alias / "labels"
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir, label_dir
    return None


def discover_dataset(dataset_root: Path, source: str) -> list[DatasetItem]:
    dataset_root = repo_path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    items: list[DatasetItem] = []
    for normalized_split in ("train", "val"):
        dirs = _split_dirs(dataset_root, normalized_split)
        if dirs is None:
            raise FileNotFoundError(f"Dataset has no {normalized_split} images/labels split: {dataset_root}")
        image_dir, label_dir = dirs
        images = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise FileNotFoundError(f"No images in {image_dir}")
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing YOLO label for {image_path}: {label_path}")
            items.append(
                DatasetItem(
                    source=source,
                    source_root=repo_relative(dataset_root),
                    source_split=image_dir.parent.name,
                    split=normalized_split,
                    image=repo_relative(image_path),
                    label=repo_relative(label_path),
                    image_sha256=sha256_file(image_path),
                    label_sha256=sha256_file(label_path),
                    object_count=count_yolo_objects(label_path),
                )
            )
    return items


def deduplicate_items(items: Iterable[DatasetItem]) -> tuple[list[DatasetItem], list[DuplicateItem]]:
    """Keep the first occurrence of identical image bytes and record split conflicts.

    Sources must be passed in explicit priority order. The first copy of duplicate
    image bytes is retained and every later copy is recorded in the provenance table.
    """

    kept: list[DatasetItem] = []
    duplicates: list[DuplicateItem] = []
    by_digest: dict[str, DatasetItem] = {}
    for item in items:
        previous = by_digest.get(item.image_sha256)
        if previous is None:
            by_digest[item.image_sha256] = item
            kept.append(item)
            continue
        duplicates.append(
            DuplicateItem(
                source=item.source,
                source_split=item.source_split,
                image=item.image,
                image_sha256=item.image_sha256,
                label_sha256=item.label_sha256,
                kept_image=previous.image,
                kept_label_sha256=previous.label_sha256,
                kept_split=previous.split,
                split_conflict=item.split != previous.split,
                label_conflict=item.label_sha256 != previous.label_sha256,
            )
        )

    train_hashes = {item.image_sha256 for item in kept if item.split == "train"}
    val_hashes = {item.image_sha256 for item in kept if item.split == "val"}
    overlap = train_hashes & val_hashes
    if overlap:
        raise RuntimeError(f"Image leakage remains between train and val: {len(overlap)} identical images")
    return kept, duplicates


def _write_list(path: Path, image_paths: list[Path]) -> None:
    lines: list[str] = []
    for image_path in image_paths:
        relative = os.path.relpath(image_path, start=path.parent).replace(os.sep, "/")
        # Ultralytics expands list entries beginning with './' relative to the list file.
        lines.append(f"./{relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_source_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _materialize_runtime_view(
    output_dir: Path, items: list[DatasetItem]
) -> list[tuple[DatasetItem, Path, Path]]:
    """Create a symlink-free runtime view so caches stay outside source data."""

    runtime: list[tuple[DatasetItem, Path, Path]] = []
    for item in items:
        image_source = repo_path(item.image)
        label_source = repo_path(item.label)
        stem = f"{_safe_source_name(item.source)}__{item.image_sha256[:12]}__{image_source.stem}"
        image_link = output_dir / item.split / "images" / f"{stem}{image_source.suffix.lower()}"
        label_link = output_dir / item.split / "labels" / f"{stem}.txt"
        image_link.parent.mkdir(parents=True, exist_ok=True)
        label_link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(image_source, image_link)
            os.link(label_source, label_link)
        except OSError:
            image_link.unlink(missing_ok=True)
            label_link.unlink(missing_ok=True)
            shutil.copy2(image_source, image_link)
            shutil.copy2(label_source, label_link)
        runtime.append((item, image_link, label_link))
    return runtime


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_mixed_dataset(
    output_dir: str | Path,
    sources: list[tuple[str, str | Path]],
) -> dict[str, object]:
    output_dir = repo_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Mixed dataset directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered: list[DatasetItem] = []
    for source_name, source_root in sources:
        discovered.extend(discover_dataset(repo_path(source_root), source_name))
    kept, duplicates = deduplicate_items(discovered)
    train_items = [item for item in kept if item.split == "train"]
    val_items = [item for item in kept if item.split == "val"]
    if not train_items or not val_items:
        raise ValueError(f"Mixed dataset requires non-empty train and val lists: train={len(train_items)}, val={len(val_items)}")

    runtime = _materialize_runtime_view(output_dir, kept)
    train_runtime = [image for item, image, _label in runtime if item.split == "train"]
    val_runtime = [image for item, image, _label in runtime if item.split == "val"]
    train_list = output_dir / "train_images.txt"
    val_list = output_dir / "val_images.txt"
    _write_list(train_list, train_runtime)
    _write_list(val_list, val_runtime)

    # Source YAML files (which contain historical external paths) are deliberately ignored.
    # The generated YAML is portable as long as training is launched from REPO_ROOT.
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {repo_relative(output_dir)}",
                f"train: {train_list.name}",
                f"val: {val_list.name}",
                "",
                "nc: 1",
                "names: ['drop']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = output_dir / "manifest.csv"
    item_fields = [*DatasetItem.__dataclass_fields__, "runtime_image", "runtime_label"]
    _write_csv(
        manifest_path,
        (
            {
                **asdict(item),
                "runtime_image": repo_relative(image_link),
                "runtime_label": repo_relative(label_link),
            }
            for item, image_link, label_link in runtime
        ),
        item_fields,
    )
    duplicate_path = output_dir / "duplicates.csv"
    duplicate_fields = list(DuplicateItem.__dataclass_fields__)
    _write_csv(duplicate_path, (asdict(item) for item in duplicates), duplicate_fields)

    source_counts: dict[str, dict[str, int]] = {}
    for item in kept:
        counts = source_counts.setdefault(item.source, {"train_images": 0, "val_images": 0, "objects": 0})
        counts[f"{item.split}_images"] += 1
        counts["objects"] += item.object_count
    summary: dict[str, object] = {
        "dataset_dir": repo_relative(output_dir),
        "data_yaml": repo_relative(data_yaml),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "train_objects": sum(item.object_count for item in train_items),
        "val_objects": sum(item.object_count for item in val_items),
        "duplicate_images_removed": len(duplicates),
        "duplicate_split_conflicts": sum(item.split_conflict for item in duplicates),
        "duplicate_label_conflicts": sum(item.label_conflict for item in duplicates),
        "manifest": repo_relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "duplicates_manifest": repo_relative(duplicate_path),
        "duplicates_manifest_sha256": sha256_file(duplicate_path),
        "sources": source_counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
