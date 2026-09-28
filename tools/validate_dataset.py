#!/usr/bin/env python3
"""检查 YOLO 数据集结构、标签、台账和跨集合泄漏。仅使用 Python 标准库。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_FIELDS = (
    "sample_id", "image_relpath", "label_relpath", "split", "group_id",
    "source_id", "capture_date", "license", "notes",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def posix(value: str) -> str:
    return Path(value.replace("\\", "/")).as_posix()


def load_classes(path: Path, errors: list[str]) -> list[str]:
    if not path.is_file():
        errors.append(f"缺少类别文件：{path}")
        return []
    result = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not result:
        errors.append("类别文件为空")
    if len(result) != len(set(result)):
        errors.append("类别名称重复")
    return result


def load_manifest(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"缺少数据台账：{path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            errors.append("manifest.csv 表头与规范不一致")
            return []
        return list(reader)


def check_label(path: Path, class_count: int, errors: list[str], counts: Counter[int]) -> None:
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        location = f"{path}:{number}"
        if len(fields) != 5:
            errors.append(f"{location} 应为5个字段，实际为{len(fields)}个")
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(f"{location} 含有无效数字")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{location} 类别编号{class_id}越界")
        values = (x, y, width, height)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{location} 坐标不是有限数")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{location} 归一化坐标超出范围")
        epsilon = 1e-6
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon:
            errors.append(f"{location} 检测框超出图像水平边界")
        if y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            errors.append(f"{location} 检测框超出图像垂直边界")
        counts[class_id] += 1


def validate(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    data_root = root / "datasets" / "processed"
    errors: list[str] = []
    warnings: list[str] = []
    classes = load_classes(root / "datasets" / "classes.txt", errors)
    rows = load_manifest(root / "datasets" / "manifest.csv", errors)
    if not data_root.is_dir():
        errors.append(f"缺少处理后数据目录：{data_root}")

    if args.structure_only:
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            return 1
        print(f"PASS structure: {len(classes)} classes, manifest schema valid")
        print("NOTE: 未检查图片、标签、数据划分和泄漏")
        return 0

    images: dict[str, list[Path]] = {}
    labels: dict[str, list[Path]] = {}
    for split in SPLITS:
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        if image_dir.is_dir():
            images[split] = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            images[split] = []
            errors.append(f"缺少图片目录：{image_dir}")
        if label_dir.is_dir():
            labels[split] = sorted(label_dir.rglob("*.txt"))
        else:
            labels[split] = []
            errors.append(f"缺少标签目录：{label_dir}")

    if sum(map(len, images.values())) == 0:
        errors.append("数据集没有图片，P1真实数据尚未完成")

    actual_images: set[str] = set()
    actual_labels: set[str] = set()
    hashes: dict[str, tuple[str, Path]] = {}
    class_counts: Counter[int] = Counter()
    for split in SPLITS:
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        stems: set[str] = set()
        for image in images[split]:
            stem = image.relative_to(image_dir).with_suffix("").as_posix()
            if stem in stems:
                errors.append(f"{split}中存在相同主名图片：{stem}")
            stems.add(stem)
            actual_images.add(image.relative_to(data_root).as_posix())
            image_hash = digest(image)
            if image_hash in hashes and hashes[image_hash][0] != split:
                old_split, old_path = hashes[image_hash]
                errors.append(f"相同图片跨集合：{old_path}({old_split}) 与 {image}({split})")
            else:
                hashes[image_hash] = (split, image)
            label = label_dir / f"{stem}.txt"
            if not label.is_file():
                errors.append(f"图片缺少标签文件：{image}")
            else:
                actual_labels.add(label.relative_to(data_root).as_posix())
                check_label(label, len(classes), errors, class_counts)
        for label in labels[split]:
            if label.relative_to(label_dir).with_suffix("").as_posix() not in stems:
                errors.append(f"标签缺少对应图片：{label}")

    sample_ids: set[str] = set()
    manifest_images: set[str] = set()
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for number, row in enumerate(rows, 2):
        where = f"manifest.csv第{number}行"
        for field in ("sample_id", "image_relpath", "label_relpath", "split", "group_id", "source_id", "license"):
            if not row[field].strip():
                errors.append(f"{where}缺少必填字段{field}")
        sample_id = row["sample_id"].strip()
        split = row["split"].strip()
        group_id = row["group_id"].strip()
        image_path = posix(row["image_relpath"].strip())
        label_path = posix(row["label_relpath"].strip())
        if sample_id in sample_ids:
            errors.append(f"{where}样本编号重复：{sample_id}")
        sample_ids.add(sample_id)
        if split not in SPLITS:
            errors.append(f"{where}集合无效：{split}")
        if group_id:
            groups[group_id].add(split)
        if image_path in manifest_images:
            errors.append(f"{where}图片路径重复：{image_path}")
        manifest_images.add(image_path)
        if image_path and image_path not in actual_images:
            errors.append(f"{where}图片不存在：{image_path}")
        if label_path and label_path not in actual_labels:
            errors.append(f"{where}标签不存在：{label_path}")

    for group_id, split_set in groups.items():
        if len(split_set) > 1:
            errors.append(f"group_id {group_id}跨数据集：{sorted(split_set)}")
    for image_path in sorted(actual_images - manifest_images):
        errors.append(f"图片未登记到manifest.csv：{image_path}")
    for class_id, name in enumerate(classes):
        if class_counts[class_id] == 0:
            warnings.append(f"类别{class_id}({name})没有标注框")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    summary = ", ".join(f"{split}={len(images[split])}" for split in SPLITS)
    print(f"SUMMARY: {summary}, boxes={sum(class_counts.values())}, errors={len(errors)}, warnings={len(warnings)}")
    return int(bool(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--structure-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(validate(parse_args()))
