from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "tools" / "validate_dataset.py"
HEADER = [
    "sample_id", "image_relpath", "label_relpath", "split", "group_id",
    "source_id", "capture_date", "license", "notes",
]


class DatasetValidatorTest(unittest.TestCase):
    def prepare(self, root: Path) -> None:
        (root / "datasets").mkdir()
        (root / "datasets" / "classes.txt").write_text(
            "smoke\nfire\nleaf_pile\n", encoding="utf-8"
        )
        for split in ("train", "val", "test"):
            (root / "datasets" / "processed" / "images" / split).mkdir(parents=True)
            (root / "datasets" / "processed" / "labels" / split).mkdir(parents=True)

    def write_manifest(self, root: Path, rows: list[list[str]]) -> None:
        with (root / "datasets" / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(rows)

    def run_check(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VALIDATOR), "--project-root", str(root)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def add_sample(
        self, root: Path, split: str, index: int, group: str, label: str
    ) -> list[str]:
        image_rel = f"images/{split}/sample_{index}.jpg"
        label_rel = f"labels/{split}/sample_{index}.txt"
        (root / "datasets" / "processed" / image_rel).write_bytes(f"image-{index}".encode())
        (root / "datasets" / "processed" / label_rel).write_text(label, encoding="utf-8")
        return [
            f"sample_{index}", image_rel, label_rel, split, group, "source",
            "2026-09-26", "self-collected", "",
        ]

    def test_valid_dataset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            rows = [
                self.add_sample(root, split, index, f"group_{index}", f"{index} 0.5 0.5 0.2 0.2\n")
                for index, split in enumerate(("train", "val", "test"))
            ]
            self.write_manifest(root, rows)
            result = self.run_check(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cross_split_group_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            rows = [
                self.add_sample(root, split, index, "same_group", "0 0.5 0.5 0.2 0.2\n")
                for index, split in enumerate(("train", "val"))
            ]
            self.write_manifest(root, rows)
            result = self.run_check(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("跨数据集", result.stdout)

    def test_box_outside_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            row = self.add_sample(root, "train", 0, "group_0", "0 0.95 0.5 0.2 0.2\n")
            self.write_manifest(root, [row])
            result = self.run_check(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("超出图像水平边界", result.stdout)


if __name__ == "__main__":
    unittest.main()
