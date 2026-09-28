import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image
from training.baseline import audit_dataset


class BaselineAuditTests(unittest.TestCase):
    def dataset(self, root):
        for index, split in enumerate(("train", "val", "test")):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (30, 20), (index * 70, 30, 60)).save(root / "images" / split / "a.png")
            (root / "labels" / split / "a.txt").write_text("0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n", encoding="utf-8")
        path = root / "data.yaml"
        path.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                        "test": "images/test", "names": {0: "smoke", 1: "fire"}}), encoding="utf-8")
        return path

    def test_real_images_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = audit_dataset(self.dataset(root))
            self.assertEqual(result["splits"]["train"]["boxes_per_class"], {0: 1, 1: 1})

    def test_cross_split_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.dataset(root)
            (root / "images" / "val" / "a.png").write_bytes((root / "images" / "train" / "a.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "Identical image crosses splits"):
                audit_dataset(path)

    def test_missing_class_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.dataset(root)
            (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Every configured class"):
                audit_dataset(path)


if __name__ == "__main__":
    unittest.main()
