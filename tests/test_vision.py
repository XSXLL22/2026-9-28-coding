import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from runtime.common import class_mapping
from runtime.vision import frames, roi_contour, roi_overlap


class VisionTests(unittest.TestCase):
    def test_coco_ids_never_become_fire_labels(self):
        self.assertEqual(class_mapping({0: "person", 1: "bicycle"}), {0: None, 1: None})
        self.assertEqual(class_mapping({0: "fire", 1: "smoke"}), {0: 1, 1: 0})

    def test_roi_intersection_fraction(self):
        config = {"roi": {"polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}}
        contour = roi_contour(config, 100, 100)
        self.assertAlmostEqual(roi_overlap([0, 0, 100, 100], contour), 0.25)
        self.assertAlmostEqual(roi_overlap([25, 25, 75, 75], contour), 1.0)
        self.assertEqual(roi_overlap([0, 0, 10, 10], contour), 0.0)

    def test_degenerate_roi_rejected(self):
        with self.assertRaises(ValueError):
            roi_contour({"roi": {"polygon": [[0, 0], [0.5, 0.5], [1, 1]]}}, 100, 100)

    def test_unicode_image_and_directory_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "中文图片"
            root.mkdir()
            for number in (1, 2):
                _, encoded = cv2.imencode(".png", np.full((20, 30, 3), number, np.uint8))
                encoded.tofile(str(root / f"{number:03d}.png"))
            records = list(frames(root))
            self.assertEqual(len(records), 2)
            self.assertIsNone(records[0][1])
            self.assertEqual(records[0][4].shape, (20, 30, 3))
            timed = list(frames(root, 5))
            self.assertEqual(timed[1][1], 200.0)

    def test_corrupt_image_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                list(frames(path))

    def test_video_frames_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(3):
                writer.write(np.full((48, 64, 3), i * 60, np.uint8))
            writer.release()
            records = list(frames(path))
            self.assertEqual(len(records), 3)
            self.assertAlmostEqual(records[2][1], 400.0, places=1)


if __name__ == "__main__":
    unittest.main()
