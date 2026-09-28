import unittest
from tools.prepare_public_pilot import clip_yolo_row


class PreparePilotTests(unittest.TestCase):
    def test_valid_box_unchanged(self):
        row = "0 0.5 0.5 0.2 0.2"
        self.assertEqual(clip_yolo_row(row), (row, False))

    def test_visible_intersection(self):
        row, changed = clip_yolo_row("0 0.51 0.5 1.0 1.0")
        _, x, y, w, h = map(float, row.split())
        self.assertTrue(changed)
        self.assertAlmostEqual(x - w / 2, 0.01)
        self.assertAlmostEqual(x + w / 2, 1.0)

    def test_invalid_values_not_silently_repaired(self):
        for row in ("0 nan 0.5 0.2 0.2", "0 0.5 0.5 -1 1", "2 0.5 0.5 0.2 0.2"):
            with self.assertRaises(ValueError):
                clip_yolo_row(row)
