import unittest

from tools.prepare_fp_review import PAIR_IOU, iou, pair_image


class FpPairingTests(unittest.TestCase):
    def test_iou_basic_and_degenerate(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 5, 15, 15]), 25/175)
        self.assertEqual(iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)
        self.assertEqual(iou([0, 0, 0, 0], [0, 0, 5, 5]), 0.0)

    def test_pair_image_classifies_added_kept_dropped(self):
        old = [{'class': 'fire', 'xyxy': [0, 0, 10, 10], 'confidence': 0.5},
               {'class': 'fire', 'xyxy': [100, 100, 110, 110], 'confidence': 0.4},
               {'class': 'smoke', 'xyxy': [50, 50, 60, 60], 'confidence': 0.3}]
        new = [{'class': 'fire', 'xyxy': [1, 1, 11, 11], 'confidence': 0.6},          # kept (IoU ~0.68)
               {'class': 'fire', 'xyxy': [200, 200, 210, 210], 'confidence': 0.3},    # added
               {'class': 'smoke', 'xyxy': [52, 52, 62, 62], 'confidence': 0.35}]      # kept (IoU ~0.68)
        kept, added, dropped, matched_old = pair_image(old, new)
        self.assertEqual([b['confidence'] for b in added], [0.3])
        self.assertEqual(len(kept), 2)
        self.assertEqual([b['confidence'] for b in dropped], [0.4])
        self.assertEqual(matched_old, 2)

    def test_pairing_is_class_aware(self):
        old = [{'class': 'smoke', 'xyxy': [0, 0, 10, 10], 'confidence': 0.5}]
        new = [{'class': 'fire', 'xyxy': [0, 0, 10, 10], 'confidence': 0.9}]
        kept, added, dropped, matched_old = pair_image(old, new)
        self.assertEqual(len(added), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(kept, [])
        self.assertEqual(matched_old, 0)

    def test_iou_threshold_boundary_is_inclusive(self):
        old = [{'class': 'fire', 'xyxy': [0, 0, 10, 10], 'confidence': 0.5}]
        offset = 10 * (1 - PAIR_IOU) / (1 + PAIR_IOU)   # IoU exactly PAIR_IOU
        new = [{'class': 'fire', 'xyxy': [offset, 0, 10 + offset, 10], 'confidence': 0.5}]
        kept, added, dropped, matched_old = pair_image(old, new)
        self.assertEqual(len(kept), 1)
        self.assertEqual(added, [])
        self.assertEqual(dropped, [])
        self.assertEqual(matched_old, 1)


if __name__ == '__main__':
    unittest.main()
