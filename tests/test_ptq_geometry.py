"""E0.1 regression tests: GT, letterbox and predictions must share one coordinate space.

These tests were written to FAIL against the pre-fix ptq_eval (load_gt stretched normalized
coordinates to 128x128 while predictions lived on the letterboxed canvas). They assert the
correct post-fix behavior; the captured failure log is the E0.1 evidence.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import ptq_eval  # noqa: E402

DEV_LABEL = ROOT / 'datasets/public_expanded_comparison/labels/val/WEB06156.txt'
DEV_IMAGE = ROOT / 'datasets/public_expanded_comparison/images/val/WEB06156.jpg'


class LetterboxMetaTests(unittest.TestCase):
    def test_landscape_768x432_gain_and_odd_even_padding(self):
        # 768x432 -> scale 1/6 -> 128x72 canvas image, 28 px padded top and bottom
        meta = ptq_eval.LetterboxTransform(768, 432)
        self.assertEqual((meta.gain_x, meta.gain_y), (128 / 768, 72 / 432))
        self.assertEqual((meta.pad_left, meta.pad_top), (0, 28))
        self.assertEqual((meta.nw, meta.nh), (128, 72))

    def test_odd_padding_asymmetric_rounding_matches_framework(self):
        # 232x101: scale = 128/232 -> nw=128, nh=round(101*0.5517)=56, 72px padded vertically
        meta = ptq_eval.LetterboxTransform(232, 101)
        self.assertEqual((meta.nw, meta.nh), (128, 56))
        # gains must be the REALIZED sizes, not the raw scale
        self.assertAlmostEqual(meta.gain_x * 232, meta.nw, places=9)
        self.assertAlmostEqual(meta.gain_y * 101, meta.nh, places=9)

    def test_square_image_has_no_padding(self):
        meta = ptq_eval.LetterboxTransform(128, 128)
        self.assertEqual((meta.gain_x, meta.gain_y, meta.pad_left, meta.pad_top), (1.0, 1.0, 0, 0))

    def test_portrait_padding_on_left_right(self):
        meta = ptq_eval.LetterboxTransform(432, 768)
        self.assertEqual((meta.nw, meta.nh), (72, 128))
        self.assertEqual((meta.pad_left, meta.pad_top), (28, 0))


class CoordinateRoundtripTests(unittest.TestCase):
    def test_roundtrip_original_to_canvas_to_original(self):
        for ow, oh in ((768, 432), (432, 768), (100, 51), (128, 128), (233, 101)):
            with self.subTest(ow=ow, oh=oh):
                meta = ptq_eval.LetterboxTransform(ow, oh)
                box = np.array([ow * 0.1, oh * 0.2, ow * 0.7, oh * 0.9])
                canvas = meta.original_box_to_canvas(box)
                back = meta.canvas_box_to_original(canvas)
                self.assertTrue(np.allclose(back, box, atol=1e-4), f'{back} vs {box}')

    def test_canvas_box_outside_content_maps_to_clipped_original(self):
        meta = ptq_eval.LetterboxTransform(768, 432)
        box = np.array([-10.0, 10.0, 140.0, 80.0])  # spills into the padding columns
        back = meta.canvas_box_to_original(box)
        self.assertTrue(np.all(back[:4:2] >= 0))          # x1, x2 clipped to >= 0
        self.assertTrue(np.all(back[:4:2] <= 768))        # and <= original width
        self.assertTrue(np.all(back[1:4:2] >= 0))


class GroundTruthSpaceTests(unittest.TestCase):
    def test_existing_load_gt_reproduces_the_confirmed_error(self):
        """Document the pre-fix behavior: load_gt(path) stretched labels to 128x128.

        This test pins the OLD signature/behavior as WRONG; after the fix, load_gt takes
        the letterbox metadata and returns ORIGINAL-pixel coordinates.
        """
        if not DEV_LABEL.exists():
            self.skipTest('dev label not available')
        with self.assertRaises(TypeError):
            # old signature load_gt(path) must no longer exist without metadata
            ptq_eval.load_gt(DEV_LABEL)

    def test_load_gt_returns_original_pixel_boxes(self):
        meta = ptq_eval.LetterboxTransform(768, 432)
        gt = ptq_eval.load_gt(DEV_LABEL, meta)
        self.assertEqual(len(gt['boxes']), len(gt['classes']))
        first_line = DEV_LABEL.read_text(encoding='utf-8').splitlines()[0].split()
        c, xc, yc, bw, bh = (float(v) for v in first_line)
        expected = [(xc - bw / 2) * 768, (yc - bh / 2) * 432, (xc + bw / 2) * 768, (yc + bh / 2) * 432]
        self.assertTrue(np.allclose(gt['boxes'][0], expected, atol=1e-6))
        # and the stretched (wrong) values from the pre-fix implementation must NOT appear
        stretched = [(xc - bw / 2) * 128, (yc - bh / 2) * 128, (xc + bw / 2) * 128, (yc + bh / 2) * 128]
        self.assertFalse(np.allclose(gt['boxes'][0], stretched, atol=1e-3))

    def test_perfect_prediction_matches_its_own_gt(self):
        """A prediction that is exactly the GT transformed to canvas must be a TP in
        original space. Under the pre-fix GT this IoU was 0 for non-square images."""
        meta = ptq_eval.LetterboxTransform(768, 432)
        gt = ptq_eval.load_gt(DEV_LABEL, meta)
        if len(gt['boxes']) == 0:
            self.skipTest('no GT boxes')
        pred_canvas = meta.original_box_to_canvas(gt['boxes'][0])
        pred_orig = meta.canvas_box_to_original(pred_canvas)
        iou = ptq_eval.iou_box(pred_orig, gt['boxes'][0])
        self.assertGreater(iou, 0.999)
        from training.error_analysis import match
        result = match([{'xyxy_original_pixels': pred_orig.tolist(), 'confidence': 0.9}],
                       [gt['boxes'][0].tolist()], 0.5)
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['fn'], 0)
        self.assertEqual(result['fp'], 0)


class EmptyAndDegenerateTests(unittest.TestCase):
    def test_empty_labels_and_no_predictions(self):
        gt = {'boxes': np.zeros((0, 4)), 'classes': np.zeros(0, dtype=int)}
        result = ptq_eval.match([], [], 0.5) if hasattr(ptq_eval, 'match') else \
            __import__('training.error_analysis', fromlist=['match']).match([], [], 0.5)
        self.assertEqual(result, {'tp': 0, 'fp': 0, 'fn': 0} or result)
        self.assertEqual(result['tp'], 0)

    def test_empty_labels_with_prediction_counts_fp(self):
        from training.error_analysis import match
        result = match([{'xyxy_original_pixels': [1, 1, 10, 10], 'confidence': 0.9}], [], 0.5)
        self.assertEqual(result['fp'], 1)
        self.assertEqual(result['tp'], 0)


if __name__ == '__main__':
    unittest.main()
