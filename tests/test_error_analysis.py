import unittest
from training.error_analysis import iou, match, match_details


class ErrorAnalysisTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)

    def test_duplicate_prediction_is_false_positive(self):
        predictions = [{"confidence": score, "xyxy_original_pixels": [0, 0, 10, 10]} for score in (0.9, 0.8)]
        self.assertEqual(match(predictions, [[0, 0, 10, 10]], 0.5), {"tp": 1, "fp": 1, "fn": 0})

    def test_miss_and_wrong_location(self):
        predictions = [{"confidence": 0.9, "xyxy_original_pixels": [20, 20, 30, 30]}]
        self.assertEqual(match(predictions, [[0, 0, 10, 10]], 0.5), {"tp": 0, "fp": 1, "fn": 1})

    def test_details_keep_original_indices_after_confidence_sort(self):
        predictions = [{"confidence": score, "xyxy_original_pixels": [0, 0, 10, 10]} for score in (0.2, 0.9)]
        result = match_details(predictions, [[0, 0, 10, 10], [20, 20, 30, 30]], 0.5)
        self.assertEqual(result["pairs"], [{"prediction_index": 1, "target_index": 0, "iou": 1.0}])
        self.assertEqual(result["fp_indices"], [0])
        self.assertEqual(result["fn_indices"], [1])


if __name__ == "__main__":
    unittest.main()
