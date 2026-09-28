import unittest

from tools.diagnose_p2 import score_records, bucket


class DiagnosticTests(unittest.TestCase):
    def test_low_confidence_and_localization_are_separated(self):
        targets = [{'class_id': 1, 'target_id': 0, 'xyxy': [0, 0, 10, 10], 'size_bucket': 'lt8'},
                   {'class_id': 1, 'target_id': 1, 'xyxy': [20, 20, 30, 30], 'size_bucket': 'ge16'}]
        preds = [{'class_id': 1, 'confidence': .1, 'xyxy_original_pixels': [0, 0, 10, 10]},
                 {'class_id': 1, 'confidence': .9, 'xyxy_original_pixels': [20, 20, 40, 40]}]
        records = [{'source': 'synthetic', 'targets': targets, 'predictions': preds}]
        report = score_records(records, .25)
        self.assertEqual(report['miss_reasons'], {'low_confidence_candidate': 1, 'localization_overlap': 1})
        self.assertEqual(score_records(records, .05)['per_class']['fire']['tp'], 1)
        self.assertEqual(report['size_counts_reference_128']['fire']['lt8']['gt'], 1)

    def test_duplicate_and_background_fp_do_not_inflate_recall(self):
        target = {'class_id': 0, 'target_id': 0, 'xyxy': [0, 0, 10, 10], 'size_bucket': 'ge16'}
        preds = [{'class_id': 0, 'confidence': s, 'xyxy_original_pixels': [0, 0, 10, 10]} for s in (.8, .7)]
        result = score_records([{'source': 'positive', 'targets': [target], 'predictions': preds},
                                {'source': 'negative', 'targets': [], 'predictions': preds[:1]}], .25)
        self.assertEqual(result['per_class']['smoke']['tp'], 1)
        self.assertEqual(result['per_class']['smoke']['fp'], 2)
        self.assertEqual(result['negative_images_with_detection'], 1)
        self.assertEqual(result['fp_reasons'], {'duplicate_or_matching_competition': 1, 'background_or_unannotated': 1})

    def test_reference_bucket_boundaries(self):
        self.assertEqual([bucket(s) for s in (7.9, 8, 15.9, 16)], ['lt8', '8to16', '8to16', 'ge16'])
