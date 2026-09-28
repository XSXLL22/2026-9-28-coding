import unittest
from tools.build_failure_review import ROOT, normalized_input_hashes


class FailureReviewTests(unittest.TestCase):
    def test_relative_and_absolute_input_paths_have_identical_identity(self):
        relative = 'datasets/public_pilot_prepared/images/val/example.jpg'
        absolute = str((ROOT / relative).resolve())
        self.assertEqual(normalized_input_hashes({relative: 'sha256'}),
                         normalized_input_hashes({absolute: 'sha256'}))
