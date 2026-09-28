import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_r11_paired_experiment import (
    FROZEN_SEED,
    assert_mosaic_history,
    assert_recipe,
    parse_args,
    recipe_differences,
)


class R11PairedExperimentTests(unittest.TestCase):
    def parse(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return parse_args(argv)

    def test_seed_is_explicit_and_frozen_seed_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse(['--output', 'unused'])
        with self.assertRaises(SystemExit):
            self.parse(['--output', 'unused', '--seed', str(FROZEN_SEED)])
        for value in ('0', '-5'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.parse(['--output', 'unused', '--seed', value])
        self.assertEqual(self.parse(['--output', 'out', '--seed', '20260927']).seed, 20260927)

    def test_recipe_differences_report_both_sides(self):
        differences = recipe_differences({'imgsz': 128, 'seed': 1}, {'imgsz': 320, 'seed': 1, 'extra': 2})
        self.assertEqual(differences, {'imgsz': {'old': 128, 'new': 320}, 'extra': {'old': None, 'new': 2}})

    def test_recipe_check_allows_only_declared_fields(self):
        reference = {'imgsz': 128, 'seed': 7, 'lr0': 0.001, 'close_mosaic': 0, 'name': 'fit'}
        candidate = {'imgsz': 320, 'seed': 7, 'lr0': 0.001, 'close_mosaic': 0, 'name': 'fit', 'save_dir': 'x'}
        self.assertTrue(assert_recipe(reference, candidate, {'imgsz', 'save_dir'}, 'test'))
        with self.assertRaises(AssertionError):
            assert_recipe(reference, {**candidate, 'lr0': 0.01}, {'imgsz', 'save_dir'}, 'test')
        with self.assertRaises(AssertionError):
            assert_recipe(reference, {**candidate, 'seed': 8}, {'imgsz', 'save_dir'}, 'test')

    def test_mosaic_history_requires_full_run_at_expected_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'epochs.jsonl'
            path.write_text('\n'.join(json.dumps({'epoch': e, 'transforms': [{'type': 'Mosaic', 'p': 1.0}]})
                                      for e in range(1, 51)), encoding='utf-8')
            assert_mosaic_history(path)
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            rows[49]['transforms'][0]['p'] = 0.0
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            with self.assertRaises(AssertionError):
                assert_mosaic_history(path)
            short = Path(tmp) / 'short.jsonl'
            short.write_text('\n'.join(json.dumps({'epoch': e, 'transforms': [{'type': 'Mosaic', 'p': 1.0}]})
                                       for e in range(1, 50)), encoding='utf-8')
            with self.assertRaises(AssertionError):
                assert_mosaic_history(short)


if __name__ == '__main__':
    unittest.main()
