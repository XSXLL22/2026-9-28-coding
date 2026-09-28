import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import quantize_input, run_pack  # noqa: E402
from pack_reader import read_pack  # noqa: E402

PACK = ROOT / 'training/export/p4_baseline128_v1/model_pack.bin'
GOLDEN = ROOT / 'tests/golden'


class PackReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = read_pack(PACK)

    def test_pack_manifest_and_structure(self):
        pack = self.pack
        self.assertEqual(pack.meta['model_id'], 'p4-baseline128')
        self.assertEqual(pack.meta['format_version'], 'p4pkb-1')
        self.assertEqual(len(pack.nodes), 127)
        self.assertEqual(pack.tensor_shapes[0], (3, 128, 128))

    def test_conv_params_shapes_and_ranges(self):
        for node in self.pack.nodes:
            if not node.op.startswith('conv'):
                continue
            self.assertEqual(node.qw.shape, (node.cout, node.cin, node.kh, node.kw))
            self.assertEqual(node.qb.shape, (node.cout,))
            self.assertEqual(len(node.M), node.cout)
            self.assertTrue(np.all(node.M >= 2 ** 30) and np.all(node.M < 2 ** 31))
            self.assertTrue(np.all(node.shift >= 0))
            if node.act == 'silu':
                self.assertEqual(len(node.silu_lut), 256)


class IntReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = read_pack(PACK)

    def test_input_quantization_matches_contract(self):
        rgb = np.array([[[0], [127], [128], [255]]], dtype=np.uint8)  # single-row image
        q = quantize_input(rgb)
        # round_half_up(pixel/255 * 127): 127 -> 63, 128 -> 64, 255 -> 127
        self.assertEqual(q.ravel().tolist(), [0, 63, 64, 127])

    def test_golden_v01_reproduces_frozen_expected_outputs(self):
        info = self._manifest()['vectors']['v01_normal_random']
        x = np.fromfile(GOLDEN / 'v01_normal_random/input.bin', dtype=np.int8).reshape(self.pack.tensor_shapes[0])
        outputs, _ = run_pack(self.pack, x)
        for node_name, meta in info['expected'].items():
            expected = (GOLDEN / 'v01_normal_random' / meta['path']).read_bytes()
            self.assertEqual(outputs[node_name].astype(np.int8).tobytes(), expected, node_name)

    def test_determinism(self):
        x = np.fromfile(GOLDEN / 'v01_normal_random/input.bin', dtype=np.int8).reshape(self.pack.tensor_shapes[0])
        a, _ = run_pack(self.pack, x)
        b, _ = run_pack(self.pack, x)
        for name in a:
            self.assertTrue(np.array_equal(a[name], b[name]), name)

    def test_scope_outputs_present(self):
        x = np.fromfile(GOLDEN / 'v01_normal_random/input.bin', dtype=np.int8).reshape(self.pack.tensor_shapes[0])
        outputs, _ = run_pack(self.pack, x)
        scope_tids = set(self.pack.meta['graph']['scope_outputs'])
        scope_nodes = [n.name for n in self.pack.nodes if n.output in scope_tids]
        self.assertEqual(len(scope_nodes), 6)
        for name in scope_nodes:
            self.assertEqual(outputs[name].dtype, np.int8)

    def _manifest(self):
        return __import__('json').loads((GOLDEN / 'manifest.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
