import json
import unittest
from pathlib import Path

import numpy as np
import torch

from training.exporter import (
    CONTRACT_PATH,
    Graph,
    Node,
    Tracer,
    fold_conv_bn,
    load_detection_model,
    op_audit,
    quantize_weights,
    round_half_up,
    run_float_executor,
    saturate_int8,
    verify_against_original,
)

WEIGHTS = 'experiments/p2_expanded_train/fit/weights/best.pt'


class RoundingAndFoldTests(unittest.TestCase):
    def test_round_half_up_matches_contract_examples(self):
        # docs/定点格式说明.md §4 table: (t + 2^(n-1)) >> n with n=2 equals floor(t/4 + 0.5)
        for t, expected in ((9, 2), (7, 2), (-5, -1), (-6, -1), (-7, -2)):
            with self.subTest(t=t):
                got = int(round_half_up(np.array([t / 4]))[0])
                shifted = (t + (1 << 1)) >> 2
                self.assertEqual(got, expected)
                self.assertEqual(shifted, expected)

    def test_fold_bn_formula_includes_mean_subtraction(self):
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(2, 3, 3, padding=1)
        bn = torch.nn.BatchNorm2d(3).eval()
        with torch.no_grad():
            bn.running_mean.fill_(1.0)
            bn.running_var.fill_(4.0)
            bn.weight.fill_(2.0)
            bn.bias.fill_(0.5)
        w_fold, b_fold = fold_conv_bn(conv, bn)
        x = torch.randn(1, 2, 8, 8)
        direct = torch.nn.functional.conv2d(x, conv.weight, conv.bias, padding=1)
        expected = bn(direct)
        got = torch.nn.functional.conv2d(x, w_fold, b_fold, padding=1)
        self.assertLess(float((expected - got).abs().max()), 1e-5)
        # mean≠0 must matter: a formula without (b0-mean) cannot reproduce this case
        wrong = bn.bias + conv.bias * bn.weight / (bn.running_var + bn.eps).sqrt()
        self.assertGreater(float((wrong - b_fold).abs().max()), 0.1)

    def test_weight_quantization_properties(self):
        w = torch.randn(8, 4, 3, 3, dtype=torch.float64)
        qw, sw = quantize_weights(w)
        self.assertEqual(qw.dtype, np.int8)
        np.testing.assert_allclose(np.abs(w.numpy()).max(axis=(1, 2, 3)) / 127.0, sw, rtol=1e-6)
        self.assertEqual(int(np.abs(qw).max()), 127)
        deq = qw.astype(np.float64) * sw.astype(np.float64)[:, None, None, None]
        self.assertLess(float(np.abs(w.numpy() - deq).max()), sw.max() / 2 + 1e-12)

    def test_saturate_int8(self):
        np.testing.assert_array_equal(saturate_int8(np.array([200, -200, 3])), np.array([127, -128, 3], dtype=np.int8))


class GraphAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_detection_model(WEIGHTS)
        cls.tracer = Tracer(cls.model.model)
        cls.graph = cls.tracer.trace()
        cls.x = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(20260928))

    def test_trace_structure_matches_checkpoint_audit(self):
        graph = self.graph
        audit = op_audit(graph)
        self.assertEqual(audit['problems'], [])
        self.assertEqual(audit['conv_nodes'], 63)          # 57 Conv wrappers + 6 bare head convs
        self.assertEqual(audit['silu_activations'], 57)
        self.assertEqual(audit['op_counts'].get('add'), 6)  # backbone C2f residuals [1,2,2,1]
        self.assertEqual(audit['op_counts'].get('upsample_nearest_2x'), 2)
        self.assertEqual(audit['op_counts'].get('maxpool_5x5'), 3)
        self.assertEqual(len(graph.scope_outputs), 6)       # 3 scales x (box, cls)
        scope_names = audit['scope_output_nodes']
        self.assertTrue(all(name.startswith('model.22.') for name in scope_names))

    def test_shapes_are_static_and_consistent(self):
        graph = self.graph
        for node in graph.nodes:
            shape = graph.tensor_shapes[node.output]
            self.assertTrue(all(v > 0 for v in shape), f'{node.name}: bad shape {shape}')
        x0 = torch.zeros(1, *graph.input_shape)
        outputs, cache = run_float_executor(graph, x0)
        for node in graph.nodes:
            got = tuple(cache[node.output].shape[1:])
            self.assertEqual(got, graph.tensor_shapes[node.output], f'{node.name} shape drift')

    def test_float_executor_matches_original_framework(self):
        verification = verify_against_original(self.model, self.graph, self.x)
        self.assertLessEqual(verification['max_abs_diff'], 1e-4)
        self.assertEqual(verification['compared_nodes'], 72)   # every Conv + maxpool/upsample/concat module

    def test_repeatability_of_executor(self):
        a, _ = run_float_executor(self.graph, self.x)
        b, _ = run_float_executor(self.graph, self.x)
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_contract_version_is_current(self):
        # Version is pinned on purpose: a contract bump must be a conscious edit here
        # (E1 raised 1.2 -> 1.3 by rewriting the accumulation bound and the shift rules).
        contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
        self.assertEqual(contract['contract_version'], 'p4-contract-1.3')
        notes = {a['version']: a['note'] for a in contract['amendment_history']}
        self.assertIn('accumulation bound', notes['p4-contract-1.3'])
        self.assertIn('max-pool', notes['p4-contract-1.2'])


if __name__ == '__main__':
    unittest.main()
