"""E1.1 numeric-domain tests with hand-computed expectations (contract v1.3 §4/§6).

Covers: int8 abs(-128) widening, per-channel accumulation bound exact-fit and rejection,
multiplier normalization boundaries (n=0 legal, negative n rejected, shift cap), the
contract's round-half-up table, shift=0 passthrough, and bias rejection before pack write.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import requant_channel, requant_scalar, saturate_int8  # noqa: E402
from training.exporter import Node  # noqa: E402
from training.pack import MAX_SHIFT, accumulation_bound, multiplier_shift  # noqa: E402


def make_node(cout, cin, kh=1, kw=1, name='t'):
    return Node(name=name, op='conv1x1_s1', op_class='PL_V1', inputs=[0], output=1,
                cin=cin, cout=cout, kh=kh, kw=kw)


class AccumulationBoundTests(unittest.TestCase):
    def test_abs_widening_counts_minus_128_as_128(self):
        # one -128 weight: sum|qw| = 128 -> bound = 128*128 + |qb|
        node = make_node(1, 1)
        qw = np.array([[-128]], dtype=np.int8)
        self.assertEqual(accumulation_bound(node, qw, np.array([0], dtype=np.int32)), 128 * 128)

    def test_exact_fit_at_int31_max_is_accepted(self):
        # 131071 terms of -128 + 1 term of +127: sum|qw| = 16777215
        # B = 127 + 128*16777215 = 2147483647 == 2^31 - 1  (hand-computed exact fit)
        terms = [-128] * 131071 + [127]
        node = make_node(1, len(terms))
        qw = np.array([terms], dtype=np.int8)
        qb = np.array([127], dtype=np.int32)
        self.assertEqual(accumulation_bound(node, qw, qb), 2 ** 31 - 1)

    def test_one_weight_past_exact_fit_is_rejected(self):
        # 131072 terms of -128: sum|qw| = 16777216 -> B = 2147483776 > 2^31 - 1
        terms = [-128] * 131072
        node = make_node(1, len(terms))
        qw = np.array([terms], dtype=np.int8)
        with self.assertRaises(ValueError):
            accumulation_bound(node, qw, np.array([0], dtype=np.int32))

    def test_bias_alone_can_violate_int31(self):
        # |qb| = 2^31 exceeds int32 storage AND pushes the bound past int31; zero weights
        # isolate the bias term. (The pack builder rejects such a bias before int32 cast.)
        node = make_node(1, 1)
        qw = np.array([[0]], dtype=np.int8)
        with self.assertRaises(ValueError):
            accumulation_bound(node, qw, np.array([2 ** 31], dtype=np.int64))

    def test_bias_at_int31_max_is_accepted(self):
        node = make_node(1, 1)
        qw = np.array([[0]], dtype=np.int8)
        self.assertEqual(accumulation_bound(node, qw, np.array([2 ** 31 - 1], dtype=np.int64)), 2 ** 31 - 1)

    def test_int32_overflow_does_not_defeat_the_check(self):
        # even with int8 abs(-128) == -128 in the raw array, the widened bound must reject
        node = make_node(1, 131072)
        qw = np.full((1, 131072), -128, dtype=np.int8)
        with self.assertRaises(ValueError):
            accumulation_bound(node, qw, np.array([0], dtype=np.int32))


class MultiplierShiftTests(unittest.TestCase):
    def test_unit_scale(self):
        m, n, rel = multiplier_shift(1.0)
        self.assertEqual((m, n), (2 ** 30, 30))
        self.assertLessEqual(rel, 2 ** -29)

    def test_half_scale(self):
        m, n, _ = multiplier_shift(0.5)
        self.assertEqual((m, n), (2 ** 30, 31))

    def test_shift_zero_is_legal_for_large_scale(self):
        m, n, _ = multiplier_shift(2 ** 30)
        self.assertEqual((m, n), (2 ** 30, 0))

    def test_scale_implying_negative_shift_rejected(self):
        with self.assertRaises(ValueError):
            multiplier_shift(2 ** 31)

    def test_absurdly_small_scale_rejected(self):
        with self.assertRaises(ValueError):
            multiplier_shift(1e-12)


class RequantRoundingTests(unittest.TestCase):
    def test_contract_round_half_up_table(self):
        acc = np.array([[[7]], [[-5]], [[-6]], [[-7]]], dtype=np.int64)
        m = np.array([1, 1, 1, 1], dtype=np.int64)
        shift = np.array([2, 2, 2, 2], dtype=np.int64)
        y = requant_channel(acc, m, shift)
        self.assertEqual(y.ravel().tolist(), [2, -1, -1, -2])

    def test_shift_zero_passes_value_through(self):
        acc = np.array([[[-7]], [[9]]], dtype=np.int64)
        y = requant_channel(acc, np.array([1, 1], dtype=np.int64), np.array([0, 0], dtype=np.int64))
        self.assertEqual(y.ravel().tolist(), [-7, 9])

    def test_negative_shift_rejected(self):
        with self.assertRaises(ValueError):
            requant_channel(np.array([[[1]]], dtype=np.int64), np.array([1], dtype=np.int64),
                            np.array([-1], dtype=np.int64))

    def test_scalar_requant_matches_channel_semantics(self):
        self.assertEqual(int(requant_scalar(np.array([[7]], dtype=np.int8), 1, 2)[0, 0]), 2)
        self.assertEqual(int(requant_scalar(np.array([[-5]], dtype=np.int8), 1, 2)[0, 0]), -1)

    def test_saturation_both_ends(self):
        self.assertEqual(saturate_int8(np.int64(300)), 127)
        self.assertEqual(saturate_int8(np.int64(-300)), -128)


if __name__ == '__main__':
    unittest.main()
