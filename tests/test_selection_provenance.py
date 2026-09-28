import unittest
from tools.inspect_a2_selection import infer_selection


class SelectionProvenanceTests(unittest.TestCase):
    def test_uses_stored_metrics_not_full_history_length(self):
        metrics = dict(zip(('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'), (.451234567, .321234567, .311234567, .151234567)))
        row = {'epoch': '33', **{key: format(value, '.6g') for key,value in metrics.items()}}
        last = dict(row, epoch='50', **{'metrics/mAP50-95(B)': '0.14'})
        self.assertEqual(infer_selection(metrics, [row,last])['epoch'], 33)

    def test_ambiguous_or_missing_metrics_do_not_guess_epoch(self):
        keys = ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)')
        metrics = {key:.5 for key in keys}
        row = {'epoch':'1', **{key:'0.5' for key in keys}}
        self.assertIsNone(infer_selection(metrics, [row,dict(row,epoch='2')])['epoch'])
        self.assertIsNone(infer_selection({}, [row])['epoch'])
