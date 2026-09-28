import unittest

from tools.verify_p3_bridge_320 import assert_provenance, build_inputs, risk_summary, tick_schedule


class P3Bridge320Tests(unittest.TestCase):
    def test_tick_schedule_covers_last_availability_and_stale_tail(self):
        ticks = tick_schedule(1000)
        self.assertEqual(ticks[0], 0)
        self.assertEqual(ticks[-1], 1000 + 3000)
        self.assertTrue(all(b - a == 250 for a, b in zip(ticks, ticks[1:-1])))
        self.assertEqual(ticks[-1] - ticks[-2], 3000)

    def test_build_inputs_appends_one_sensor_per_vision_sample_time(self):
        config = {'node_id': 'n', 'session_id': 's', 'clock_id': 'c'}
        vision = [{'kind': 'vision', 'sample_time_ms': 500},
                  {'kind': 'vision', 'sample_time_ms': 0},
                  {'kind': 'vision', 'sample_time_ms': 500}]
        inputs, sensors = build_inputs(vision, config)
        self.assertEqual(sensors, 2)
        sensor_rows = [r for r in inputs if r['kind'] == 'sensor']
        self.assertEqual(sorted(r['sample_time_ms'] for r in sensor_rows), [0, 500])
        self.assertTrue(all(r['origin'] == 'simulation' for r in sensor_rows))

    def test_risk_summary_collapses_repeated_event_transitions(self):
        snapshots = [{'risk_level': None}, {'risk_level': 'ATTENTION'}, {'risk_level': 'WARNING'}]
        events = [{'tick_ms': 500, 'previous_risk_level': None, 'risk_level': 'ATTENTION'},
                  {'tick_ms': 750, 'previous_risk_level': None, 'risk_level': 'ATTENTION'},
                  {'tick_ms': 1000, 'previous_risk_level': 'ATTENTION', 'risk_level': 'WARNING'},
                  {'tick_ms': 1250, 'previous_risk_level': 'WARNING', 'risk_level': 'WARNING'}]
        summary = risk_summary(snapshots, events)
        self.assertEqual(summary['risk_level_counts'], {None: 1, 'ATTENTION': 1, 'WARNING': 1})
        self.assertEqual([(t['tick'], t['previous'], t['new']) for t in summary['transitions']],
                         [(500, None, 'ATTENTION'), (1000, 'ATTENTION', 'WARNING')])

    def test_provenance_requires_candidate_model_and_working_point(self):
        run = {'model_sha256': 'model-a', 'args': {'conf': 0.25}}
        import tempfile
        from pathlib import Path
        from runtime.common import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'run.json').write_text('{}', encoding='utf-8')
            expected_run = file_hash(Path(tmp) / 'run.json')
            records = [{'kind': 'vision', 'model_sha256': 'model-a', 'run_sha256': expected_run, 'confidence_floor': 0.25},
                       {'kind': 'sensor', 'model_sha256': None, 'run_sha256': None, 'confidence_floor': None}]
            self.assertEqual(assert_provenance(records, run, tmp), 'model-a')
            with self.assertRaises(AssertionError):
                assert_provenance([{'kind': 'vision', 'model_sha256': 'other', 'run_sha256': expected_run,
                                    'confidence_floor': 0.25}], run, tmp)
            with self.assertRaises(AssertionError):
                assert_provenance([{'kind': 'vision', 'model_sha256': 'model-a', 'run_sha256': 'x',
                                    'confidence_floor': 0.25}], run, tmp)
            with self.assertRaises(AssertionError):
                assert_provenance([{'kind': 'vision', 'model_sha256': 'model-a', 'run_sha256': expected_run,
                                    'confidence_floor': 0.15}], run, tmp)


if __name__ == '__main__':
    unittest.main()
