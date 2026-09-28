import json
import unittest
from pathlib import Path

from runtime.fusion import replay, validate_record
from tools.fusion_scenarios import sensor, vision, scenarios


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/fusion_simulation.json').read_text(encoding='utf-8'))

    def run_records(self, records, ticks):
        return replay(records, ticks, self.config)

    def test_missing_is_unknown(self):
        result = self.run_records([], [0, 5000])
        self.assertIsNone(result['snapshots'][-1]['risk_level'])
        self.assertEqual(result['snapshots'][-1]['health'], 'UNAVAILABLE')
        self.assertIsNone(result['snapshots'][-1]['retained_risk_basis'])
        self.assertFalse(result['snapshots'][-1]['risk_assessment_valid'])

    def test_single_source_alarm_despite_missing_sensor_and_stale_does_not_clear(self):
        result = self.run_records([vision(self.config, t, fire=.95) for t in (0, 500, 1000)], [0, 500, 1000, 4000])
        self.assertEqual([s['risk_level'] for s in result['snapshots']], [None, 'ATTENTION', 'ALARM', 'ALARM'])
        self.assertFalse(result['snapshots'][2]['risk_assessment_valid'])
        self.assertEqual(result['snapshots'][-1]['health'], 'UNAVAILABLE')
        self.assertEqual(result['snapshots'][-1]['retained_risk_basis']['rule_codes'], ['strong_fire'])

    def test_repeated_ticks_and_duplicate_records_do_not_create_duration(self):
        record = vision(self.config, 0, fire=.99)
        result = self.run_records([record] * 10, [0, 500, 1000, 1500, 5000])
        self.assertTrue(all(s['risk_level'] is None for s in result['snapshots']))
        self.assertEqual(len(result['duplicate_record_ids']), 9)

    def test_conflicting_duplicate_is_error(self):
        record = sensor(self.config, 0)
        changed = dict(record, value=100)
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.run_records([record, changed], [0])

    def test_gap_breaks_continuity_even_without_intermediate_tick(self):
        records = [vision(self.config, t, fire=.95) for t in (0, 2000)]
        result = self.run_records(records, [2000])
        self.assertIsNone(result['snapshots'][0]['risk_level'])

    def test_future_available_not_consumed_and_stale_late_record_is_not_used(self):
        record = vision(self.config, 0, fire=.95, available=2000)
        result = self.run_records([record], [0, 1000])
        self.assertEqual(result['unconsumed_future_record_ids'], [record['record_id']])
        result = self.run_records([record], [0, 2000])
        self.assertIsNone(result['snapshots'][-1]['risk_level'])
        self.assertEqual(result['snapshots'][-1]['health'], 'UNAVAILABLE')

    def test_delayed_sample_cannot_bridge_expired_interval_by_omitting_ticks(self):
        records = [vision(self.config, 0, fire=.9), vision(self.config, 1000, fire=.9, available=2000)]
        sparse = self.run_records(records, [0, 2000])['snapshots'][-1]
        dense = self.run_records(records, [0, 1500, 2000])['snapshots'][-1]
        self.assertEqual(sparse['risk_level'], dense['risk_level'])
        self.assertIsNone(sparse['risk_level'])

    def test_out_of_order_arrival_cannot_overwrite_newer_sample(self):
        records = [vision(self.config, 1000), vision(self.config, 0, fire=.9, available=1500)]
        result = self.run_records(records, [1000, 1500])
        self.assertEqual(result['ignored_records'][0]['reason'], 'non_newer_sample')
        self.assertEqual(result['snapshots'][-1]['latest_record_ids']['vision'], 'vision-1000')

    def test_same_sample_different_id_does_not_accumulate(self):
        first = vision(self.config, 0, fire=.9)
        second = dict(first, record_id='repackaged', available_time_ms=1000)
        result = self.run_records([first, second], [0, 1000])
        self.assertIsNone(result['snapshots'][-1]['risk_level'])
        self.assertEqual(len(result['ignored_records']), 1)

    def test_repeated_frame_id_with_new_timestamps_does_not_accumulate(self):
        records = [dict(vision(self.config, t, fire=.9), frame_id=0) for t in (0, 500, 1000)]
        result = self.run_records(records, [0, 500, 1000])
        self.assertIsNone(result['snapshots'][-1]['risk_level'])
        self.assertEqual([r['reason'] for r in result['ignored_records']], ['non_newer_frame', 'non_newer_frame'])

    def test_fresh_clear_required_for_recovery(self):
        records, ticks = scenarios(self.config)['fresh_recovery']
        result = self.run_records(records, ticks)
        self.assertEqual(result['snapshots'][-2]['risk_level'], 'ALARM')
        self.assertEqual(result['snapshots'][-1]['risk_level'], 'NORMAL')
        self.assertTrue(result['snapshots'][-1]['risk_assessment_valid'])

    def test_mixed_time_sources_do_not_form_combination(self):
        records = [vision(self.config, t, smoke=.7) for t in (0, 500, 1000)] + [sensor(self.config, 0, 65)]
        result = self.run_records(records, [0, 500, 1000])
        self.assertNotEqual(result['snapshots'][-1]['risk_level'], 'ALARM')
        self.assertIn('modalities:unaligned', result['snapshots'][-1]['reason_codes'])

    def test_simultaneous_group_order_does_not_change_events(self):
        records, ticks = scenarios(self.config)['smoke_heat_combination']
        first = self.run_records(records, ticks)
        second = self.run_records(list(reversed(records)), ticks)
        self.assertEqual(first['events'], second['events'])
        self.assertEqual(first['snapshots'][-1]['risk_level'], 'ALARM')

    def test_high_detection_floor_blocks_normal_but_not_strong_alarm(self):
        records = [r for t in range(0, 3500, 500) for r in (dict(vision(self.config, t), confidence_floor=.25), sensor(self.config, t))]
        result = self.run_records(records, [0, 3000])
        self.assertIsNone(result['snapshots'][-1]['risk_level'])
        self.assertIn('vision:floor_above_recovery_threshold', result['snapshots'][-1]['reason_codes'])
        records = [dict(vision(self.config, t, fire=.9), confidence_floor=.25) for t in (0, 500, 1000)]
        self.assertEqual(self.run_records(records, [1000])['snapshots'][0]['risk_level'], 'ALARM')

    def test_invalid_observation_interrupts_persistence(self):
        bad = dict(vision(self.config, 500), valid=False, health='UNAVAILABLE', invalid_reason='camera_error')
        records = [vision(self.config, 0, fire=.9), bad, vision(self.config, 1000, fire=.9)]
        self.assertIsNone(self.run_records(records, [1000])['snapshots'][0]['risk_level'])

    def test_camera_fault_with_same_frame_id_is_not_hidden_by_frame_deduplication(self):
        initial = vision(self.config, 0, fire=.9)
        fault = dict(vision(self.config, 500), frame_id=0, valid=False, health='UNAVAILABLE', invalid_reason='camera_error')
        result = self.run_records([initial, fault], [0, 500])
        self.assertEqual(result['snapshots'][-1]['health'], 'UNAVAILABLE')
        self.assertIn('vision:invalid:camera_error', result['snapshots'][-1]['reason_codes'])
        self.assertFalse(result['ignored_records'])

    def test_validation_rejects_bad_values_units_clocks_and_classes(self):
        variants = [dict(sensor(self.config, 0), value=float('nan')), dict(sensor(self.config, 0), value=True),
                    dict(sensor(self.config, 0), unit='ppm'), dict(sensor(self.config, 0), value=None),
                    dict(sensor(self.config, 0), sample_time_ms=2, available_time_ms=1),
                    dict(sensor(self.config, 0), clock_id='unmapped-clock'),
                    dict(sensor(self.config, 0), origin='measured'),
                    dict(sensor(self.config, 0), valid=False, invalid_reason='missing', value=0),
                    dict(vision(self.config, 0), supported_classes=[0])]
        wrong_class = vision(self.config, 0, fire=.9)
        wrong_class['detections'][0]['project_class_id'] = 2
        variants.append(wrong_class)
        for record in variants:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    validate_record(record, self.config)

    def test_identity_change_rejected(self):
        first, second = vision(self.config, 0), vision(self.config, 500)
        second['model_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_records([first, second], [0, 500])

    def test_exact_replay_has_identical_event_ids(self):
        records, ticks = scenarios(self.config)['fresh_recovery']
        self.assertEqual(self.run_records(records, ticks), self.run_records(records, ticks))

    def test_warning_and_sensor_attention_are_reachable(self):
        records = [vision(self.config, t, smoke=.7) for t in range(0, 2000, 500)]
        self.assertEqual(self.run_records(records, [0, 1500])['snapshots'][-1]['risk_level'], 'WARNING')
        records = [sensor(self.config, t, 50) for t in (0, 500)]
        self.assertEqual(self.run_records(records, [0, 500])['snapshots'][-1]['risk_level'], 'ATTENTION')

    def test_reject_invalid_ticks_and_non_simulation_config(self):
        for ticks in ([], [0, 0], [100, 0], [True], [-1]):
            with self.assertRaises(ValueError): self.run_records([], ticks)
        self.config['simulation_only'] = False
        with self.assertRaises(ValueError): self.run_records([], [0])


if __name__ == '__main__':
    unittest.main()
