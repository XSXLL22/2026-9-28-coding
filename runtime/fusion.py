"""Deterministic, simulation-only single-node fusion and offline JSONL replay.

No device control or production alarm thresholds. Observation time advances dwell;
ticks advance freshness only. Missing data cannot clear a retained risk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

from runtime.common import file_hash, write_json

LEVELS = ('NORMAL', 'ATTENTION', 'WARNING', 'ALARM')
RULE_VERSION = 'simulation-fusion-1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def milliseconds(value):
    return type(value) is int and value >= 0


def text(value):
    return isinstance(value, str) and bool(value.strip())


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_config(config):
    require(config.get('schema_version') == 1 and config.get('simulation_only') is True,
            'Only schema 1 simulation_only configurations are supported')
    require(config.get('rule_version') == RULE_VERSION, 'Unsupported rule_version')
    for key in ('node_id', 'session_id', 'clock_id'):
        require(text(config.get(key)), f'Missing {key}')
    for key in ('max_age_ms', 'max_gap_ms', 'alignment_ms', 'attention_ms', 'warning_ms', 'alarm_ms', 'recovery_ms'):
        require(milliseconds(config.get(key)), f'Invalid {key}')
    require(config['max_age_ms'] > 0 and config['max_gap_ms'] > 0 and config['recovery_ms'] > 0, 'Freshness/gap/recovery must be positive')
    thresholds = config.get('thresholds', {})
    for key in ('visual_clear', 'visual_weak', 'smoke_warning', 'fire_strong', 'temperature_clear', 'temperature_attention', 'temperature_warning'):
        require(number(thresholds.get(key)), f'Invalid threshold {key}')
    require(0 <= thresholds['visual_clear'] < thresholds['visual_weak'] <= thresholds['smoke_warning'] <= 1,
            'Visual thresholds need hysteresis and valid probabilities')
    require(thresholds['visual_weak'] <= thresholds['fire_strong'] <= 1, 'Invalid fire threshold')
    require(thresholds['temperature_clear'] < thresholds['temperature_attention'] < thresholds['temperature_warning'],
            'Temperature thresholds need hysteresis')
    require(config.get('scope') == 'full_frame_simulation', 'ROI rules are not implemented')


def validate_record(record, config):
    require(isinstance(record, dict), 'Record must be an object')
    require(record.get('schema_version') == 1, 'Unsupported record schema')
    for key in ('node_id', 'session_id', 'clock_id'):
        require(record.get(key) == config[key], f'Unmapped or foreign {key}')
    require(record.get('origin') == 'simulation', 'Prototype accepts explicitly simulated records only')
    require(text(record.get('record_id')), 'Missing record_id')
    for key in ('sample_time_ms', 'available_time_ms'):
        require(milliseconds(record.get(key)), f'Invalid {key}')
    require(record['available_time_ms'] >= record['sample_time_ms'], 'Availability precedes sampling')
    require(type(record.get('valid')) is bool, 'valid must be boolean')
    require(record.get('health') in ('OK', 'DEGRADED', 'UNAVAILABLE'), 'Invalid input health')
    require(record.get('invalid_reason') is None if record['valid'] else text(record.get('invalid_reason')),
            'valid/invalid_reason mismatch')
    require(not record['valid'] or record['health'] == 'OK', 'Unhealthy record cannot be valid evidence')
    require(record.get('stream_id') in ('vision', 'ambient_temperature'), 'Unknown stream')
    if record['stream_id'] == 'ambient_temperature':
        require(record.get('kind') == 'sensor' and record.get('sensor_type') == 'ambient_temperature', 'Sensor type mismatch')
        require(record.get('unit') == 'degC', 'Ambient temperature requires degC, not surface temperature or ppm')
        require(text(record.get('sensor_id')), 'Missing sensor_id')
        require(record.get('calibration_status') == 'simulation_only', 'No physical calibration is claimed')
        require(number(record.get('value')) if record['valid'] else record.get('value') is None,
                'Invalid sensor value; missing/invalid values must be null')
    else:
        require(record.get('kind') == 'vision', 'Vision kind mismatch')
        require(number(record.get('confidence_floor')) and 0 <= record['confidence_floor'] <= 1, 'Missing confidence floor')
        require(milliseconds(record.get('frame_id')), 'Invalid frame_id')
        require(sha(record.get('model_sha256')), 'Missing model identity')
        require(sha(record.get('run_sha256')) and sha(record.get('vision_config_sha256')), 'Missing vision provenance')
        require(record.get('roi_status') in ('example_not_site_calibrated', 'simulation_only', 'site_calibrated'), 'Missing ROI status')
        classes = record.get('supported_classes')
        require(isinstance(classes, list) and all(type(c) is int and c in (0, 1, 2) for c in classes)
                and len(classes) == len(set(classes)) and {0, 1}.issubset(classes), 'Vision must declare smoke/fire support')
        width, height = record.get('image_width'), record.get('image_height')
        require(type(width) is int and width > 0 and type(height) is int and height > 0, 'Invalid image dimensions')
        detections = record.get('detections')
        require(isinstance(detections, list), 'detections must be a list')
        require(record['valid'] or not detections, 'Invalid vision record must not carry usable detections')
        for det in detections:
            require(isinstance(det, dict), 'Detection must be an object')
            require(type(det.get('project_class_id')) is int and det['project_class_id'] in classes, 'Unknown or unsupported class')
            require(number(det.get('confidence')) and 0 <= det['confidence'] <= 1, 'Invalid confidence')
            require(det['confidence'] >= record['confidence_floor'], 'Detection below declared confidence floor')
            require(number(det.get('roi_box_area_fraction')) and 0 <= det['roi_box_area_fraction'] <= 1, 'Invalid ROI overlap')
            box = det.get('xyxy_original_pixels')
            require(isinstance(box, list) and len(box) == 4 and all(number(v) for v in box), 'Invalid box')
            require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, 'Box outside image or empty')


class Fusion:
    """One replay session. Required modalities are vision and ambient temperature.

    Strong vision can upgrade risk with temperature missing, while assessment_valid
    remains false to signal incomplete coverage. Recovery requires both modalities.
    """

    def __init__(self, config):
        validate_config(config)
        self.config = copy.deepcopy(config)
        self.config_hash = digest(config)
        self.latest = {}
        self.trackers = {}
        self.level = None
        self.risk_basis = None
        self.last_tick = -1
        self.last_event_key = None
        self.ignored = []

    def usable(self, stream, now):
        record = self.latest.get(stream)
        return record if record and record['valid'] and record['health'] == 'OK' and now - record['sample_time_ms'] <= self.config['max_age_ms'] else None

    def conditions(self, now):
        cfg, thresholds = self.config, self.config['thresholds']
        vision, sensor = self.usable('vision', now), self.usable('ambient_temperature', now)
        scores = {c: max((d['confidence'] for d in vision['detections'] if d['project_class_id'] == c), default=0)
                  if vision else None for c in (0, 1)}
        temperature = sensor['value'] if sensor else None
        result = {}

        def add(name, predicate, records, level, duration):
            if predicate:
                result[name] = (records, level, duration)

        add('weak_visual', vision and max(scores.values()) >= thresholds['visual_weak'], [vision], 1, cfg['attention_ms'])
        add('warm_ambient', sensor and temperature >= thresholds['temperature_attention'], [sensor], 1, cfg['attention_ms'])
        add('persistent_smoke', vision and scores[0] >= thresholds['smoke_warning'], [vision], 2, cfg['warning_ms'])
        add('hot_ambient', sensor and temperature >= thresholds['temperature_warning'], [sensor], 2, cfg['warning_ms'])
        add('strong_fire', vision and scores[1] >= thresholds['fire_strong'], [vision], 3, cfg['alarm_ms'])
        aligned = vision and sensor and abs(vision['sample_time_ms'] - sensor['sample_time_ms']) <= cfg['alignment_ms']
        add('smoke_and_heat', aligned and scores[0] >= thresholds['smoke_warning'] and temperature >= thresholds['temperature_warning'],
            [vision, sensor], 3, cfg['alarm_ms'])
        add('clear', aligned and vision['confidence_floor'] <= thresholds['visual_clear']
            and max(scores.values()) < thresholds['visual_clear'] and temperature < thresholds['temperature_clear'],
            [vision, sensor], 0, cfg['recovery_ms'])
        return result

    def observe(self, now):
        active = self.conditions(now)
        for name in list(self.trackers):
            if name not in active:
                del self.trackers[name]
        for name, (records, level, duration) in active.items():
            stamp = min(r['sample_time_ms'] for r in records)
            if now - stamp > self.config['max_gap_ms']:
                self.trackers.pop(name, None)
                continue
            tracker = self.trackers.get(name)
            if tracker is None or stamp - tracker['end'] > self.config['max_gap_ms'] or now - tracker['end'] > self.config['max_gap_ms']:
                tracker = {'start': stamp, 'end': stamp, 'evidence_ids': [],
                           'start_evidence_ids': sorted({r['record_id'] for r in records})}
                self.trackers[name] = tracker
            # Holding the same frame never advances end or fills a missing interval.
            if stamp >= tracker['end']:
                tracker['end'] = stamp
                tracker['evidence_ids'] = sorted({r['record_id'] for r in records})
            tracker.update(level=level, duration=duration)

    def ingest_group(self, records, now):
        for record in records:
            old = self.latest.get(record['stream_id'])
            if old and record['sample_time_ms'] <= old['sample_time_ms']:
                self.ignored.append({'record_id': record['record_id'], 'reason': 'non_newer_sample'})
                continue
            if old and record['stream_id'] == 'vision' and record['valid'] and record['frame_id'] <= old['frame_id']:
                self.ignored.append({'record_id': record['record_id'], 'reason': 'non_newer_frame'})
                continue
            self.latest[record['stream_id']] = record
        self.observe(now)

    def tick(self, now):
        require(milliseconds(now) and now > self.last_tick, 'Ticks must be strictly increasing nonnegative milliseconds')
        self.last_tick = now
        self.observe(now)
        previous = self.level
        mature = {name: t for name, t in self.trackers.items() if t['end'] - t['start'] >= t['duration']}
        abnormal = {name: t for name, t in mature.items() if t['level'] > 0}
        reasons, evidence = [], []
        if abnormal:
            candidate = max(t['level'] for t in abnormal.values())
            if self.level is None or candidate > self.level:
                self.level = candidate
                supporting = {name: t for name, t in abnormal.items() if t['level'] == candidate}
                self.risk_basis = {'established_tick_ms': now, 'rule_codes': sorted(supporting),
                                   'record_ids': sorted({rid for t in supporting.values() for rid in t['start_evidence_ids'] + t['evidence_ids']})}
            elif candidate < self.level:
                reasons.append('risk_retained_above_current_evidence')
            for name, tracker in sorted(abnormal.items()):
                reasons.append(name)
                evidence.extend(tracker['evidence_ids'] + tracker['start_evidence_ids'])
        elif 'clear' in mature:
            if self.level != 0:
                self.risk_basis = {'established_tick_ms': now, 'rule_codes': ['fresh_sustained_clear'],
                                   'record_ids': sorted(set(mature['clear']['start_evidence_ids'] + mature['clear']['evidence_ids']))}
            self.level = 0
            reasons.append('fresh_sustained_clear')
            evidence.extend(mature['clear']['evidence_ids'] + mature['clear']['start_evidence_ids'])
        else:
            reasons.append('risk_retained_pending_clear' if self.level is not None else 'insufficient_observation')
        health_reasons, ages = [], {}
        for stream in ('vision', 'ambient_temperature'):
            record = self.latest.get(stream)
            ages[stream] = now - record['sample_time_ms'] if record else None
            if record is None:
                health_reasons.append(f'{stream}:missing')
            elif not record['valid'] or record['health'] != 'OK':
                health_reasons.append(f'{stream}:invalid:{record["invalid_reason"]}')
            elif ages[stream] > self.config['max_age_ms']:
                health_reasons.append(f'{stream}:stale')
        visual, ambient = self.usable('vision', now), self.usable('ambient_temperature', now)
        if visual and visual['confidence_floor'] > self.config['thresholds']['visual_clear']:
            health_reasons.append('vision:floor_above_recovery_threshold')
        if visual and ambient and abs(visual['sample_time_ms'] - ambient['sample_time_ms']) > self.config['alignment_ms']:
            health_reasons.append('modalities:unaligned')
        usable_count = sum(self.usable(s, now) is not None for s in ('vision', 'ambient_temperature'))
        health = 'OK' if not health_reasons else 'UNAVAILABLE' if usable_count == 0 else 'DEGRADED'
        reasons.extend(health_reasons)
        valid = health == 'OK' and self.level is not None
        snapshot = {'schema_version': 1, 'node_id': self.config['node_id'], 'session_id': self.config['session_id'],
                    'clock_id': self.config['clock_id'], 'tick_ms': now, 'origin': 'simulation', 'simulation_only': True,
                    'risk_level': LEVELS[self.level] if self.level is not None else None,
                    'risk_assessment_valid': valid, 'health': health, 'reason_codes': sorted(reasons),
                    'evidence_record_ids': sorted(set(evidence)), 'data_age_ms': ages,
                    'retained_risk_basis': copy.deepcopy(self.risk_basis),
                    'latest_record_ids': {s: r['record_id'] for s, r in sorted(self.latest.items())},
                    'rule_version': RULE_VERSION, 'config_sha256': self.config_hash,
                    'model_sha256': self.latest.get('vision', {}).get('model_sha256'),
                    'active_intervals': copy.deepcopy(self.trackers),
                    'unsupported_rules': ['leaf_pile', 'surface_temperature', 'physical_smoke_sensor', 'ROI_distance']}
        key = (self.level, valid, health, tuple(health_reasons))
        event = None
        if key != self.last_event_key:
            event = dict(snapshot, previous_risk_level=LEVELS[previous] if previous is not None else None)
            event['event_id'] = digest(event)
            self.last_event_key = key
        return snapshot, event


def replay(records, ticks, config):
    validate_config(config)
    records = copy.deepcopy(records)
    seen, unique, duplicates, identities = {}, [], [], {}
    for record in records:
        validate_record(record, config)
        rid = record['record_id']
        encoded = canonical(record)
        if rid in seen:
            require(seen[rid] == encoded, f'Conflicting duplicate record_id: {rid}')
            duplicates.append(rid)
            continue
        seen[rid] = encoded
        identity = (record['sensor_id'],) if record['stream_id'] == 'ambient_temperature' else (
            record['model_sha256'], record['run_sha256'], record['vision_config_sha256'], tuple(record['supported_classes']))
        stream = record['stream_id']
        require(stream not in identities or identities[stream] == identity, 'Stream identity changed inside session')
        identities[stream] = identity
        unique.append(record)
    require(isinstance(ticks, list) and bool(ticks), 'At least one explicit tick required')
    require(all(milliseconds(t) for t in ticks) and all(a < b for a, b in zip(ticks, ticks[1:])), 'Invalid tick timeline')
    unique.sort(key=lambda r: (r['available_time_ms'], r['sample_time_ms'], r['record_id']))
    groups = [(time, list(group)) for time, group in itertools.groupby(unique, key=lambda r: r['available_time_ms'])]
    engine, cursor, snapshots, events = Fusion(config), 0, [], []
    for tick in ticks:
        while cursor < len(groups) and groups[cursor][0] <= tick:
            available, group = groups[cursor]
            engine.ingest_group(group, available)
            cursor += 1
        snapshot, event = engine.tick(tick)
        snapshots.append(snapshot)
        if event: events.append(event)
    return {'schema_version': 1, 'status': 'complete', 'simulation_only': True, 'rule_version': RULE_VERSION,
            'config_sha256': digest(config), 'input_records_sha256': digest(records),
            'snapshots': snapshots, 'events': events, 'duplicate_record_ids': duplicates,
            'ignored_records': engine.ignored,
            'unconsumed_future_record_ids': [r['record_id'] for _, group in groups[cursor:] for r in group]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Explicitly simulated records JSONL')
    parser.add_argument('--ticks', required=True, help='JSON array of replay times in milliseconds')
    parser.add_argument('--config', default='configs/fusion_simulation.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'args': vars(args)}
    try:
        records = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        ticks = json.loads(Path(args.ticks).read_text(encoding='utf-8'))
        result = replay(records, ticks, config)
        for name in ('snapshots', 'events'):
            (output / f'{name}.jsonl').write_text(''.join(canonical(row) + '\n' for row in result[name]), encoding='utf-8')
        report.update({k: v for k, v in result.items() if k not in ('snapshots', 'events')})
        report.update(input_sha256=file_hash(args.input), ticks_sha256=file_hash(args.ticks),
                      source_sha256=file_hash(__file__), snapshot_count=len(result['snapshots']), event_count=len(result['events']))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
