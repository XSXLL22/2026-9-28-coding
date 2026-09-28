"""Replay the real 320-candidate detection log through the P3 fusion CLI on a simulated timeline."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from runtime.common import file_hash, write_json
from runtime.fusion import canonical
from tools.fusion_scenarios import sensor

TICK_STEP_MS = 250
STALE_TAIL_MS = 3000


def tick_schedule(last_available_ms):
    ticks = list(range(0, last_available_ms + 1, TICK_STEP_MS))
    return ticks + [ticks[-1] + STALE_TAIL_MS]


def build_inputs(records, config):
    vision_times = sorted({r['sample_time_ms'] for r in records if r['kind'] == 'vision'})
    sensors = [sensor(config, stamp, temperature=25.0) for stamp in vision_times]
    return records + sensors, len(sensors)


def risk_summary(snapshots, events):
    counts = Counter(s['risk_level'] for s in snapshots)
    collapsed = []
    for e in events:
        change = (e.get('previous_risk_level'), e['risk_level'])
        if change[0] == change[1] or (collapsed and collapsed[-1]['change'] == change):
            continue
        collapsed.append({'tick': e['tick_ms'], 'change': change, 'previous': change[0], 'new': change[1]})
    return {'risk_level_counts': dict(counts),
            'transitions': [{k: t[k] for k in ('tick', 'previous', 'new')} for t in collapsed]}


def assert_provenance(records, run, run_dir):
    expected_model = run['model_sha256']
    expected_run = file_hash(Path(run_dir) / 'run.json')
    for r in records:
        if r['kind'] != 'vision':
            continue
        if r['model_sha256'] != expected_model:
            raise AssertionError('Vision record carries a model other than the 320 candidate')
        if r['run_sha256'] != expected_run:
            raise AssertionError('Vision record references a different inference run')
        if r['confidence_floor'] != run['args']['conf']:
            raise AssertionError('Vision confidence floor does not match the inference working point')
    return expected_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', default='experiments/p3_bridge_320_v1')
    parser.add_argument('--run', default='experiments/p2_r13_working_points/inference_S0_conf0.25_nms0.45')
    parser.add_argument('--config', default='configs/fusion_simulation.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {'status': 'running', 'simulation_only': True, 'commands': []}
    try:
        bridge, run_dir = Path(args.bridge), Path(args.run)
        adapter_report = json.loads((bridge / 'report.json').read_text(encoding='utf-8'))
        if adapter_report['status'] != 'complete' or file_hash(bridge / 'records.jsonl') != adapter_report['output_sha256']:
            raise ValueError('Adapter output changed or is incomplete')
        run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if run['status'] != 'complete' or run['args']['imgsz'] != 320:
            raise ValueError('Expected a completed 320-size inference run')
        records = [json.loads(line) for line in (bridge / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        report['model_sha256'] = assert_provenance(records, run, run_dir)
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        inputs, sensor_count = build_inputs(records, config)
        input_path = output / 'input.jsonl'
        input_path.write_text(''.join(canonical(r) + '\n' for r in inputs), encoding='utf-8')
        last_available = max(r['available_time_ms'] for r in inputs)
        ticks = output / 'ticks.json'
        write_json(ticks, tick_schedule(last_available))
        report.update(vision_records=len(records), sensor_records=sensor_count, ticks=len(tick_schedule(last_available)))
        for name in ('first', 'repeat'):
            command = [sys.executable, '-m', 'runtime.fusion', '--input', str(input_path), '--ticks', str(ticks),
                       '--config', args.config, '--output', str(output / name)]
            completed = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', errors='replace',
                                       env={**os.environ, 'PYTHONUTF8': '1'})
            (output / f'{name}.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
            report['commands'].append({'command': command, 'returncode': completed.returncode})
            if completed.returncode:
                raise RuntimeError(f'{name} fusion CLI failed; see saved log')
        for name in ('events.jsonl', 'snapshots.jsonl'):
            if (output / 'first' / name).read_bytes() != (output / 'repeat' / name).read_bytes():
                raise AssertionError('CLI replay was not byte-for-byte reproducible')
        snapshots = [json.loads(line) for line in (output / 'first' / 'snapshots.jsonl').read_text(encoding='utf-8').splitlines()]
        if snapshots[0]['latest_record_ids'].get('vision') is not None:
            raise AssertionError('Consumed a future available visual record')
        final = snapshots[-1]
        if final['risk_assessment_valid'] or final['health'] != 'UNAVAILABLE':
            raise AssertionError('Stale inputs were treated as current observations')
        floor_marks = [s for s in snapshots if 'vision:floor_above_recovery_threshold' in (s.get('reason_codes') or [])]
        if not floor_marks:
            raise AssertionError('Expected vision:floor_above_recovery_threshold marking for the 0.25 confidence floor')
        events = [json.loads(line) for line in (output / 'first' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        summary = risk_summary(snapshots, events)
        bad = copy.deepcopy(inputs)
        bad[0]['clock_id'] = 'unmapped-clock'
        invalid = output / 'invalid_clock.jsonl'
        invalid.write_text(''.join(canonical(r) + '\n' for r in bad), encoding='utf-8')
        command = [sys.executable, '-m', 'runtime.fusion', '--input', str(invalid), '--ticks', str(ticks),
                   '--config', args.config, '--output', str(output / 'expected_failure')]
        completed = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', errors='replace',
                                   env={**os.environ, 'PYTHONUTF8': '1'})
        (output / 'expected_failure.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        failure = json.loads((output / 'expected_failure' / 'report.json').read_text(encoding='utf-8'))
        if completed.returncode == 0 or failure['status'] != 'failed' or 'clock_id' not in failure['error']:
            raise AssertionError('Invalid clock was not explicitly rejected')
        report['commands'].append({'command': command, 'returncode': completed.returncode, 'expected_failure': True})
        detections_per_frame = [len(r['detections']) for r in records]
        report.update(status='passed', byte_identical=True, future_filter='passed', stale_check='passed',
                      floor_mark_snapshots=len(floor_marks), invalid_clock='rejected_as_expected',
                      frames_with_detections=sum(1 for n in detections_per_frame if n),
                      total_detections=sum(detections_per_frame),
                      max_detection_confidence=max((d['confidence'] for r in records for d in r['detections']), default=None),
                      risk_summary=summary, events=len(events),
                      limitation='Real 320-model detections on reused development stills, replayed on an invented uniform timeline with synthetic 25 degC sensors. Not an event-level alarm evaluation and not campus acceptance.')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'verification.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'frames_with_detections', 'total_detections',
                                              'max_detection_confidence', 'byte_identical', 'floor_mark_snapshots')},
                     ensure_ascii=False))
    print(json.dumps(report['risk_summary']['risk_level_counts'], ensure_ascii=False))
    print(json.dumps(report['risk_summary']['transitions'], ensure_ascii=False))


if __name__ == '__main__':
    main()
