"""Exercise P2-adapter records through the fusion CLI, including a rejected clock."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from runtime.common import file_hash, write_json
from runtime.fusion import canonical
from tools.fusion_scenarios import sensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', default='experiments/p3_p2_bridge_v1')
    parser.add_argument('--config', default='configs/fusion_simulation.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'simulation_only': True, 'commands': []}
    try:
        bridge = Path(args.bridge)
        adapter_report = json.loads((bridge / 'report.json').read_text(encoding='utf-8'))
        if adapter_report['status'] != 'complete' or file_hash(bridge / 'records.jsonl') != adapter_report['output_sha256']:
            raise ValueError('Adapter output changed or is incomplete')
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        records = [json.loads(line) for line in (bridge / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(records) != 3:
            raise ValueError('This integration fixture expects exactly three still-video frames')
        records += [sensor(config, stamp) for stamp in (0, 200, 400)]
        inputs = output / 'input.jsonl'
        inputs.write_text(''.join(canonical(r)+'\n' for r in records), encoding='utf-8')
        ticks = output / 'ticks.json'
        write_json(ticks, [0, 100, 200, 300, 400, 500, 1000, 2000, 4000])
        for name, path, expected_code in (('first', inputs, 0), ('repeat', inputs, 0)):
            command = [sys.executable, '-m', 'runtime.fusion', '--input', str(path), '--ticks', str(ticks),
                       '--config', args.config, '--output', str(output / name)]
            completed = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', errors='replace')
            (output / f'{name}.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
            report['commands'].append({'command': command, 'returncode': completed.returncode})
            if completed.returncode != expected_code:
                raise RuntimeError(f'{name} CLI failed; see saved log')
        for name in ('events.jsonl', 'snapshots.jsonl'):
            if (output / 'first' / name).read_bytes() != (output / 'repeat' / name).read_bytes():
                raise AssertionError('CLI replay was not byte-for-byte reproducible')
        states = [json.loads(line) for line in (output / 'first' / 'snapshots.jsonl').read_text(encoding='utf-8').splitlines()]
        if states[0]['latest_record_ids'].get('vision') is not None:
            raise AssertionError('Consumed a future available visual record')
        if states[-1]['risk_assessment_valid'] or states[-1]['health'] != 'UNAVAILABLE':
            raise AssertionError('Stale inputs were treated as current observations')
        if any(s['risk_level'] == 'NORMAL' for s in states):
            raise AssertionError('A short high-floor visual fixture incorrectly established normality')
        bad = copy.deepcopy(records)
        bad[0]['clock_id'] = 'unmapped-clock'
        invalid = output / 'invalid_clock.jsonl'
        invalid.write_text(''.join(canonical(r)+'\n' for r in bad), encoding='utf-8')
        command = [sys.executable, '-m', 'runtime.fusion', '--input', str(invalid), '--ticks', str(ticks),
                   '--config', args.config, '--output', str(output / 'expected_failure')]
        completed = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', errors='replace')
        (output / 'expected_failure.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        failure = json.loads((output / 'expected_failure' / 'report.json').read_text(encoding='utf-8'))
        if completed.returncode == 0 or failure['status'] != 'failed' or 'clock_id' not in failure['error']:
            raise AssertionError('Invalid clock was not explicitly rejected')
        report['commands'].append({'command': command, 'returncode': completed.returncode, 'expected_failure': True})
        report.update(status='passed', byte_identical=True, future_filter='passed', stale_check='passed',
                      invalid_clock='rejected_as_expected', records=len(records), ticks=len(states),
                      events_sha256=file_hash(output / 'first' / 'events.jsonl'),
                      model_sha256=records[0]['model_sha256'],
                      limitation='Existing still-image video fixture + invented sensor values/times. Not a real fire sequence.')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'verification.json', report)


if __name__ == '__main__':
    main()
