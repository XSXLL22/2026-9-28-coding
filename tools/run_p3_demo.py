"""Generate deterministic simulated replay artifacts; no physical fire experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.common import file_hash, write_json
from runtime.fusion import canonical, replay
from tools.fusion_scenarios import scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/fusion_simulation.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'simulation_only': True, 'scenarios': {}}
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        write_json(output / 'config_snapshot.json', config)
        for name, (records, ticks) in scenarios(config).items():
            directory = output / name
            directory.mkdir()
            (directory / 'input.jsonl').write_text(''.join(canonical(r) + '\n' for r in records), encoding='utf-8')
            write_json(directory / 'ticks.json', ticks)
            result = replay(records, ticks, config)
            repeated = replay(records, ticks, config)
            if result != repeated:
                raise AssertionError(f'Non-deterministic replay: {name}')
            for field in ('snapshots', 'events'):
                (directory / f'{field}.jsonl').write_text(''.join(canonical(r) + '\n' for r in result[field]), encoding='utf-8')
            write_json(directory / 'report.json', {k: v for k, v in result.items() if k not in ('snapshots', 'events')})
            report['scenarios'][name] = {'records': len(records), 'ticks': len(ticks), 'events': len(result['events']),
                                         'repeat_identical': True, 'final': result['snapshots'][-1],
                                         'events_sha256': file_hash(directory / 'events.jsonl')}
        report.update(status='complete', source_sha256=file_hash('runtime/fusion.py'),
                      limitation='Synthetic fixtures only; no real sensor accuracy, video event recall, latency or FPGA measurement.')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
