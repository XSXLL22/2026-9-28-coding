"""Wrap completed P2 predictions in an explicit SIMULATED replay timeline.

Never invent acquisition times from frame IDs or inference duration. A manifest is
mandatory even when source timestamps exist, because this adapter does not map real clocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.common import file_hash, write_json
from runtime.fusion import canonical, require, validate_config, validate_record


def adapt(run_dir, manifest_path, config):
    validate_config(config)
    run_dir, manifest_path = Path(run_dir), Path(manifest_path)
    run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    require(run['status'] == 'complete', 'P2 run incomplete')
    vision_config = json.loads((run_dir / 'config_snapshot.json').read_text(encoding='utf-8'))
    original_config = Path(run['args']['config'])
    require(file_hash(original_config) == run['config_sha256'], 'Original vision configuration changed')
    require(json.loads(original_config.read_text(encoding='utf-8')) == vision_config, 'Vision snapshot content changed')
    require(vision_config['node_id'] == config['node_id'], 'Node differs from P2 config')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('schema_version') == 1 and manifest.get('origin') == 'simulation', 'Explicit simulation manifest required')
    timeline = {}
    for row in manifest['frames']:
        require(type(row['frame_id']) is int and row['frame_id'] not in timeline, 'Duplicate/invalid manifest frame ID')
        timeline[row['frame_id']] = row
    supported = sorted({c for c in run['class_mapping'].values() if c is not None})
    predictions = [json.loads(line) for line in (run_dir / 'detections.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(predictions) == run['frame_count'], 'Prediction count mismatch')
    frame_ids = [r['frame_id'] for r in predictions]
    require(len(frame_ids) == len(set(frame_ids)) and set(frame_ids) == set(timeline), 'Manifest must cover each frame exactly once')
    input_hashes = {str(Path(path).resolve()): value for path, value in run['input_hashes'].items()}
    verified_sources = set()
    records = []
    for prediction in predictions:
        require(prediction['model_sha256'] == run['model_sha256'], 'Mixed model versions')
        source = Path(prediction['source']).resolve()
        if str(source) not in verified_sources:
            require(file_hash(source) == input_hashes[str(source)], 'Source image/video changed')
            verified_sources.add(str(source))
        timing = timeline[prediction['frame_id']]
        detections = prediction['detections']
        require(all(d['project_class_id'] is not None for d in detections), 'Unmapped classes cannot be silently discarded')
        record = {'schema_version': 1, 'origin': 'simulation', 'kind': 'vision', 'stream_id': 'vision',
                  'node_id': config['node_id'], 'session_id': config['session_id'], 'clock_id': config['clock_id'],
                  'record_id': f'p2-frame-{prediction["frame_id"]}', 'sample_time_ms': timing['sample_time_ms'],
                  'available_time_ms': timing['available_time_ms'], 'valid': True, 'invalid_reason': None, 'health': 'OK',
                  'frame_id': prediction['frame_id'], 'model_sha256': run['model_sha256'],
                  'run_sha256': file_hash(run_dir / 'run.json'), 'vision_config_sha256': run['config_sha256'],
                  'supported_classes': supported, 'roi_status': vision_config['roi']['status'],
                  'image_width': prediction['image_width'], 'image_height': prediction['image_height'],
                  'confidence_floor': run['args']['conf'], 'detections': detections,
                  'provenance': {'type': 'p2_predictions_on_simulated_timeline', 'source': str(source),
                                 'source_sha256': input_hashes[str(source)],
                                 'source_timestamp_ms': prediction['source_timestamp_ms'],
                                 'timestamp_basis': prediction['timestamp_basis'],
                                 'timeline_sha256': file_hash(manifest_path),
                                 'predictions_sha256': file_hash(run_dir / 'detections.jsonl')}}
        validate_record(record, config)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--timeline', required=True)
    parser.add_argument('--config', default='configs/fusion_simulation.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'simulation_only': True, 'args': vars(args)}
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        records = adapt(args.run, args.timeline, config)
        (output / 'records.jsonl').write_text(''.join(canonical(r) + '\n' for r in records), encoding='utf-8')
        report.update(status='complete', records=len(records), output_sha256=file_hash(output / 'records.jsonl'))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
