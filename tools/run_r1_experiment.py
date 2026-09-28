"""Frozen R1 matched 320 training/inference resolution experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from runtime.common import environment, file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path('experiments/p2_expanded_train')
    report = {'status': 'running', 'hypothesis': 'Only imgsz changes from 128 to 320; original training recipe retained',
              'commands': [], 'stage': 'preflight', 'environment': environment(), 'source_sha256': file_hash(__file__)}

    def run(stage, arguments):
        command = [sys.executable] + arguments
        report['stage'] = stage
        entry = {'stage': stage, 'command': command, 'log': str(output / f'{stage}.log')}
        report['commands'].append(entry)
        write_json(output / 'report.json', report)
        print(f'START {stage}', flush=True)
        with (output / f'{stage}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8='1'))
        entry['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'{stage} failed with code {result.returncode}; inspect {entry["log"]}')
        print(f'PASS {stage}', flush=True)

    try:
        # Exercise the real CSV dependency before spending a training epoch.
        import polars as pl
        report['polars_preflight'] = {'version': pl.__version__,
                                    'reference_rows': pl.read_csv(baseline/'fit/results.csv').height}
        if report['polars_preflight']['reference_rows'] != 50:
            raise AssertionError('Reference CSV row count changed')
        previous = read(baseline / 'report.json')
        if previous['status'] != 'complete' or report['environment']['packages'] != previous['environment']['packages']:
            raise ValueError('Baseline incomplete or package environment changed')
        if file_hash('models/yolov8n.pt') != previous['initial_weights_sha256']:
            raise ValueError('Initial weights changed')
        if file_hash(baseline / 'fit/weights/best.pt') != previous['best_weights_sha256']:
            raise ValueError('Reference baseline weights changed')
        # Check frozen data identities before training; test files are audited, never inferred.
        for audit_path in (baseline / 'data_audit.json', Path('experiments/p2_expanded_new_evaluation/data_audit.json')):
            for row in read(audit_path)['files']:
                source = Path(row['image'])
                parts = list(source.parts)
                parts[len(parts)-1-parts[::-1].index('images')] = 'labels'
                label = Path(*parts).with_suffix('.txt')
                if file_hash(source) != row['image_sha256'] or file_hash(label) != row['label_sha256']:
                    raise ValueError(f'Frozen data changed: {source}')
        report['reference_model_sha256'] = previous['best_weights_sha256']
        report['initial_weights_sha256'] = previous['initial_weights_sha256']
        write_json(output / 'reference_recipe.json', yaml.safe_load((baseline / 'fit/args.yaml').read_text(encoding='utf-8')))
        run('training', ['-m', 'training.baseline', 'train', '--data', 'datasets/public_expanded_v1_prepared/data.yaml',
                        '--weights', 'models/yolov8n.pt', '--output', str(output/'train'), '--epochs', '50', '--batch', '8',
                        '--imgsz', '320', '--eval-sizes', '320', '--device', 'cpu', '--threads', '4',
                        '--seed', '20260926', '--warmup-bias-lr', '0.1', '--close-mosaic', '0'])
        old_recipe = yaml.safe_load((baseline/'fit/args.yaml').read_text(encoding='utf-8'))
        new_recipe = yaml.safe_load((output/'train/fit/args.yaml').read_text(encoding='utf-8'))
        differences = {key: {'old': old_recipe.get(key), 'new': new_recipe.get(key)} for key in old_recipe.keys() | new_recipe.keys()
                       if old_recipe.get(key) != new_recipe.get(key)}
        report['recipe_differences'] = differences
        unexpected = set(differences) - {'imgsz', 'project', 'name', 'save_dir'}
        if unexpected or old_recipe['close_mosaic'] != 0 or new_recipe['close_mosaic'] != 0 or old_recipe['imgsz'] != 128 or new_recipe['imgsz'] != 320 or new_recipe['warmup_bias_lr'] != .1:
            raise AssertionError(f'Not a single-variable training comparison: {differences}')
        new_audit, old_audit = read(output/'train/data_audit.json'), read(baseline/'data_audit.json')
        if new_audit != old_audit:
            raise AssertionError('Training audit differs from frozen baseline')
        history = list(csv.DictReader((output/'train/fit/results.csv').open(encoding='utf-8')))
        if len(history) != 50:
            raise AssertionError('Unexpected training length')
        observed = [json.loads(line) for line in (output/'train/augmentation_epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        if [row['epoch'] for row in observed] != list(range(1, 51)):
            raise AssertionError('Missing or repeated epoch observations')
        for row in observed:
            probabilities = [t['p'] for t in row['transforms'] if t['type'] == 'Mosaic']
            expected = 1.0
            if not probabilities or any(p != expected for p in probabilities):
                raise AssertionError(f'Unexpected Mosaic state at epoch {row["epoch"]}: {probabilities}')
        report['mosaic_state_check'] = 'passed: epochs 1-50 p=1'
        report['actual_recipe_check'] = 'passed'
        report['first_three_epoch_lrs'] = [{key: row[key] for key in ('epoch', 'lr/pg0', 'lr/pg1', 'lr/pg2')} for row in history[:3]]
        weights = str(output/'train/fit/weights/best.pt')
        run('evaluation', ['-m', 'training.baseline', 'evaluate', '--data', 'datasets/public_expanded_comparison/data.yaml',
                          '--weights', weights, '--output', str(output/'evaluation'), '--eval-sizes', '320', '--device', 'cpu'])
        run('inference', ['-m', 'runtime.vision', '--weights', weights, '--source', 'datasets/public_expanded_comparison/images/val',
                         '--output', str(output/'inference'), '--imgsz', '320', '--device', 'cpu', '--conf', '0.25', '--iou', '0.45'])
        run('review', ['-m', 'tools.build_failure_review', '--run', str(output/'inference'), '--audit', str(output/'evaluation/data_audit.json'),
                      '--labels', 'datasets/public_expanded_comparison/labels/val', '--output', str(output/'review'), '--size-reference', '128'])
        run('reference_review', ['-m', 'tools.build_failure_review', '--run', 'experiments/p2_expanded_new_inference',
                                 '--audit', 'experiments/p2_expanded_new_evaluation/data_audit.json',
                                 '--labels', 'datasets/public_expanded_comparison/labels/val',
                                 '--output', str(output/'reference_review'), '--size-reference', '128'])
        run('comparison', ['-m', 'tools.compare_vision_runs', '--old-review', str(output/'reference_review/review.json'),
                          '--new-review', str(output/'review/review.json'), '--output', str(output/'comparison.json'), '--html', '--allow-cross-size'])
        report['candidate_model_sha256'] = file_hash(weights)
        report['comparison'] = read(output/'comparison.json')
        report['reference_validation'] = read('experiments/p2_expanded_new_evaluation/report.json')['validation']['128']
        report['candidate_validation'] = read(output/'evaluation/report.json')['validation']['320']
        report.update(status='complete', stage='complete',
                      limitations=['One seed, historical reference training; no multi-seed stability claim.',
                                   '109-image development validation, not independent test or campus acceptance.',
                                   'No production model promotion or threshold change performed.'])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output/'report.json', report)


if __name__ == '__main__':
    main()
