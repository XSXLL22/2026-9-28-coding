"""R1.1 paired-seed repeat: one new fixed seed, 128 and 320 arms, single variable imgsz.

The frozen R1 runner (tools/run_r1_experiment.py, seed=20260926) is intentionally
left untouched. This entry point trains BOTH arms at one explicitly provided new
seed so the 320 gain is compared against a seed-matched 128 reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from runtime.common import environment, file_hash, write_json

FROZEN_SEED = 20260926
DIRECTORY_FIELDS = {'project', 'name', 'save_dir'}
TRAIN_DATA = 'datasets/public_expanded_v1_prepared/data.yaml'
COMMON_DATA = 'datasets/public_expanded_comparison/data.yaml'
COMMON_LABELS = 'datasets/public_expanded_comparison/labels/val'
BASELINE = Path('experiments/p2_expanded_train')
OLD_128_REVIEW = 'experiments/p2_r1_resolution320_retry1/reference_review/review.json'
OLD_320_REVIEW = 'experiments/p2_r1_resolution320_retry1/review/review.json'
OLD_128_EVALUATION = 'experiments/p2_expanded_new_evaluation/report.json'
OLD_320_EVALUATION = 'experiments/p2_r1_resolution320_retry1/evaluation/report.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_recipe(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def recipe_differences(old, new):
    return {key: {'old': old.get(key), 'new': new.get(key)} for key in old.keys() | new.keys()
            if old.get(key) != new.get(key)}


def assert_recipe(reference, candidate, allowed, message):
    differences = recipe_differences(reference, candidate)
    unexpected = sorted(set(differences) - allowed)
    if unexpected:
        raise AssertionError(f'{message}: unexpected recipe differences {unexpected}: {differences}')
    return differences


def assert_mosaic_history(path, epochs=50, expected_p=1.0):
    observed = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if [row['epoch'] for row in observed] != list(range(1, epochs + 1)):
        raise AssertionError(f'Missing or repeated epoch observations: {path}')
    for row in observed:
        probabilities = [t['p'] for t in row['transforms'] if t['type'] == 'Mosaic']
        if not probabilities or any(p != expected_p for p in probabilities):
            raise AssertionError(f'Unexpected Mosaic state at epoch {row["epoch"]}: {probabilities}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True,
                        help='Explicit new fixed seed shared by both arms; must differ from the frozen 20260926')
    args = parser.parse_args(argv)
    if args.seed <= 0:
        parser.error('seed must be positive')
    if args.seed == FROZEN_SEED:
        parser.error(f'seed {FROZEN_SEED} is the frozen R1 seed; R1.1 requires a new fixed seed')
    return args


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed = args.seed
    report = {'status': 'running', 'node': 'R1.1 seed-repeat',
              'hypothesis': 'The R1 320-over-128 gain repeats at a second fixed seed; imgsz is the only recipe variable between arms',
              'seed': seed, 'frozen_reference_seed': FROZEN_SEED,
              'commands': [], 'stage': 'preflight', 'environment': environment(), 'source_sha256': file_hash(__file__)}

    def run(stage, arguments):
        command = [sys.executable] + arguments
        report['stage'] = stage
        entry = {'stage': stage, 'command': command, 'log': str(output / f'{stage}.log')}
        report['commands'].append(entry)
        write_json(output / 'report.json', report)
        print(f'START {stage}', flush=True)
        started = time.perf_counter()
        with (output / f'{stage}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8='1'))
        entry['returncode'] = result.returncode
        entry['seconds'] = round(time.perf_counter() - started, 1)
        if result.returncode:
            raise RuntimeError(f'{stage} failed with code {result.returncode}; inspect {entry["log"]}')
        print(f'PASS {stage} ({entry["seconds"]}s)', flush=True)

    def train_command(arm, imgsz, eval_sizes):
        return ['-m', 'training.baseline', 'train', '--data', TRAIN_DATA, '--weights', 'models/yolov8n.pt',
                '--output', str(output / f'train{arm}'), '--epochs', '50', '--batch', '8', '--imgsz', str(imgsz),
                '--eval-sizes', *eval_sizes, '--device', 'cpu', '--threads', '4', '--seed', str(seed),
                '--warmup-bias-lr', '0.1', '--close-mosaic', '0']

    try:
        # Exercise the real CSV dependency before spending training epochs.
        import polars as pl
        report['polars_preflight'] = {'version': pl.__version__,
                                      'reference_rows': pl.read_csv(BASELINE / 'fit/results.csv').height}
        if report['polars_preflight']['reference_rows'] != 50:
            raise AssertionError('Reference CSV row count changed')
        baseline_report = read(BASELINE / 'report.json')
        if baseline_report['status'] != 'complete' or report['environment']['packages'] != baseline_report['environment']['packages']:
            raise ValueError('Baseline incomplete or package environment changed')
        if file_hash('models/yolov8n.pt') != baseline_report['initial_weights_sha256']:
            raise ValueError('Initial weights changed')
        if file_hash(BASELINE / 'fit/weights/best.pt') != baseline_report['best_weights_sha256']:
            raise ValueError('Reference baseline weights changed')
        for predecessor in ('experiments/p2_r1_resolution320_retry1/report.json', OLD_128_EVALUATION, OLD_320_EVALUATION):
            if read(predecessor).get('status') != 'complete':
                raise ValueError(f'Predecessor incomplete: {predecessor}')
        for review in (OLD_128_REVIEW, OLD_320_REVIEW):
            if read(review).get('status') != 'complete':
                raise ValueError(f'Reference review incomplete: {review}')
        # Check frozen data identities before training; test files are audited, never inferred.
        for audit_path in (BASELINE / 'data_audit.json', Path('experiments/p2_expanded_new_evaluation/data_audit.json')):
            for row in read(audit_path)['files']:
                source = Path(row['image'])
                parts = list(source.parts)
                parts[len(parts)-1-parts[::-1].index('images')] = 'labels'
                label = Path(*parts).with_suffix('.txt')
                if file_hash(source) != row['image_sha256'] or file_hash(label) != row['label_sha256']:
                    raise ValueError(f'Frozen data changed: {source}')
        report['initial_weights_sha256'] = baseline_report['initial_weights_sha256']
        write_json(output / 'reference_recipe.json', load_recipe(BASELINE / 'fit/args.yaml'))
        reference_recipe = load_recipe(BASELINE / 'fit/args.yaml')

        run('training_128', train_command(128, 128, ['96', '128']))
        run('training_320', train_command(320, 320, ['320']))

        recipe_128 = load_recipe(output / 'train128/fit/args.yaml')
        recipe_320 = load_recipe(output / 'train320/fit/args.yaml')
        report['recipe_differences'] = {
            'arm128_vs_frozen_baseline': assert_recipe(reference_recipe, recipe_128, DIRECTORY_FIELDS | {'seed'}, '128 arm is not the frozen recipe'),
            'arm320_vs_arm128': assert_recipe(recipe_128, recipe_320, DIRECTORY_FIELDS | {'imgsz'}, 'Arms differ beyond imgsz'),
            'arm320_vs_frozen_baseline': assert_recipe(reference_recipe, recipe_320, DIRECTORY_FIELDS | {'imgsz', 'seed'}, '320 arm is not the frozen recipe plus imgsz/seed'),
        }
        if (recipe_128['seed'] != seed or recipe_320['seed'] != seed or recipe_128['imgsz'] != 128
                or recipe_320['imgsz'] != 320 or recipe_128['close_mosaic'] != 0 or recipe_320['close_mosaic'] != 0
                or recipe_128['warmup_bias_lr'] != .1 or recipe_320['warmup_bias_lr'] != .1):
            raise AssertionError('Paired recipe constraints violated')
        report['actual_recipe_check'] = 'passed'

        audit_128, audit_320 = read(output / 'train128/data_audit.json'), read(output / 'train320/data_audit.json')
        if audit_128 != audit_320 or audit_128 != read(BASELINE / 'data_audit.json'):
            raise AssertionError('Training audit differs between arms or from frozen baseline')
        for arm in (128, 320):
            history = list(csv.DictReader((output / f'train{arm}/fit/results.csv').open(encoding='utf-8')))
            if len(history) != 50:
                raise AssertionError(f'Unexpected training length for arm {arm}')
            assert_mosaic_history(output / f'train{arm}/augmentation_epochs.jsonl')
            report[f'first_three_epoch_lrs_{arm}'] = [{key: row[key] for key in ('epoch', 'lr/pg0', 'lr/pg1', 'lr/pg2')}
                                                      for row in history[:3]]
        report['mosaic_state_check'] = 'passed: both arms epochs 1-50 p=1'
        report['training_audit_check'] = 'passed: arms and frozen baseline identical'

        weights_128 = str(output / 'train128/fit/weights/best.pt')
        weights_320 = str(output / 'train320/fit/weights/best.pt')
        run('evaluation_128', ['-m', 'training.baseline', 'evaluate', '--data', COMMON_DATA,
                               '--weights', weights_128, '--output', str(output / 'evaluation128'),
                               '--eval-sizes', '128', '--device', 'cpu'])
        run('evaluation_320', ['-m', 'training.baseline', 'evaluate', '--data', COMMON_DATA,
                               '--weights', weights_320, '--output', str(output / 'evaluation320'),
                               '--eval-sizes', '320', '--device', 'cpu'])
        if read(output / 'evaluation128/data_audit.json') != read('experiments/p2_expanded_new_evaluation/data_audit.json') \
                or read(output / 'evaluation320/data_audit.json') != read('experiments/p2_expanded_new_evaluation/data_audit.json'):
            raise AssertionError('Common development audit changed')
        report['common_data_audit'] = 'identical to frozen common evaluation'

        for arm, imgsz in ((128, 128), (320, 320)):
            run(f'inference_{arm}', ['-m', 'runtime.vision', '--weights', str(output / f'train{arm}/fit/weights/best.pt'),
                                     '--source', 'datasets/public_expanded_comparison/images/val',
                                     '--output', str(output / f'inference{arm}'), '--imgsz', str(imgsz), '--device', 'cpu',
                                     '--conf', '0.25', '--iou', '0.45'])
            run(f'review_{arm}', ['-m', 'tools.build_failure_review', '--run', str(output / f'inference{arm}'),
                                  '--audit', str(output / f'evaluation{arm}/data_audit.json'),
                                  '--labels', COMMON_LABELS, '--output', str(output / f'review{arm}'), '--size-reference', '128'])
        run('comparison_paired', ['-m', 'tools.compare_vision_runs', '--old-review', str(output / 'review128/review.json'),
                                  '--new-review', str(output / 'review320/review.json'), '--output', str(output / 'comparison_paired.json'),
                                  '--html', '--allow-cross-size'])
        run('comparison_seed_128', ['-m', 'tools.compare_vision_runs', '--old-review', OLD_128_REVIEW,
                                    '--new-review', str(output / 'review128/review.json'), '--output', str(output / 'comparison_seed_128.json')])
        run('comparison_seed_320', ['-m', 'tools.compare_vision_runs', '--old-review', OLD_320_REVIEW,
                                    '--new-review', str(output / 'review320/review.json'), '--output', str(output / 'comparison_seed_320.json')])

        report['arm128_model_sha256'] = file_hash(weights_128)
        report['arm320_model_sha256'] = file_hash(weights_320)
        report['reference_validation_128'] = read(OLD_128_EVALUATION)['validation']['128']
        report['reference_validation_320'] = read(OLD_320_EVALUATION)['validation']['320']
        report['arm128_validation'] = read(output / 'evaluation128/report.json')['validation']['128']
        report['arm320_validation'] = read(output / 'evaluation320/report.json')['validation']['320']
        report['comparison_paired'] = read(output / 'comparison_paired.json')
        report['comparison_seed_128'] = read(output / 'comparison_seed_128.json')
        report['comparison_seed_320'] = read(output / 'comparison_seed_320.json')
        report.update(status='complete', stage='complete',
                      limitations=['Two seeds total; still insufficient for a statistical stability claim.',
                                   '109-image development validation reused across experiments; not independent test or campus acceptance.',
                                   'No model promotion, threshold change, or P3 modification performed.'])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
