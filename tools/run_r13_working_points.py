"""R1.3 pre-declared working-point sweep on a frozen candidate; every scheme re-runs inference."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from runtime.common import environment, file_hash, write_json
from training.error_analysis import analyze

COMMON_DATA_IMAGES = 'datasets/public_expanded_comparison/images/val'
COMMON_LABELS = 'datasets/public_expanded_comparison/labels/val'
# Declared in experiments/P2_R1.3工作点比较计划.md before any sweep result existed.
SCHEMES = [('S0_conf0.25_nms0.45', 0.25, 0.45),
           ('S1_conf0.35_nms0.45', 0.35, 0.45),
           ('S2_conf0.45_nms0.45', 0.45, 0.45),
           ('S3_conf0.25_nms0.30', 0.25, 0.30),
           ('S4_conf0.35_nms0.30', 0.35, 0.30)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--imgsz', type=int, choices=(128, 320), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'schemes': [{'name': n, 'conf': c, 'nms_iou': i} for n, c, i in SCHEMES],
              'weights_sha256': file_hash(args.weights), 'imgsz': args.imgsz, 'labels': COMMON_LABELS,
              'environment': environment(), 'source_sha256': file_hash(__file__)}
    write_json(output / 'report.json', report)
    try:
        rows = []
        for name, conf, nms in SCHEMES:
            started = time.perf_counter()
            inference_dir = output / f'inference_{name}'
            subprocess.run([sys.executable, '-m', 'runtime.vision', '--weights', args.weights,
                            '--source', COMMON_DATA_IMAGES, '--output', str(inference_dir),
                            '--imgsz', str(args.imgsz), '--device', args.device,
                            '--conf', str(conf), '--iou', str(nms)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUTF8='1'))
            metrics = analyze(inference_dir / 'detections.jsonl', COMMON_LABELS)
            run = json.loads((inference_dir / 'run.json').read_text(encoding='utf-8'))
            if run['args']['conf'] != conf or run['args']['iou'] != nms:
                raise AssertionError(f'{name}: inference working point does not match the declared scheme')
            row = {'scheme': name, 'conf': conf, 'nms_iou': nms,
                   'inference_dir': inference_dir.name, 'run_sha256': file_hash(inference_dir / 'run.json'),
                   'model_sha256': run['model_sha256'], 'per_class': metrics['per_class'],
                   'negative_images': metrics['negative_images'],
                   'negative_images_with_detection': metrics['negative_images_with_detection'],
                   'seconds': round(time.perf_counter() - started, 1)}
            if run['model_sha256'] != report['weights_sha256']:
                raise AssertionError(f'{name}: unexpected model weights')
            rows.append(row)
            report['results'] = rows
            write_json(output / 'report.json', report)
            print(f"PASS {name}: smoke tp/fp/fn {row['per_class']['smoke']['tp']}/{row['per_class']['smoke']['fp']}/{row['per_class']['smoke']['fn']}"
                  f" fire {row['per_class']['fire']['tp']}/{row['per_class']['fire']['fp']}/{row['per_class']['fire']['fn']}"
                  f" neg-img {row['negative_images_with_detection']}/{row['negative_images']}", flush=True)
        report.update(status='complete',
                      note='Fixed working-point box metrics on the reused 109-image development set; not event alarm rates and not the untouched test split.')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
