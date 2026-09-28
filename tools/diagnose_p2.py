"""Read-only model diagnosis: train augmentations and held-out development candidates.

No optimizer, backward pass, training or test-set inference is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from runtime.common import ROOT, configure_ultralytics, environment, file_hash, write_json
from training.error_analysis import iou, match_details

NAMES = ('smoke', 'fire')
GRID = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)


def bucket(short):
    return 'lt8' if short < 8 else '8to16' if short < 16 else 'ge16'


def score_records(records, threshold):
    totals = {name: Counter(tp=0, fp=0, fn=0) for name in NAMES}
    sizes = {name: {key: Counter(gt=0, tp=0) for key in ('lt8', '8to16', 'ge16')} for name in NAMES}
    misses, false_positives, negative_count = [], [], 0
    for record in records:
        selected = [p for p in record['predictions'] if p['confidence'] >= threshold]
        if not record['targets'] and selected:
            negative_count += 1
        for c, name in enumerate(NAMES):
            targets = [t for t in record['targets'] if t['class_id'] == c]
            preds = [p for p in selected if p['class_id'] == c]
            match = match_details(preds, [t['xyxy'] for t in targets], 0.5)
            paired = {p['target_index'] for p in match['pairs']}
            totals[name].update(tp=len(paired), fp=len(match['fp_indices']), fn=len(match['fn_indices']))
            for index, target in enumerate(targets):
                sizes[name][target['size_bucket']].update(gt=1, tp=int(index in paired))
                if index not in paired:
                    low = [p for p in record['predictions'] if p['class_id'] == c]
                    best_iou = max((iou(p['xyxy_original_pixels'], target['xyxy']) for p in low), default=0)
                    # Mutually exclusive diagnostic labels, not human-verified causes.
                    if any(iou(p['xyxy_original_pixels'], target['xyxy']) >= .5 and p['confidence'] >= threshold for p in low):
                        reason = 'matching_competition'
                    elif best_iou >= .5:
                        reason = 'low_confidence_candidate'
                    elif any(p['class_id'] != c and iou(p['xyxy_original_pixels'], target['xyxy']) >= .5 for p in selected):
                        reason = 'other_class_overlap'
                    elif best_iou >= .1:
                        reason = 'localization_overlap'
                    else:
                        reason = 'no_overlap_candidate_at_floor'
                    misses.append({'source': record['source'], 'target_id': target['target_id'],
                                   'class': name, 'reason': reason, 'best_same_class_iou': best_iou,
                                   'size_bucket': target['size_bucket']})
            for index in match['fp_indices']:
                pred = preds[index]
                same = max((iou(pred['xyxy_original_pixels'], t['xyxy']) for t in targets), default=0)
                other = max((iou(pred['xyxy_original_pixels'], t['xyxy']) for t in record['targets'] if t['class_id'] != c), default=0)
                reason = ('duplicate_or_matching_competition' if same >= .5 else
                          'other_class_overlap' if other >= .5 else
                          'localization_overlap' if same >= .1 else 'background_or_unannotated')
                false_positives.append({'source': record['source'], 'class': name, 'reason': reason,
                                        'confidence': pred['confidence'], 'best_same_class_iou': same})
    for values in totals.values():
        tp, fp, fn = (values[k] for k in ('tp', 'fp', 'fn'))
        values['precision'] = tp / (tp + fp) if tp + fp else None
        values['recall'] = tp / (tp + fn) if tp + fn else None
    return {'threshold': threshold, 'per_class': totals, 'size_counts_reference_128': sizes,
            'negative_images_with_detection': negative_count,
            'miss_reasons': dict(Counter(m['reason'] for m in misses)), 'misses': misses,
            'fp_reasons': dict(Counter(p['reason'] for p in false_positives)), 'false_positives': false_positives}


def augment_audit(args, output):
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.augment import RandomPerspective

    stored = yaml.safe_load(Path(args.recipe).read_text(encoding='utf-8'))
    data = yaml.safe_load(Path(args.train_data).read_text(encoding='utf-8'))
    base = Path(data['path'])
    train = base / data['train']
    rows, variants, previews = [], {}, []
    for variant in ('current', 'closed_mosaic', 'no_mosaic'):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cfg = get_cfg()
        for key, value in stored.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.cache = False
        if variant == 'no_mosaic':
            cfg.mosaic = 0.0
        dataset = build_yolo_dataset(cfg, str(train), stored['batch'], data, mode='train')
        if variant == 'closed_mosaic':
            dataset.close_mosaic(cfg)
        order = list(range(len(dataset)))
        random.Random(args.seed).shuffle(order)
        # Isolate diagnostic sampling from dataset initialization RNG consumption.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        context = {}
        counts = {name: Counter(enter_filter=0, kept=0, rejected=0) for name in NAMES}
        final_sizes = {name: Counter() for name in NAMES}
        original_call = RandomPerspective.__call__
        original_candidates = RandomPerspective.box_candidates
        original_get = dataset.get_image_and_label

        def get_source(index):
            item = original_get(index)
            context.setdefault('sources', []).append(item['im_file'])
            return item

        def call(transform, labels):
            context['classes'] = labels['cls'].reshape(-1).copy()
            return original_call(transform, labels)

        def candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=.1, eps=1e-16):
            keep = original_candidates(box1, box2, wh_thr, ar_thr, area_thr, eps)
            if len(keep) != len(context['classes']):
                raise AssertionError('Class/filter association changed in installed library')
            for index, flag in enumerate(keep):
                w1, h1 = box1[2:, index] - box1[:2, index]
                w2, h2 = box2[2:, index] - box2[:2, index]
                reasons = []
                if not (w2 > wh_thr and h2 > wh_thr): reasons.append('width_or_height')
                if not w2 * h2 / (w1 * h1 + eps) > area_thr: reasons.append('retained_area')
                if not max(w2 / (h2 + eps), h2 / (w2 + eps)) < ar_thr: reasons.append('aspect_ratio')
                name = NAMES[int(context['classes'][index])]
                counts[name].update(enter_filter=1, kept=int(flag), rejected=int(not flag))
                counts[name].update(reasons)
                rows.append({'variant': variant, 'sample_index': context['sample_index'], 'class': name,
                             'sources': list(context['sources']), 'box_index_at_filter': index,
                             'pre_filter_scaled_box': box1[:, index].tolist(),
                             'transformed_clipped_box': box2[:, index].tolist(),
                             'kept': bool(flag), 'rejection_reasons': reasons})
            return keep

        with patch.object(dataset, 'get_image_and_label', side_effect=get_source), \
             patch.object(RandomPerspective, '__call__', call), \
             patch.object(RandomPerspective, 'box_candidates', staticmethod(candidates)):
            for position, index in enumerate(order[:args.augmentation_samples]):
                context.update(sample_index=index, sources=[])
                item = dataset[index]
                array = item['img'].permute(1, 2, 0).numpy()
                image = Image.fromarray(array)
                draw = ImageDraw.Draw(image)
                for c, box in zip(item['cls'].reshape(-1).tolist(), item['bboxes'].tolist()):
                    x, y, w, h = box
                    final_sizes[NAMES[int(c)]].update([bucket(min(w * image.width, h * image.height))])
                    draw.rectangle(((x-w/2)*image.width, (y-h/2)*image.height,
                                    (x+w/2)*image.width, (y+h/2)*image.height), outline='lime', width=1)
                if position < 8:
                    name = f'{variant}_{position:02d}.png'
                    image.resize((384, 384), Image.Resampling.NEAREST).save(output / name)
                    previews.append({'file': name, 'variant': variant, 'sources': list(context['sources']),
                                     'primary_source': item['im_file']})
        variants[variant] = {'samples': min(args.augmentation_samples, len(dataset)), 'filter': counts,
                             'final_box_sizes': final_sizes,
                             'hyp': {k: getattr(cfg, k) for k in ('mosaic', 'mixup', 'cutmix', 'copy_paste', 'scale')}}
    write_json(output / 'augmentation_candidates.json', rows)
    write_json(output / 'augmentation_previews.json', previews)
    return {'variants': variants, 'seed': args.seed,
            'scope': 'Actual train dataset transforms, randomized diagnostic order; not a replay of training batches. '
                     'Counts begin at RandomPerspective filter, after earlier Mosaic clipping. Source image list is '
                     'tracked, individual source GT identity is NOT tracked across Mosaic. Reasons may overlap. '
                     'No claim of end-to-end GT survival or causal accuracy gain.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', default='experiments/p2_expanded_train/fit/weights/best.pt')
    parser.add_argument('--recipe', default='experiments/p2_expanded_train/fit/args.yaml')
    parser.add_argument('--train-data', default='datasets/public_expanded_v1_prepared/data.yaml')
    parser.add_argument('--audit', default='experiments/p2_expanded_new_evaluation/data_audit.json')
    parser.add_argument('--augmentation-samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    if args.augmentation_samples <= 0:
        parser.error('augmentation-samples must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'args': vars(args), 'training_performed': False}
    try:
        import torch
        configure_ultralytics()
        from ultralytics import YOLO
        from runtime.vision import read_image
        import yaml
        torch.set_num_threads(4)
        report['environment'] = environment()
        if report['environment']['packages']['ultralytics'] != '8.3.228':
            raise ValueError('Diagnostic hooks require audited Ultralytics 8.3.228')
        report['model_sha256'] = file_hash(args.weights)
        report['recipe_sha256'] = file_hash(args.recipe)
        report['audit_sha256'] = file_hash(args.audit)
        audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
        training_audit = json.loads((Path(args.recipe).parents[1] / 'data_audit.json').read_text(encoding='utf-8'))
        for entry in training_audit['files']:
            if entry['split'] != 'train': continue
            source = Path(entry['image'])
            label = source.parent.parent.parent / 'labels' / 'train' / (source.stem + '.txt')
            if file_hash(source) != entry['image_sha256'] or file_hash(label) != entry['label_sha256']:
                raise ValueError(f'Training data changed: {source}')
        report['augmentation'] = augment_audit(args, output)
        model = YOLO(args.weights)
        if model.names != {0: 'smoke', 1: 'fire'}:
            raise ValueError('Expected smoke/fire model')
        records = []
        for entry in audit['files']:
            if entry['split'] != 'val': continue
            source = Path(entry['image'])
            label = source.parent.parent.parent / 'labels' / 'val' / (source.stem + '.txt')
            if file_hash(source) != entry['image_sha256'] or file_hash(label) != entry['label_sha256']:
                raise ValueError(f'Development data changed: {source}')
            frame = read_image(source)
            height, width = frame.shape[:2]
            targets = []
            for index, line in enumerate(label.read_text(encoding='utf-8').splitlines()):
                if not line.strip(): continue
                c, x, y, w, h = map(float, line.split())
                targets.append({'class_id': int(c), 'target_id': index,
                                'xyxy': [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height],
                                'size_bucket': bucket(min(w*width, h*height)*128/max(width, height))})
            result = model.predict(frame, imgsz=128, conf=.001, iou=.45, max_det=3000,
                                   device='cpu', rect=False, verbose=False)[0]
            if len(result.boxes) >= 3000:
                raise RuntimeError('Candidate cap reached; diagnostic would be censored')
            predictions = [{'class_id': int(c), 'confidence': score, 'xyxy_original_pixels': box}
                           for box, score, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())]
            records.append({'source': str(source), 'image_sha256': entry['image_sha256'],
                            'label_sha256': entry['label_sha256'], 'targets': targets, 'predictions': predictions})
        if len(records) != 109:
            raise ValueError('Expected frozen 109-image development set')
        write_json(output / 'candidates.json', records)
        report['sweep'] = [score_records(records, t) for t in GRID]
        baseline = next(s for s in report['sweep'] if s['threshold'] == .25)
        if [(v['tp'], v['fp'], v['fn']) for v in baseline['per_class'].values()] != [(36, 16, 36), (14, 14, 76)]:
            raise AssertionError('0.25 results differ from frozen baseline; investigate before interpreting sweep')
        report['baseline_crosscheck'] = 'passed'
        recipe = yaml.safe_load(Path(args.recipe).read_text(encoding='utf-8'))
        history = list(csv.DictReader((Path(args.recipe).parent / 'results.csv').open(encoding='utf-8')))
        report['learning_rate'] = {'recipe': {k: recipe[k] for k in ('optimizer', 'lr0', 'warmup_bias_lr', 'warmup_epochs', 'nbs', 'batch')},
                                   'first_four_epochs': [{k: row[k] for k in ('epoch', 'lr/pg0', 'lr/pg1', 'lr/pg2')} for row in history[:4]],
                                   'note': 'Existing logs only. Bias warmup=0.0 remains an untested training hypothesis.'}
        report.update(status='complete', images=len(records), negative_images=sum(not r['targets'] for r in records),
                      inference={'confidence_floor': .001, 'nms_iou': .45, 'max_det': 3000, 'imgsz': 128},
                      limitations=['Post-NMS candidates only; no candidate means none at this floor and overlap rule.',
                                   'Development set, not independent test. Automatic overlap reasons are not visual ground truth.',
                                   'No epoch simulation and no causal augmentation ablation performed.'])
        if file_hash(args.weights) != report['model_sha256']:
            raise AssertionError('Weights changed during diagnosis')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'images': len(records), 'sweep': [
        {k: s[k] for k in ('threshold', 'per_class', 'negative_images_with_detection', 'miss_reasons')} for s in report['sweep']]}))


if __name__ == '__main__':
    main()
