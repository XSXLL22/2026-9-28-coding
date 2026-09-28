"""Pair R1-vs-reference FP boxes and emit crops for visual false-positive review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import file_hash, write_json

PAIR_IOU = 0.3
CONTEXT_FACTOR = 2.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0


def fp_boxes(review, image_name):
    for record in review['records']:
        if Path(record['source']).name == image_name:
            return [{'class': p['class'], 'xyxy': p['xyxy'], 'confidence': p['confidence']}
                    for p in record['predictions'] if p['status'] == 'FP']
    raise KeyError(image_name)


def pair_image(old_boxes, new_boxes):
    kept, added, dropped = [], [], []
    for box in new_boxes:
        match = max((iou(box['xyxy'], o['xyxy']) for o in old_boxes if o['class'] == box['class']), default=0.0)
        (kept if match >= PAIR_IOU else added).append({**box, 'best_old_iou': round(match, 4)})
    for box in old_boxes:
        match = max((iou(box['xyxy'], n['xyxy']) for n in new_boxes if n['class'] == box['class']), default=0.0)
        if match < PAIR_IOU:
            dropped.append({**box, 'best_new_iou': round(match, 4)})
    matched_old = sum(1 for box in old_boxes
                      if max((iou(box['xyxy'], n['xyxy']) for n in new_boxes if n['class'] == box['class']), default=0.0) >= PAIR_IOU)
    return kept, added, dropped, matched_old


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-review', default='experiments/p2_r1_resolution320_retry1/review/review.json')
    parser.add_argument('--old-review', default='experiments/p2_r1_resolution320_retry1/reference_review/review.json')
    parser.add_argument('--output', default='experiments/p2_r1_fp_review')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    new, old = json.loads(Path(args.new_review).read_text(encoding='utf-8')), json.loads(Path(args.old_review).read_text(encoding='utf-8'))
    for review in (new, old):
        if review['status'] != 'complete':
            raise ValueError('Reviews must be complete')
    old_by_image = {Path(r['source']).name: r for r in old['records']}
    crops = output / 'crops'
    crops.mkdir(parents=True)
    entries, dropped_all, kept_total, matched_old_total = [], [], 0, 0
    for record in new['records']:
        name = Path(record['source']).name
        kept, added, dropped, matched_old = pair_image(fp_boxes(old, name), fp_boxes(new, name))
        kept_total += len(kept)
        matched_old_total += matched_old
        dropped_all.extend({**box, 'image': name} for box in dropped)
        if not added:
            continue
        with Image.open(record['source']) as image:
            original = image.convert('RGB')
        width, height = original.size
        ground_truth = [{'class': g['class'], 'xyxy': [round(v, 1) for v in g['xyxy']], 'status': g['status']}
                        for g in record['ground_truth']]
        for index, box in enumerate(added):
            x1, y1, x2, y2 = box['xyxy']
            margin_x, margin_y = (x2-x1)*(CONTEXT_FACTOR-1)/2, (y2-y1)*(CONTEXT_FACTOR-1)/2
            crop_box = (max(0, int(x1-margin_x)), max(0, int(y1-margin_y)),
                        min(width, int(x2+margin_x)), min(height, int(y2+margin_y)))
            crop = original.crop(crop_box)
            if crop.width < 320 and crop.width < width:
                crop = crop.resize((320, max(1, int(crop.height*320/crop.width))))
            crop_path = f'{name[:-4]}_fp{index}_{box["class"]}.jpg'
            crop.save(crops / crop_path, quality=90)
            entries.append({'id': f'{name[:-4]}_fp{index}', 'image': name, 'class': box['class'],
                            'confidence': round(box['confidence'], 3), 'xyxy': [round(v, 1) for v in box['xyxy']],
                            'best_old_iou': box['best_old_iou'], 'crop': f'crops/{crop_path}',
                            'image_has_targets': bool(ground_truth),
                            'nearby_ground_truth': ground_truth, 'classification': 'pending_review'})
    result = {'status': 'complete', 'pair_rule': f'same-class IoU>={PAIR_IOU} against old-model FP boxes',
              'new_review_sha256': file_hash(args.new_review), 'old_review_sha256': file_hash(args.old_review),
              'new_model_sha256': new['model_sha256'], 'old_model_sha256': old['model_sha256'],
              'totals': {'new_fp': sum(1 for r in new['records'] for p in r['predictions'] if p['status'] == 'FP'),
                         'old_fp': sum(1 for r in old['records'] for p in r['predictions'] if p['status'] == 'FP'),
                         'kept': kept_total, 'matched_old': matched_old_total,
                         'added': len(entries), 'dropped': len(dropped_all)},
              'added_fp': entries, 'dropped_fp': dropped_all}
    write_json(output / 'fp_pairs.json', result)
    print(json.dumps(result['totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
