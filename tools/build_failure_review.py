"""Build a local visual error review from existing predictions; never run inference."""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import CLASSES, file_hash, write_json
from training.error_analysis import analyze, match_details


def normalized_input_hashes(hashes):
    """Inference sources may be project-relative or absolute; preserve hash checks."""
    return {str((ROOT / path).resolve()): digest for path, digest in hashes.items()}


def main():
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Completed still-image inference directory')
    parser.add_argument('--audit', required=True, help='Training data_audit.json with label hashes')
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size-reference', type=int, choices=(96, 128, 256, 320), help='Fixed scale for target-size bins; defaults to inference size')
    args = parser.parse_args()
    run_dir, output = Path(args.run), Path(args.output)
    run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    if run['status'] != 'complete':
        raise ValueError('Inference run is not complete')
    size_reference = args.size_reference or run['args']['imgsz']
    input_hashes = normalized_input_hashes(run['input_hashes'])
    audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    audited = {str(Path(row['image']).resolve()): row for row in audit['files']}
    predictions_path = run_dir / 'detections.jsonl'
    records = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines()]
    aggregate = analyze(predictions_path, args.labels)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'run_sha256': file_hash(run_dir / 'run.json'),
              'predictions_sha256': file_hash(predictions_path), 'audit_sha256': file_hash(args.audit),
              'model_sha256': run['model_sha256'], 'input_size': run['args']['imgsz'],
              'confidence': run['args']['conf'], 'match_iou': 0.5,
              'nms_iou': run['args']['iou'], 'size_reference': size_reference,
              'size_note': 'Approximate letterbox target short side: original pixels * size_reference / max(image dimensions). Diagnostic bins, not COCO size categories.',
              'manual_causes': 'Not assessed: occlusion, lighting, background and annotation quality require review.'}
    try:
        rows, cards = [], []
        size_counts = {name: {bucket: Counter(gt=0, tp=0, fn=0) for bucket in ('lt8', '8to16', 'ge16')} for name in CLASSES}
        totals = {name: Counter(tp=0, fp=0, fn=0) for name in CLASSES}
        for record in records:
            source = Path(record['source']).resolve()
            label = Path(args.labels) / (source.stem + '.txt')
            expected = audited[str(source)]
            image_hash = file_hash(source)
            if image_hash != expected['image_sha256'] or image_hash != input_hashes[str(source)]:
                raise ValueError(f'Image changed since inference/audit: {source}')
            if file_hash(label) != expected['label_sha256']:
                raise ValueError(f'Label changed since training audit: {label}')
            if record['model_sha256'] != run['model_sha256']:
                raise ValueError('Mixed model versions')
            with Image.open(source) as image:
                original = image.convert('RGB')
            width, height = original.size
            if (width, height) != (record['image_width'], record['image_height']):
                raise ValueError('Image dimensions do not match prediction record')
            targets = {name: [] for name in CLASSES}
            names = list(CLASSES)
            for line in label.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    c, x, y, w, h = map(float, line.split())
                    targets[names[int(c)]].append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
            row = {'source': str(source), 'label': str(label), 'frame_id': record['frame_id'],
                   'per_class': {}, 'ground_truth': [], 'predictions': [], 'manual_cause': 'pending_review'}
            panels = []
            for name, class_id in CLASSES.items():
                preds = [p for p in record['detections'] if p['project_class_id'] == class_id]
                details = match_details(preds, targets[name], 0.5)
                counts = {'tp': len(details['pairs']), 'fp': len(details['fp_indices']), 'fn': len(details['fn_indices'])}
                row['per_class'][name] = counts
                totals[name].update(counts)
                missed = set(details['fn_indices'])
                for index, box in enumerate(targets[name]):
                    short_side = min(box[2]-box[0], box[3]-box[1]) * size_reference / max(width, height)
                    bucket = 'lt8' if short_side < 8 else '8to16' if short_side < 16 else 'ge16'
                    status = 'FN' if index in missed else 'TP'
                    size_counts[name][bucket].update(gt=1, **{status.lower(): 1})
                    row['ground_truth'].append({'class': name, 'index_in_class': index, 'xyxy': box,
                                                'status': status, 'approx_input_short_side_px': short_side, 'size_bucket': bucket})
                pairs = {p['prediction_index']: p for p in details['pairs']}
                for index, prediction in enumerate(preds):
                    pair = pairs.get(index)
                    row['predictions'].append({'class': name, 'xyxy': prediction['xyxy_original_pixels'],
                                              'confidence': prediction['confidence'], 'status': 'TP' if pair else 'FP',
                                              'matched_target_index_in_class': pair['target_index'] if pair else None,
                                              'matched_iou': pair['iou'] if pair else None})
            for title, boxes in [('GROUND TRUTH: green=matched orange=missed', row['ground_truth']),
                                 ('PREDICTIONS: green=correct red=false positive', row['predictions'])]:
                panel = original.copy()
                panel.thumbnail((760, 570))
                scale_x, scale_y = panel.width / width, panel.height / height
                canvas = Image.new('RGB', (760, 610), 'white')
                canvas.paste(panel, (0, 34))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 8), title, fill='black')
                for box in boxes:
                    x1,y1,x2,y2 = box['xyxy']
                    coords = (x1*scale_x, y1*scale_y+34, x2*scale_x, y2*scale_y+34)
                    color = {'TP':'#008040', 'FN':'#df7800', 'FP':'#e00030'}[box['status']]
                    draw.rectangle(coords, outline=color, width=2)
                    caption = f"{box['class']} {box['status']}"
                    if 'confidence' in box:
                        caption += f" {box['confidence']:.2f}"
                    draw.text((coords[0], max(34, coords[1]-12)), caption, fill=color, stroke_width=1, stroke_fill='white')
                panels.append(canvas)
            combined = Image.new('RGB', (1520,610), 'white')
            combined.paste(panels[0], (0,0)); combined.paste(panels[1], (760,0))
            row['preview'] = f"frame_{record['frame_id']:06d}.jpg"
            combined.save(output / row['preview'], quality=90)
            row['fp'] = sum(c['fp'] for c in row['per_class'].values())
            row['fn'] = sum(c['fn'] for c in row['per_class'].values())
            row['has_error'] = bool(row['fp'] or row['fn'])
            rows.append(row)
        for name in CLASSES:
            if any(totals[name][key] != aggregate['per_class'][name][key] for key in ('tp','fp','fn')):
                raise AssertionError('Detailed review differs from aggregate analysis')
        rows.sort(key=lambda row: (row['has_error'], row['fn'] + row['fp']), reverse=True)
        for row in rows:
            cards.append(f"<section><h2>{html.escape(Path(row['source']).name)} · FN {row['fn']} / FP {row['fp']}</h2>"
                         f"<a href='{row['preview']}'><img src='{row['preview']}' alt='标注与预测对照'></a></section>")
        report.update(status='complete', images=len(rows), error_images=sum(r['has_error'] for r in rows),
                      totals=totals, size_counts=size_counts, records=rows, aggregate_crosscheck='passed')
        (output / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P2 失败样本复核</title>'
            '<style>body{font:16px sans-serif;max-width:1540px;margin:32px auto;padding:0 20px;background:#f4f5f7;color:#18212b}'
            'section{background:white;padding:16px;margin:24px 0}img{width:100%}h2{font-size:18px}</style>'
            f"<p>小目标统计参考尺度：{size_reference}。</p><h1>P2 标注与预测对照 · {run['args']['imgsz']} 输入</h1><p>复用既有预测，未重新训练或推理。失败样本优先。"
            '左：真值（橙色漏检、绿色匹配）；右：预测（红色误检、绿色正确）。点击图片查看大图。</p>'
            f"<p>{len(rows)} 张图，其中 {report['error_images']} 张存在误检或漏检。匹配 IoU=0.5，置信度={run['args']['conf']}。"
            '遮挡、光照和背景干扰原因尚待人工复核，不能仅由框匹配判断。统计详见 review.json。</p>'
            + ''.join(cards) + '</html>', encoding='utf-8')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'review.json', report)
    print(json.dumps({'status': report['status'], 'images': report['images'], 'error_images': report['error_images'],
                      'totals': totals, 'size_counts': size_counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
