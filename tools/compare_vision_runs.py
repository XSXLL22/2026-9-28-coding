"""Compare old/new models on exactly the same audited validation images."""
import argparse
import json
import html
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import file_hash, write_json


def write_comparison_page(output, report, old, new, old_directory, new_directory):
    first = {Path(r['source']).name:r for r in old['records']}
    second = {Path(r['source']).name:r for r in new['records']}
    rows = []
    for name in ('smoke','fire'):
        a,b=report['old_totals'][name],report['new_totals'][name]
        rows.append(f"<tr><td>{name}</td><td>{a['tp']}/{a['fp']}/{a['fn']}</td><td>{b['tp']}/{b['fp']}/{b['fn']}</td></tr>")
    cards=[]
    ranked=sorted(report['image_changes'], key=lambda r:(len(r['lost_targets']),r['new_fn']+r['new_fp'],len(r['recovered_targets'])),reverse=True)
    for change in ranked:
        name=change['image']
        if not (change['old_fp']+change['new_fp']+change['old_fn']+change['new_fn']):
            continue
        old_path=Path(old_directory)/first[name]['preview']
        new_path=Path(new_directory)/second[name]['preview']
        def local_link(path):
            return html.escape(Path(os.path.relpath(path,output.parent)).as_posix(),quote=True)
        cards.append(f"<details><summary>{html.escape(name)} · 补检 {len(change['recovered_targets'])} / 丢失 {len(change['lost_targets'])}"
                     f" · 新 FP {change['new_fp']} / FN {change['new_fn']}</summary>"
                     f"<p>旧模型：左真值，右预测</p><img loading='lazy' src='{local_link(old_path)}'>"
                     f"<p>新模型：左真值，右预测</p><img loading='lazy' src='{local_link(new_path)}'></details>")
    output.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P2 扩充数据对照</title>'
        '<style>body{font:16px sans-serif;max-width:1450px;margin:30px auto;padding:0 20px;background:#f3f5f7;color:#17212d}'
        'table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #bbc4cc}'
        'details{background:white;padding:16px;margin:14px 0}summary{cursor:pointer}img{width:100%}</style>'
        f"<h1>P2 扩充数据：同集逐图对照</h1><p>{report['images']} 张共同验证图，输入 {old['input_size']} → {new['input_size']}，目标分组参考 {report['size_reference']}，置信度 {old['confidence']}。"
        '这仍是开发验证，不是校园/独立测试结果。</p><table><tr><th>类别</th><th>旧 TP/FP/FN</th><th>新 TP/FP/FN</th></tr>'
        + ''.join(rows) + '</table>'
        f"<p>补检目标 {report['recovered_targets']} 个；原有检出丢失 {report['lost_targets']} 个。按丢失与剩余错误优先展开复核。</p>"
        '<p>绿色：匹配；橙色：漏检；红色：误检。图片中的原因需视觉复核，不能仅由统计自动归因。</p>'
        + ''.join(cards) + '</html>',encoding='utf-8')


def compare(old, new, *, allow_cross_size=False):
    for key in ('status','confidence','match_iou'):
        if old[key] != new[key]:
            raise ValueError(f'Comparison setting mismatch: {key}')
    if old['input_size'] != new['input_size'] and not allow_cross_size:
        raise ValueError('Comparison setting mismatch: input_size; explicit cross-size mode required')
    reference = old.get('size_reference', old['input_size'])
    if reference != new.get('size_reference', new['input_size']):
        raise ValueError('Comparison setting mismatch: size_reference')
    if allow_cross_size and ('nms_iou' not in old or 'nms_iou' not in new):
        raise ValueError('Cross-size comparison requires explicit nms_iou provenance in both reviews')
    if old.get('nms_iou') != new.get('nms_iou'):
        raise ValueError('Comparison setting mismatch: nms_iou')
    if old['status'] != 'complete':
        raise ValueError('Both reviews must be complete')
    first = {Path(r['source']).name:r for r in old['records']}
    second = {Path(r['source']).name:r for r in new['records']}
    if first.keys() != second.keys() or len(first) != len(old['records']) or len(second) != len(new['records']):
        raise ValueError('Different images or duplicate filenames')
    changes = []
    negative = {'images':0, 'old_with_predictions':0, 'new_with_predictions':0}
    for name,a in first.items():
        b = second[name]
        if file_hash(a['source']) != file_hash(b['source']) or file_hash(a['label']) != file_hash(b['label']):
            raise ValueError('Different image pixels or labels')
        def targets(row):
            return {(r['class'],r['index_in_class']):r for r in row['ground_truth']}
        ta,tb = targets(a),targets(b)
        if ta.keys() != tb.keys() or any(ta[k]['xyxy'] != tb[k]['xyxy'] for k in ta):
            raise ValueError('Ground truth mismatch')
        if any(ta[k].get('size_bucket') != tb[k].get('size_bucket') for k in ta):
            raise ValueError('Ground truth size bucket mismatch')
        recovered = [list(k) for k in ta if ta[k]['status']=='FN' and tb[k]['status']=='TP']
        lost = [list(k) for k in ta if ta[k]['status']=='TP' and tb[k]['status']=='FN']
        changes.append({'image':name, 'recovered_targets':recovered, 'lost_targets':lost,
                        'old_fp':a['fp'], 'new_fp':b['fp'], 'old_fn':a['fn'], 'new_fn':b['fn']})
        if not ta:
            negative['images'] += 1
            negative['old_with_predictions'] += bool(a['predictions'])
            negative['new_with_predictions'] += bool(b['predictions'])
    return {'status':'complete', 'old_input_size':old['input_size'], 'new_input_size':new['input_size'],
            'size_reference':reference, 'cross_size_mode':allow_cross_size, 'nms_iou':old.get('nms_iou'), 'images':len(first), 'old_model':old['model_sha256'], 'new_model':new['model_sha256'],
            'old_totals':old['totals'], 'new_totals':new['totals'], 'negative_images':negative,
            'old_size_counts':old['size_counts'], 'new_size_counts':new['size_counts'],
            'recovered_targets':sum(len(r['recovered_targets']) for r in changes),
            'lost_targets':sum(len(r['lost_targets']) for r in changes), 'image_changes':changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-review',type=Path,required=True)
    parser.add_argument('--new-review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--html',action='store_true',help='Also write an expandable local image comparison page')
    parser.add_argument('--allow-cross-size', action='store_true', help='Explicit resolution experiment; requires identical size-reference and NMS provenance')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.html and args.output.with_suffix('.html').exists():
        raise FileExistsError(args.output.with_suffix('.html'))
    old=json.loads(args.old_review.read_text(encoding='utf-8'))
    new=json.loads(args.new_review.read_text(encoding='utf-8'))
    report=compare(old,new,allow_cross_size=args.allow_cross_size)
    report.update(old_review_sha256=file_hash(args.old_review),new_review_sha256=file_hash(args.new_review))
    write_json(args.output,report)
    if args.html:
        write_comparison_page(args.output.with_suffix('.html'),report,old,new,args.old_review.parent,args.new_review.parent)
    print(json.dumps({k:v for k,v in report.items() if k not in ('image_changes','old_size_counts','new_size_counts')},ensure_ascii=False))


if __name__ == '__main__':
    main()
