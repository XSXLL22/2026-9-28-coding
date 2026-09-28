"""Verify completed R1.1 paired-seed evidence and write the milestone brief."""
import argparse
import csv
import json
from pathlib import Path

from runtime.common import configure_ultralytics, file_hash, write_json
from tools.inspect_a2_selection import infer_selection

FROZEN_SEED = 20260926
R1_RUN = Path('experiments/p2_r1_resolution320_retry1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arm_selection(run, arm):
    configure_ultralytics()
    import torch
    torch.set_num_threads(1)
    weights = run / f'train{arm}/fit/weights/best.pt'
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    with (run / f'train{arm}/fit/results.csv').open(encoding='utf-8') as stream:
        return infer_selection(checkpoint.get('train_metrics', {}), list(csv.DictReader(stream)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='experiments/p2_r11_seed_repeat')
    parser.add_argument('--interrupted-run', default=None,
                        help='Optional earlier failed run of the same seed; first saved epochs must match except elapsed time')
    parser.add_argument('--output', default='experiments/P2_R1.1种子重复简报.md')
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    if output.exists() or (root / 'verification.json').exists():
        raise FileExistsError('Report or verification already exists')
    report = read(root / 'report.json')
    if report['status'] != 'complete' or report['actual_recipe_check'] != 'passed' \
            or not str(report['training_audit_check']).startswith('passed') \
            or report['common_data_audit'] != 'identical to frozen common evaluation':
        raise ValueError('R1.1 must finish successfully with all checks passed')
    if report['seed'] == FROZEN_SEED:
        raise ValueError('R1.1 was run with the frozen seed; a new seed is required')
    # The comparisons must reference exactly the reviews produced inside this run,
    # plus the frozen-seed reference reviews they were built from.
    for name, review_path in (('comparison_seed_128', Path('experiments/p2_r1_resolution320_retry1/reference_review/review.json')),
                              ('comparison_seed_320', R1_RUN / 'review/review.json')):
        if report[name]['old_review_sha256'] != file_hash(review_path):
            raise AssertionError(f'{name} references a changed review: {review_path}')
    if report['comparison_paired']['old_review_sha256'] != file_hash(root / 'review128/review.json') \
            or report['comparison_paired']['new_review_sha256'] != file_hash(root / 'review320/review.json') \
            or report['comparison_seed_128']['new_review_sha256'] != file_hash(root / 'review128/review.json') \
            or report['comparison_seed_320']['new_review_sha256'] != file_hash(root / 'review320/review.json'):
        raise AssertionError('Comparison review hashes do not match run reviews')
    for arm in (128, 320):
        if file_hash(root / f'train{arm}/fit/weights/best.pt') != report[f'arm{arm}_model_sha256']:
            raise AssertionError(f'Arm {arm} weights changed')
        if file_hash('training/baseline.py') != read(root / f'train{arm}/report.json')['entrypoint_sha256']:
            raise AssertionError(f'Training source changed for arm {arm}')
    selection_128 = arm_selection(root, 128)
    selection_320 = arm_selection(root, 320)
    restart_check = 'not requested'
    if args.interrupted_run:
        interrupted = Path(args.interrupted_run)
        if read(interrupted / 'report.json').get('status') != 'failed':
            raise ValueError('Interrupted run must have failed; completed runs are not restart references')
        for arm in (128, 320):
            with (interrupted / f'train{arm}/fit/results.csv').open(encoding='utf-8') as stream:
                first_interrupted = next(csv.DictReader(stream))
            with (root / f'train{arm}/fit/results.csv').open(encoding='utf-8') as stream:
                first_restarted = next(csv.DictReader(stream))
            if any(first_interrupted[k] != first_restarted[k] for k in first_interrupted if k != 'time'):
                raise AssertionError(f'Arm {arm} first epoch differs from interrupted run')
        restart_check = 'both arms first epoch identical to interrupted run except elapsed time'
    verification = {'status': 'complete', 'seed': report['seed'],
                    'arm128_selection': selection_128, 'arm320_selection': selection_320,
                    'arm128_sha256': report['arm128_model_sha256'], 'arm320_sha256': report['arm320_model_sha256'],
                    'review_hash_checks': 'passed', 'common_data_audit': 'identical',
                    'restart_first_epoch': restart_check}
    write_json(root / 'verification.json', verification)

    seed = report['seed']
    ref128, ref320 = report['reference_validation_128'], report['reference_validation_320']
    arm128, arm320 = report['arm128_validation'], report['arm320_validation']
    paired, seed128, seed320 = report['comparison_paired'], report['comparison_seed_128'], report['comparison_seed_320']
    r1 = read(R1_RUN / 'report.json')['comparison']

    def metric_rows():
        rows = []
        for seed_label, old, new in ((str(FROZEN_SEED), ref128, ref320), (str(seed), arm128, arm320)):
            for name, key in (('mAP50', 'metrics/mAP50(B)'), ('mAP50:95', 'metrics/mAP50-95(B)')):
                rows.append(f'| {seed_label} | {name} | {old["aggregate"][key]:.2%} | {new["aggregate"][key]:.2%} | {(new["aggregate"][key]-old["aggregate"][key])*100:+.2f} |')
        return rows

    def class_rows(comparison):
        return [f"| {c} | {'/'.join(str(comparison['old_totals'][c][k]) for k in ('tp','fp','fn'))} | "
                f"{'/'.join(str(comparison['new_totals'][c][k]) for k in ('tp','fp','fn'))} |" for c in ('smoke', 'fire')]

    def size_rows(old_counts, new_counts):
        rows = []
        for c in ('smoke', 'fire'):
            for bucket in ('lt8', '8to16', 'ge16'):
                x, y = old_counts[c][bucket], new_counts[c][bucket]
                if x['gt'] != y['gt']:
                    raise AssertionError('Target group population changed')
                rows.append(f"| {c} | {bucket} | {x['gt']} | {x['tp']} | {y['tp']} |")
        return rows

    import os
    link = Path(os.path.relpath(root, output.parent)).as_posix()
    neg_paired, neg_r1 = paired['negative_images'], r1['negative_images']
    text = f'''# P2 R1.1 收益重复验证简报

2026-09-28。按[训练前计划](P2_R1.1种子重复实验计划.md)完成第二个固定种子 seed={seed} 下的 128 与 320 配对训练、共同开发验证、固定工作点推理及三组逐图对照。冻结 R1 工具未修改；R1.1 使用新入口 `tools.run_r11_paired_experiment`，种子显式给出。首次运行在训练后因入口命令构造错误中断，原目录保留，retry1 完整重跑且两臂首轮已保存数值与失败运行一致（排除耗时）。

## 实验控制与核验

两臂同种子、同官方 YOLOv8n 初始化、同 394/114 数据、50 轮、batch=8、CPU 4 线程、AdamW、lr0=0.001、warmup_bias_lr=0.1、close_mosaic=0。实际 `fit/args.yaml` 逐项核对：128 臂相对冻结基线仅差输出目录与 seed；320 臂相对 128 臂仅差 imgsz。两臂数据审核与冻结基线完全相同，50 轮增强观察确认 Mosaic 概率均为 1。

- 新种子 128 臂权重 SHA256：`{report['arm128_model_sha256']}`，best 对应轮次：{selection_128['epoch']}（指标匹配推断）。
- 新种子 320 臂权重 SHA256：`{report['arm320_model_sha256']}`，best 对应轮次：{selection_320['epoch']}（指标匹配推断）。

## 两个种子的配对结果（共同 109 张开发图）

| 种子 | 指标 | 128 | 320 | 差（320−128，百分点） |
|---|---|---:|---:|---:|
{chr(10).join(metric_rows())}

AP 使用库内 conf=0.001、NMS=0.7；114 张训练验证图用于选 best。R1 的单种子结论（320 提高 mAP50 与小目标检出）是否重复，以本表为准。

## 固定工作点（conf=0.25、NMS IoU=0.45、匹配 IoU=0.5）

| 种子 | smoke TP/FP/FN（128 → 320） | fire TP/FP/FN（128 → 320） |
|---|---|---|
| {FROZEN_SEED}（R1 对照） | {r1['old_totals']['smoke']['tp']}/{r1['old_totals']['smoke']['fp']}/{r1['old_totals']['smoke']['fn']} → {r1['new_totals']['smoke']['tp']}/{r1['new_totals']['smoke']['fp']}/{r1['new_totals']['smoke']['fn']} | {r1['old_totals']['fire']['tp']}/{r1['old_totals']['fire']['fp']}/{r1['old_totals']['fire']['fn']} → {r1['new_totals']['fire']['tp']}/{r1['new_totals']['fire']['fp']}/{r1['new_totals']['fire']['fn']} |
| {seed}（本轮） | {'/'.join(str(paired['old_totals']['smoke'][k]) for k in ('tp','fp','fn'))} → {'/'.join(str(paired['new_totals']['smoke'][k]) for k in ('tp','fp','fn'))} | {'/'.join(str(paired['old_totals']['fire'][k]) for k in ('tp','fp','fn'))} → {'/'.join(str(paired['new_totals']['fire'][k]) for k in ('tp','fp','fn'))} |

本轮配对补检 {paired['recovered_targets']} 个、丢失 {paired['lost_targets']} 个；负样本 {neg_paired['images']} 张中有预测的图片 {neg_paired['old_with_predictions']} → {neg_paired['new_with_predictions']}（旧种子为 {neg_r1['old_with_predictions']} → {neg_r1['new_with_predictions']}）。

## 小目标（分组固定 128 参考尺度，不随输入变化）

| 类别 | 短边组 | GT | 本轮 128 检出 | 本轮 320 检出 |
|---|---|---:|---:|---:|
{chr(10).join(size_rows(paired['old_size_counts'], paired['new_size_counts']))}

lt8、8to16、ge16 对应缩放短边 <8、8～<16、≥16 像素。

## 同尺寸种子波动

- 128：旧种子 vs 新种子 {seed128['old_totals']['smoke']['tp']+seed128['old_totals']['fire']['tp']} → {seed128['new_totals']['smoke']['tp']+seed128['new_totals']['fire']['tp']} TP，{seed128['old_totals']['smoke']['fp']+seed128['old_totals']['fire']['fp']} → {seed128['new_totals']['smoke']['fp']+seed128['new_totals']['fire']['fp']} FP；补检 {seed128['recovered_targets']}、丢失 {seed128['lost_targets']}。
- 320：旧种子 vs 新种子 {seed320['old_totals']['smoke']['tp']+seed320['old_totals']['fire']['tp']} → {seed320['new_totals']['smoke']['tp']+seed320['new_totals']['fire']['tp']} TP，{seed320['old_totals']['smoke']['fp']+seed320['old_totals']['fire']['fp']} → {seed320['new_totals']['smoke']['fp']+seed320['new_totals']['fire']['fp']} FP；补检 {seed320['recovered_targets']}、丢失 {seed320['lost_targets']}。

同尺寸的种子波动是解读 320 收益时的本底噪声，不能把小于该波动幅度的差异当作稳定收益。

## 结论与限制

- 两个种子的配对方向是否一致见上表；两个种子仍不能宣称统计稳定性，也不据此自动推广 320 为默认。
- 后续仍按 [R1 交接](../docs/P2_R1交接.md)：R1.2 新增误检分类、R1.3 工作点比较，之后才考虑 P3 对接与 ROI/切片。
- 16 张 test 继续保留不推理；无硬件采购、真实采集或 P3 修改；耗时为本机软件数据，不代表 FPGA 性能。

## 证据

- [总报告与命令日志]({link}/report.json)、[核验记录]({link}/verification.json)。
- [配对逐图对照]({link}/comparison_paired.html)、[128 种子对照]({link}/comparison_seed_128.json)、[320 种子对照]({link}/comparison_seed_320.json)。
- 两臂复核页：[{link}/review128/index.html]({link}/review128/index.html)、[{link}/review320/index.html]({link}/review320/index.html)。
'''
    output.write_text(text, encoding='utf-8')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
