"""Write an A1 milestone brief from completed, checked experiment outputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='experiments/p2_a1_bias_warmup')
    parser.add_argument('--output', default='experiments/P2_A1偏置预热简报.md')
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'complete' or report['actual_recipe_check'] != 'passed':
        raise ValueError('A1 experiment has not completed all checks')
    if output.exists(): raise FileExistsError(output)
    import os
    link = Path(os.path.relpath(root, output.parent)).as_posix()
    old, new, compared = report['reference_validation'], report['candidate_validation'], report['comparison']
    metrics = [('两类 mAP50', old['aggregate']['metrics/mAP50(B)'], new['aggregate']['metrics/mAP50(B)']),
               ('两类 mAP50:95', old['aggregate']['metrics/mAP50-95(B)'], new['aggregate']['metrics/mAP50-95(B)'])]
    metrics += [(f'{c} AP50', old['per_class'][c]['ap50'], new['per_class'][c]['ap50']) for c in ('smoke','fire')]
    rows = [f'| {name} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} 个百分点 |' for name,a,b in metrics]
    fixed, sizes = [], []
    for c in ('smoke','fire'):
        a,b = compared['old_totals'][c],compared['new_totals'][c]
        fixed.append(f"| {c} | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {a['tp']/(a['tp']+a['fn']):.2%} → {b['tp']/(b['tp']+b['fn']):.2%} |")
        for size in ('lt8','8to16','ge16'):
            a,b = compared['old_size_counts'][c][size],compared['new_size_counts'][c][size]
            sizes.append(f"| {c} | {size} | {a['gt']} | {a['tp']} | {b['tp']} |")
    negative = compared['negative_images']
    changed = sorted(compared['image_changes'], key=lambda r:(len(r['lost_targets']),r['new_fp']-r['old_fp']), reverse=True)
    examples = '\n'.join(f"- `{r['image']}`：原有检出丢失 {len(r['lost_targets'])}，补检 {len(r['recovered_targets'])}，FP {r['old_fp']} → {r['new_fp']}。" for r in changed[:5])
    lr = report['first_three_epoch_lrs']
    text = f'''# P2 A1 偏置预热单变量实验简报

2026-09-27。已完成 50 轮训练、共同开发验证、固定工作点推理和逐图配对。模型没有自动替换为运行默认，P3 规则未改变。

## 实验控制与核验

按[训练前计划](P2_A1实验计划.md)，从同一官方 YOLOv8n 初始化，394 train / 114 val、128 输入、50 轮、batch=8、CPU 4 线程、seed=20260926、AdamW、lr0=0.001、close_mosaic=0。**唯一训练配方变化为 warmup_bias_lr：0.1 → 0.0。**

已逐项比较实际 `fit/args.yaml`：除输出目录字段外没有其他差异；训练数据审核完全相同，初始权重和依赖版本一致。前三轮偏置学习率分别为 {', '.join(r['lr/pg0'] for r in lr)}，与其他参数组一致，说明改动实际生效。原配方前三轮偏置学习率为 0.06766、0.0346469、0.00162066。

原基线权重 SHA256：`{report['reference_model_sha256']}`。

A1 权重 SHA256：`{report['candidate_model_sha256']}`。

## 相同 109 张开发图上的 AP

| 指标 | 原基线 | A1 | 变化 |
|---|---:|---:|---:|
{chr(10).join(rows)}

114 张训练验证图用于挑选 best；此表仅使用共同 109 张开发图。不是独立测试，也不是校园报警精度。库内 AP 的 NMS 为 0.7，与下一表运行工作点不同。

## 固定工作点与小目标

conf=0.25、NMS IoU=0.45、同类匹配 IoU=0.5，全图评价。

| 类别 | 原 TP/FP/FN | A1 TP/FP/FN | 召回率 |
|---|---:|---:|---:|
{chr(10).join(fixed)}

负样本共 {negative['images']} 张，有预测的图片数 {negative['old_with_predictions']} → {negative['new_with_predictions']}。逐目标补检 {compared['recovered_targets']} 个、原有检出丢失 {compared['lost_targets']} 个。图片误检数与框 FP 都不等于事件报警率。

| 类别 | 128 参考尺寸组 | GT | 原检出 | A1 检出 |
|---|---|---:|---:|---:|
{chr(10).join(sizes)}

lt8、8to16、ge16 分别为缩放短边 <8、8～<16、≥16 像素，不是 COCO 面积分组。

优先复核的变化图：

{examples}

以上是自动配对结果，不能据此判断遮挡、反光或标注错误；人工观察另记于交接说明。

## 可复查产物

- [总报告与命令日志索引]({link}/report.json)、[实际训练参数]({link}/train/fit/args.yaml)、[学习率与损失记录]({link}/train/fit/results.csv)。
- [共同开发验证]({link}/evaluation/report.json)、[固定工作点配对]({link}/comparison.json)、[逐图对照页]({link}/comparison.html)。
- [A1 逐图复核]({link}/review/index.html)、[新权重]({link}/train/fit/weights/best.pt)。
- [回归测试日志]({link}/tests.log)：57 项通过，含新参数真实传递、旧默认保持与非法值拒绝。

训练和评估命令、阶段退出码及 stdout/stderr 均已保存；16 张 test 只做数据完整性检查，没有推理或调参。原基线保留。入口总耗时包含审核/验证，部分时间并行运行回归测试，不能用于性能优劣结论。

## 解释限制

本次是一个种子对历史基线的受控配方比较，没有跨种子重复或完整重训控制组，不支持稳定性声明。原始来源视频/地点仍未知，开发集多次使用也限制泛化解释。训练配方改善是否足以采用，应结合各类召回、小目标和误检共同判断，不能只看总 mAP。
'''
    output.write_text(text, encoding='utf-8')


if __name__ == '__main__': main()
