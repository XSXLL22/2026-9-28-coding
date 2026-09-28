"""Render this versioned P2 experiment report directly from verified artifacts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.common import file_hash


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def main():
    output=ROOT/'experiments/P2扩充实验报告.md'
    if output.exists():
        raise FileExistsError(output)
    training=read('experiments/p2_expanded_train/report.json')
    old=read('experiments/p2_expanded_old_evaluation/report.json')
    new=read('experiments/p2_expanded_new_evaluation/report.json')
    comparison=read('experiments/p2_expanded_comparison.json')
    prep=read('datasets/public_expanded_v1_prepared/preparation.json')
    common=read('datasets/public_expanded_comparison/comparison_preparation.json')
    pipeline=read('experiments/p2_expanded_pipeline_check/verification.json')
    if pipeline['status']!='passed' or pipeline['repeat_detection_count']!=80:
        raise ValueError('Versioned pipeline verification is incomplete or different')
    if any(r['status']!='complete' for r in (training,old,new,comparison,prep,common)):
        raise ValueError('Cannot publish incomplete experiment')
    if file_hash(training['best_weights']) != training['best_weights_sha256']:
        raise ValueError('Trained weights changed')
    if not (comparison['old_model']==old['initial_weights_sha256'] and
            comparison['new_model']==new['initial_weights_sha256']==training['best_weights_sha256']):
        raise ValueError('Model identity mismatch')
    if old['args']['data'] != new['args']['data']:
        raise ValueError('Evaluation dataset mismatch')
    a,b=old['validation']['128'],new['validation']['128']
    delta=100*(b['aggregate']['metrics/mAP50(B)']-a['aggregate']['metrics/mAP50(B)'])
    lines=['# P2 扩充数据与同集对照实验报告','', '2026-09-27。全部数字来自已完成的本地报告，未重新采集校园数据。','',
           f"本轮共同验证集 mAP50 从 {a['aggregate']['metrics/mAP50(B)']:.2%} 变为 {b['aggregate']['metrics/mAP50(B)']:.2%}，变化 {delta:+.2f} 个百分点。"
           '这只是公开开发验证结果，不能视为校园报警能力或独立测试精度。','',
           '## 数据与运行条件','',
           '原始扩充集 656 张（train 512 / val 128 / test 16），约 90.76 MiB。核对原始哈希、解码、标签和 EXIF；'
           '独立裁剪 3 处越界框；63 位 pHash 距离≤6 筛出 301 对相似候选（110 对跨集合），42 个相似分组。'
           '隔离 train 118、val 14，最终训练 394 张、选权重验证 114 张、测试 16 张。'
           '进一步排除与旧训练图处于同一筛查组的 5 张验证图，新旧模型共同评估 109 张。','',
           '共同集合包含 43 张负样本、72 个烟雾框、90 个明火框。测试集未推理、不选阈值。'
           '真实视频/地点分组仍未知，筛查只排除已识别候选，不保证所有相关场景均已隔离。'
           '数据与抽样视觉审核详见 [审核记录](P2扩充数据审核记录.md)。','',
           f"新训练从官方 YOLOv8n 原始权重开始，128 输入、CPU、batch 8、4 线程、50 轮、seed 20260926、AdamW、初始学习率 0.001。"
           f"训练入口（含审核和尺寸评估）耗时 {training['duration_seconds']:.1f} 秒。"
           '旧模型从 48 张图训练 50 轮；新旧数据组成及每轮步数都改变，因此不是只控制样本数量的单变量消融。','',
           f"新 best.pt SHA256：`{training['best_weights_sha256']}`。",'',
           '## 同一 109 张验证图上的 AP','',
           '| 模型 | smoke AP50 | fire AP50 | 两类 mAP50 | mAP50:95 |',
           '|---|---:|---:|---:|---:|']
    for name,r in [('旧 48 张训练',a),('新 394 张训练',b)]:
        lines.append(f"| {name} | {r['per_class']['smoke']['ap50']:.2%} | {r['per_class']['fire']['ap50']:.2%} | "
                     f"{r['aggregate']['metrics/mAP50(B)']:.2%} | {r['aggregate']['metrics/mAP50-95(B)']:.2%} |")
    lines += ['', '## 固定工作点与退步检查','',
              '以下为 conf=0.25、NMS IoU=0.45、同类匹配 IoU=0.5，按全图统计，不按 ROI 过滤。AP 使用库内阈值扫描与不同 NMS 设置，不能直接拿其 P/R 代替本表。','',
              '| 模型 | 类别 | TP | FP | FN | Precision | Recall |','|---|---|---:|---:|---:|---:|---:|']
    for model,key in [('旧','old_totals'),('新','new_totals')]:
        for name in ('smoke','fire'):
            r=comparison[key][name]; tp,fp,fn=(r[k] for k in ('tp','fp','fn'))
            precision=f'{tp/(tp+fp):.2%}' if tp+fp else '无定义'
            recall=f'{tp/(tp+fn):.2%}' if tp+fn else '无定义'
            lines.append(f'| {model} | {name} | {tp} | {fp} | {fn} | {precision} | {recall} |')
    n=comparison['negative_images']
    lines += ['',f"逐目标配对：新模型补检 {comparison['recovered_targets']} 个旧模型漏掉的目标，"
              f"同时丢失 {comparison['lost_targets']} 个旧模型曾检出的目标。"
              f"{n['images']} 张负样本中，出现预测的图片由 {n['old_with_predictions']} 张变为 {n['new_with_predictions']} 张。"
              '这是图像级误检统计，不是事件级报警率。','',
              '## 小目标表现','',
              '近似缩放短边为原图框短边×128/原图最长边；分桶不是 COCO 面积分桶。','',
              '| 类别 | 缩放短边 | 标注框 | 旧检出 | 新检出 |','|---|---|---:|---:|---:|']
    for name in ('smoke','fire'):
        for bucket,label in [('lt8','<8 px'),('8to16','8–<16 px'),('ge16','≥16 px')]:
            x,y=comparison['old_size_counts'][name][bucket],comparison['new_size_counts'][name][bucket]
            if x['gt'] != y['gt']:
                raise ValueError('Size-bin population mismatch')
            lines.append(f"| {name} | {label} | {x['gt']} | {x['tp']} | {y['tp']} |")
    lines += ['', '## 证据与复现','',
              '- [逐图新旧对照页](p2_expanded_comparison.html)：优先展示丢失检出与剩余错误，点击展开。',
              '- [配对统计 JSON](p2_expanded_comparison.json)：逐目标补检/丢失、负样本与尺寸分桶。',
              '- [新模型训练报告](p2_expanded_train/report.json) / [新权重](p2_expanded_train/fit/weights/best.pt)。',
              '- [旧模型同集评估](p2_expanded_old_evaluation/report.json) / [新模型同集评估](p2_expanded_new_evaluation/report.json)。',
              '- [旧逐图复核](p2_expanded_old_review/index.html) / [新逐图复核](p2_expanded_new_review/index.html)。',
              '- [操作说明](../docs/P2扩充数据操作说明.md)与[训练前计划](P2扩充实验计划.md)。','',
              '数据准备、训练、推理和逐图复核各自记录成功/失败状态；成对比较核对输入图片、标签、工作点及模型身份。'
              '当前工具与原有功能共 28 项测试通过，依赖检查通过。','',
              '新模型的单图、目录、视频入口再次通过；同一 109 张图在 128 输入下重复推理，80 个检测框的类别、坐标、分数及 ROI 比例完全一致。'
              '见 [运行核验](p2_expanded_pipeline_check/verification.json)。视频仍为静态图构造的输入夹具，不是火情时序实验。','',
              '已定点查看 3 个丢失目标所在图片和新增 FP 较多的图片，详见 [退步样本复核](P2扩充模型退步样本复核.md)。'
              '部分 FP 是可见烟火的定位偏差，不能等同于无火背景上的报警。','',
              '## 解释边界与下一步','',
              '结果仍来自单次训练、单个种子、规模有限且原始视频分组未知的公开数据。新模型的权重选择使用了这批验证数据，'
              '不得把结果称为最终独立测试或报告跨种子稳定性。烟雾/明火误检漏检仍须逐图复核；落叶、校园数据和现场 ROI 未补齐。','',
              '旧模型评估与新模型训练部分并行，本轮耗时不能用于公平速度优劣结论；所有速度均非 FPGA 测量。'
              '主输入仍为 128，未因一次实验修改硬件目标。未启用 GPU，未实施 P3。','',
              '下一步先复核新增误检与丢失目标，并补来源分组及困难负样本；如进入 P3，只把视觉输出视为带置信度和有效性限制的证据。']
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
