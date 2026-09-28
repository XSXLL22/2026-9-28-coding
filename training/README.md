# 训练与量化

已锁定 Ultralytics 8.3.228 / YOLOv8n，提供两个可运行入口：

- `python -m training.baseline train|evaluate`：严格数据检查、训练、96/128 输入验证和模型来源记录。
- `python -m training.error_analysis`：根据真实检测 JSONL 与标签统计固定阈值下的 TP/FP/FN。

R1 增加软件诊断尺寸 320：训练使用 `--imgsz 320 --eval-sizes 320`，评估使用 `--eval-sizes 320`，推理使用 `runtime.vision --imgsz 320`。默认训练/推理尺寸仍为 128。`tools.build_failure_review --size-reference 128` 固定目标分组尺度；`tools.compare_vision_runs --allow-cross-size` 才允许比较不同输入，且要求双方 NMS 和分组参考一致。旧复核 JSON 没有 NMS 字段时应从原始推理记录重新生成，不凭记忆补值。

训练入口新增可选 `--warmup-bias-lr`（有限数值，范围 0～1，仅限 train）。省略时不向框架传此参数，保留原默认行为；显式传 `0.0` 用于 A1 偏置预热消融。实际传入的全部训练选项及入口代码哈希写入 `report.json`，框架解析后的配置另存于 `fit/args.yaml`。参数传递与非法值拒绝已由入口测试覆盖。

A2 新增 `--close-mosaic N`：仅限 train，整数 0～epochs，省略时保持项目原默认 0。训练每轮首次 `on_train_batch_start` 将实际增强图的 Mosaic 等概率写入 `augmentation_epochs.jsonl`，用于确认末段切换；观察器不调用增强或额外消耗随机数。50 轮训练配 `--close-mosaic 10` 时，第 41 轮起应记录 Mosaic 概率为 0。关闭 Mosaic 仍保留其他几何变换。

R1.1 起多种子配对实验使用 `tools.run_r11_paired_experiment --seed <新种子>`（种子必须显式给出且不等于冻结种子 20260926），一次运行同时训练 128/320 两臂并完成三组对照；冻结的 `tools.run_r1_experiment` 保持单种子不变。误检复核用 `tools.prepare_fp_review`（同类 IoU≥0.3 新旧 FP 配对 + 上下文裁剪图）；工作点扫点用 `tools.run_r13_working_points`（预声明网格，每方案真实重推理）。

命令见 [P2 操作说明](../docs/P2操作说明.md)，实测结论见 [基线报告](../experiments/baseline_report.md)。当前为公开烟火两类小样本软件试验；PTQ、QAT、FPGA 算子适配和导出尚未实现。
