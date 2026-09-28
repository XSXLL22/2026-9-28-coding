# 数据集目录

本目录保存数据台账、类别定义、划分清单和数据说明。原始图片及处理后的图片默认不提交到 Git。

## 目录约定

```text
datasets/
  classes.txt
  manifest.csv
  标注规范.md
  raw/README.md
  processed/README.md
  processed/images/{train,val,test}/
  processed/labels/{train,val,test}/
  splits/{train,val,test}.txt
```

标注使用 YOLO 文本格式，每行是 `class_id x_center y_center width height`，坐标归一化到 0–1。没有目标的负样本必须保留对应的空标签文件，避免无法区分“确认无目标”和“漏标”。

`manifest.csv` 每张图片一行，字段如下：

| 字段 | 含义 |
|---|---|
| sample_id | 全局唯一样本编号 |
| image_relpath | 相对 `datasets/processed` 的图片路径 |
| label_relpath | 相对 `datasets/processed` 的标签路径 |
| split | train、val 或 test |
| group_id | 同一地点、视频或连续采集序列的分组编号 |
| source_id | 对应 `sources.csv` 的来源编号 |
| capture_date | ISO 日期，未知留空 |
| license | 数据许可或 `self-collected` |
| notes | 其他说明 |

正式验收命令：

```powershell
python tools/validate_dataset.py
```

在尚未放入数据时，只检查目录和台账格式：

```powershell
python tools/validate_dataset.py --structure-only
```
