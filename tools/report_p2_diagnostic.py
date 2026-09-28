"""Render diagnostic charts and a local review page from completed D0 output."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    directory = Path(args.run)
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'complete':
        raise ValueError('Diagnosis is not complete')
    if (directory / 'index.html').exists() or (directory / 'threshold_curves.png').exists():
        raise FileExistsError('Review already exists; do not overwrite')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, color in (('smoke', '#276ba4'), ('fire', '#ce5133')):
        points = [s['per_class'][name] for s in report['sweep']]
        x, y = [p['recall'] for p in points], [p['precision'] for p in points]
        axes[0].plot(x, y, 'o-', color=color, label=name)
        for a, b, sweep in zip(x, y, report['sweep']):
            axes[0].annotate(str(sweep['threshold']), (a, b), xytext=(3, 4), textcoords='offset points', fontsize=8)
        axes[1].plot([s['threshold'] for s in report['sweep']], [p['recall'] for p in points], 'o-', color=color, label=name)
    axes[0].set(xlabel='Recall', ylabel='Precision', title='Six operating points (not full AP curve)', xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel='Confidence threshold', ylabel='Recall', title='Frozen 109-image development set', ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(directory / 'threshold_curves.png', dpi=160)
    plt.close(fig)
    rows = []
    for sweep in report['sweep']:
        smoke, fire = (sweep['per_class'][name] for name in ('smoke', 'fire'))
        rows.append(f'<tr><td>{sweep["threshold"]}</td><td>{smoke["tp"]}/72</td><td>{smoke["fp"]}</td>'
                    f'<td>{fire["tp"]}/90</td><td>{fire["fp"]}</td><td>{sweep["negative_images_with_detection"]}/43</td></tr>')
    previews = json.loads((directory / 'augmentation_previews.json').read_text(encoding='utf-8'))
    cards = []
    for row in previews:
        if row['variant'] == 'no_mosaic': continue
        sources = ', '.join(Path(s).name for s in row['sources'])
        cards.append(f'<figure><img src="{row["file"]}" alt="{row["variant"]} 实际增强图与标签">'
                     f'<figcaption>{row["variant"]} · {html.escape(sources)}</figcaption></figure>')
    text = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P2 D0 诊断</title>'
            '<style>body{font:16px/1.6 system-ui;max-width:1120px;margin:32px auto;padding:0 24px;color:#18283c}'
            'table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd3db}'
            '.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}figure{margin:0}'
            'img{max-width:100%}figcaption{overflow-wrap:anywhere;font-size:13px}aside{background:#fff5df;padding:16px}</style>'
            '<h1>P2 D0：识别瓶颈诊断</h1><aside>没有训练或更新权重。只使用开发验证图；不是校园验收或独立测试。'
            '自动框重叠分类不等于人工确认的错误原因。</aside><h2>阈值取舍</h2><img src="threshold_curves.png" alt="六个阈值的精确率召回率及召回率变化">'
            '<table><tr><th>阈值</th><th>烟雾检出</th><th>烟雾误检框</th><th>明火检出</th><th>明火误检框</th><th>负样本有预测</th></tr>'
            + ''.join(rows) + '</table><h2>实际训练增强抽样</h2><p>current：全程 Mosaic 配方；closed_mosaic：关闭 Mosaic 后的状态。'
            '每种状态抽样 128 次，下面各展示前 8 次；绿框为增强后标签。图片放大显示不增加输入信息。'
            'Mosaic 引入多张图并裁剪，不能把两种状态的框总数差直接当作标签损失率。</p><div class="grid">'
            + ''.join(cards) + '</div><p><a href="report.json">完整统计</a> · <a href="augmentation_candidates.json">过滤阶段追踪</a>'
            ' · <a href="candidates.json">逐图低分候选</a></p></html>')
    (directory / 'index.html').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
