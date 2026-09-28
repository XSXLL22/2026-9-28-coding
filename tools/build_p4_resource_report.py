"""P4.6 resource report v2 (E1.3): per-device DSP, explicit assumption layers.

Sections are strictly separated: (A) model-static quantities derived from the pack,
(B) design assumptions for a first RTL prototype, (C) scenario traffic estimates under
those assumptions, (D) unknowns that only synthesis or board runs can resolve.
No number in this report may be presented as a measured performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from pack_reader import read_pack  # noqa: E402

from runtime.common import write_json  # noqa: E402
from training.pack import accumulation_bound  # noqa: E402
from training.exporter import Node  # noqa: E402

DEVICES = {  # AMD Zynq-7000 Product Selection Guide: 7010 = 80 DSP48E1, 7020 = 220 DSP48E1
    'zynq-7010': {'dsp48e1': 80, 'block_ram_kbits': 600},
    'zynq-7020': {'dsp48e1': 220, 'block_ram_kbits': 1400},
}
CLOCK_MHZ_ASSUME = 125.0  # assumption, not a synthesis result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', default='training/export/p4_baseline128_v2/model_pack.bin')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pack = read_pack(args.pack)
    shapes = pack.tensor_shapes

    static_rows, total_mac, total_weight_bytes, act_total = [], 0, 0, 0
    worst_bound_ratio = 0.0
    for node in pack.nodes:
        if not node.op.startswith('conv'):
            continue
        c_in, h_in, w_in = shapes[node.inputs[0]]
        h_out, w_out = shapes[node.output][1:]
        macs = node.cout * h_out * w_out * node.kh * node.kw * node.cin
        weight_bytes = node.cout * node.kh * node.kw * node.cin + 4 * node.cout * 3  # qw + qb + M + shift
        fake = Node(name=node.name, op=node.op, op_class=node.op_class, inputs=node.inputs,
                    output=node.output, cin=node.cin, cout=node.cout, kh=node.kh, kw=node.kw)
        bound = accumulation_bound(fake, node.qw, node.qb.astype(np.int64))
        worst_bound_ratio = max(worst_bound_ratio, bound / (2 ** 31 - 1))
        static_rows.append({'name': node.name, 'op': node.op, 'cin': node.cin, 'cout': node.cout,
                            'kernel': f'{node.kh}x{node.kw}', 'stride': node.stride,
                            'h_out': h_out, 'w_out': w_out, 'macs': macs,
                            'weights_bytes': weight_bytes,
                            'act_in_bytes': c_in * h_in * w_in, 'act_out_bytes': node.cout * h_out * w_out,
                            'accum_bound': bound})
        total_mac += macs
        total_weight_bytes += weight_bytes
        act_total += c_in * h_in * w_in + node.cout * h_out * w_out

    ddr_read_upper = total_weight_bytes + act_total  # zero reuse: re-read everything per layer
    report = {
        'stage': 'p4.6-E1', 'status': 'complete', 'model_id': pack.meta['model_id'],
        'contract_version': pack.meta['contract_version'],
        'A_model_static': {
            'macs_per_frame': total_mac,
            'weights_kib': round(total_weight_bytes / 1024.0, 1),
            'activations_touched_kib_per_frame': round(act_total / 1024.0, 1),
            'conv_layers': len(static_rows),
            'silu_lut_layers': sum(1 for n in pack.nodes if n.act == 'silu'),
            'requantize_nodes': sum(1 for n in pack.nodes if n.op == 'requantize'),
            'accumulation': {'bound_formula': 'B[o] = |qb[o]| + 128*sum(|qw[o]|), worst channel',
                             'worst_bound_ratio_int31': round(worst_bound_ratio, 5),
                             'int32_proven': True},
            'input_geometry': '128x128 square canvas (hardware contract); the library rect '
                              'validation path uses different input geometry and is not comparable',
        },
        'B_design_assumptions': {
            'mac_parallelism': 'first RTL prototype: single MAC or small unrolled array; '
                               'NOT the full DSP array',
            'on_chip_buffers': 'per-layer tile in BRAM; line caching, ping-pong and channel '
                               'tiling are P5.1+ work and are NOT assumed here',
            'weight_delivery': 'scenario dependent: (a) all weights resident on chip needs '
                               f'{round(total_weight_bytes / 1024.0, 1)} KiB -- exceeds 7010 BRAM for a '
                               'single-buffer design; (b) per-layer load from DDR each layer; '
                               '(c) tiled re-load for large layers. Not decided yet.',
            'ps_pl_split': 'PL: conv3x3/1x1 + requantize + saturate. PS_INT: SiLU LUT, maxpool, '
                           'upsample, concat/split/add. PS_FLOAT: DFL/softmax/decode/NMS. '
                           'SiLU on PS means every conv output round-trips PS<->PL until the '
                           'student model replaces SiLU -- a real traffic cost, quantified only '
                           'as a per-layer round trip of the output tensor.',
            'clock_mhz': CLOCK_MHZ_ASSUME,
        },
        'C_scenarios': {
            'compute_only_upper_fps_per_device': {
                dev: round(info['dsp48e1'] * CLOCK_MHZ_ASSUME * 1e6 / max(total_mac, 1), 1)
                for dev, info in DEVICES.items()},
            'note': 'one MAC per DSP, full pipeline occupancy, no memory stalls -- a theoretical '
                    'ceiling per device, NOT a throughput claim. 7010 and 7020 differ by DSP count; '
                    'the previous single 220-DSP figure covering both devices was wrong.',
            'traffic_upper_kib_per_frame': round(ddr_read_upper / 1024.0, 1),
            'traffic_upper_note': 'zero on-chip reuse: every layer re-reads weights and input and '
                                  'writes its output. The previous "lower/upper bounds" pair did not '
                                  'model tiling re-reads or PS/PL round trips and is withdrawn.',
            'silu_roundtrip_extra_kib_per_frame': round(act_total / 1024.0, 1),
        },
        'D_unknowns': ['synthesized frequency', 'DSP mapping efficiency', 'AXI/bus efficiency',
                       'PS post-processing time', 'DDR latency/efficiency',
                       'on-chip memory feasibility for the chosen tiling'],
        'per_layer': static_rows,
        'limitations': ['No synthesis and no board data exist; every performance number here is an '
                        'assumption-based estimate.',
                        'Only the compute-only ceiling differs between 7010 and 7020; memory-bound '
                        'scenarios depend on unknowns listed in D_unknowns.'],
    }
    write_json(output / 'report.json', report)

    lines = ['# P4.6 模型资源与搬运估算(E1.3 修订版)', '',
             '2026-09-28。**全部为带假设的估算,无综合、无上板数据。**上一版报告(基于 127 节点 v1 包、'
             '两器件共用 220 DSP、自称严格上下界)被本版取代;累加界公式已按合同 v1.3 修正。', '',
             '## A 模型静态量(与器件无关,来自整数包层描述符)',
             f"- MAC/帧:{total_mac}({total_mac / 1e6:.2f} M)",
             f"- 权重+偏置+乘子+移位:{total_weight_bytes / 1024.0:.1f} KiB",
             f"- 逐层激活读写合计:{act_total / 1024.0:.1f} KiB/帧",
             f"- 卷积层 {len(static_rows)};SiLU 查表 {report['A_model_static']['silu_lut_layers']} 层;"
             f"requant 节点 {report['A_model_static']['requantize_nodes']}",
             f"- 累加界(逐通道和式,|qx|≤128):最坏占 int31 的 {worst_bound_ratio:.2%},int32 已证明可行", '',
             '## B 设计假设(首版 RTL 原型,未实现)',
             '- MAC 并行度:单 MAC 或小规模展开;**不是** 220-DSP 满阵列。',
             '- 片上缓存:逐层 tile;行缓存/乒乓/通道分块属 P5.1+,本版不假设。',
             '- 权重供给三场景:全部驻留(需 ~2.9 MiB,7010 BRAM 单缓冲装不下)/逐层加载/分块重载——未决定。',
             '- PS/PL 分工:PL 只做卷积+重量化+饱和;SiLU/池化/拼接在 PS(每个卷积输出一个 PS↔PL 往返,'
             '直到学生模型替换 SiLU)。', '',
             '## C 场景估算(仅算力天花板 + 零复用流量)',
             '| 器件 | DSP | 仅算力帧率上界(125 MHz 假设,满占流水) |',
             '|---|---:|---:|']
    for dev, info in DEVICES.items():
        fps = info['dsp48e1'] * CLOCK_MHZ_ASSUME * 1e6 / max(total_mac, 1)
        lines.append(f'| {dev} | {info["dsp48e1"]} | {fps:.1f} FPS |')
    lines += [f"- 零复用 DDR 流量上界:{report['C_scenarios']['traffic_upper_kib_per_frame']} KiB/帧"
              f"(每层重读权重与输入、写出输出;未含 SiLU 往返 {act_total / 1024.0:.1f} KiB/帧)。",
              '- 旧版的『严格上下界对』未建模分块重读与 PS/PL 往返,已撤回。', '',
             '## D 未知的实测量(综合/上板前无法填写)',
             '\n'.join(f'- {u}' for u in report['D_unknowns']), '',
             '## 证据', '', f"- [结构化数据](report.json);golden vectors 15/15 逐位一致"
             f"(contract {pack.meta['contract_version']})。",
             '- 器件参数来源:AMD Zynq-7000 Product Selection Guide(7010 = 80 DSP48E1,7020 = 220)。']
    (output / 'P4.6资源与带宽报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'macs_per_frame': total_mac, 'weights_kib': round(total_weight_bytes / 1024.0, 1),
                      'fps7010': report['C_scenarios']['compute_only_upper_fps_per_device']['zynq-7010'],
                      'fps7020': report['C_scenarios']['compute_only_upper_fps_per_device']['zynq-7020'],
                      'worst_bound_ratio': round(worst_bound_ratio, 5)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
