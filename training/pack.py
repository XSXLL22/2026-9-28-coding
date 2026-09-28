"""P4.2/4.1 pack builder: integer graph, quant parameters, accumulation checks, P4PKB1 emission.

Consumes the traced float graph (training.exporter) and the calibration report
(training.calibrate), and emits a self-contained model pack:

  model_pack.bin   P4PKB01 binary consumed by both integer references (Python and C)
  model_pack.json  human-readable manifest with provenance and per-layer parameters

Integer graph = float graph plus explicit requantize nodes wherever the contract demands a
scale change (every concat input; the second operand of every add).
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from runtime.common import file_hash, write_json
from training.exporter import (
    ACT_CODES, CLASS_CODES, CONTRACT_PATH, Graph, Node, OP_CODES, Tracer,
    load_detection_model, quantize_weights, round_half_up, saturate_int8,
)

MAGIC = b'P4PKB01\n'
FORMAT_VERSION = 'p4pkb-1'
M_MIN, M_MAX = 2 ** 30, 2 ** 31


def multiplier_shift(real_scale):
    """Contract v1.3 §4: M = round_half_up(r*2^n) in [2^30, 2^31), n in [0, MAX_SHIFT].
    n = 0 is legal (r >= 2^30); negative n is rejected because the C/RTL right shift
    would be undefined; absurdly small r (n > MAX_SHIFT) is rejected for the same reason.
    """
    if not np.isfinite(real_scale) or real_scale <= 0:
        raise ValueError(f'invalid real scale {real_scale}')
    n = int(np.ceil(np.log2(M_MIN / real_scale)))
    if n < 0:
        raise ValueError(f'real scale {real_scale} implies negative shift; '
                         f'scale ratios >= 2^{M_MIN.bit_length()} are outside the contract')
    if n > MAX_SHIFT:
        raise ValueError(f'real scale {real_scale} implies shift {n} > {MAX_SHIFT}')
    while True:
        m = int(round_half_up(np.array([real_scale * (2.0 ** n)]))[0])
        if m < M_MAX:
            break
        n += 1
        if n > MAX_SHIFT:
            raise ValueError(f'shift exceeded {MAX_SHIFT} while normalizing M')
    while n > 0 and m < M_MIN:
        n -= 1
        m = int(round_half_up(np.array([real_scale * (2.0 ** n)]))[0])
        if m >= M_MAX:
            n += 1
            m = int(round_half_up(np.array([real_scale * (2.0 ** n)]))[0])
            break
    rel = abs(m * (2.0 ** (-n)) - real_scale) / real_scale
    return m, n, rel


def silu_lut(sy):
    """Contract §5: 256-entry table, out = quantize(silu_fp(v*sy), sy); lookup only at runtime."""
    v = np.arange(-128, 129, dtype=np.float64) * sy
    out = v / (1.0 + np.exp(-v))          # silu_fp
    return saturate_int8(round_half_up(out / sy)).astype(np.uint8)


MAX_ABS_INPUT = 128  # activations span [-128, 127]; |qx| can reach 128 (contract v1.3 §6)
MAX_SHIFT = 62       # keeps acc*M + half inside int64 with margin for any legal acc


def accumulation_bound(node, qw, qb):
    """Contract v1.3 §6, per output channel:
        B[o] = |qb[o]| + max_abs_input * sum(|qw[o,:]|),  max_abs_input = 128
    (activations span [-128, 127], so |qx| may be 128; the product form with 127 was
    unsound). Returns the worst-channel bound as a Python int and RAISES when it does
    not fit int32 -- the check is the function's job, not the caller's comment.

    qw must be widened to a wider dtype before this call; np.abs on an int8 -128
    overflows silently back to -128.
    """
    weights = np.abs(qw.astype(np.int64)).reshape(node.cout, -1)
    bias = np.abs(qb.astype(np.int64))
    bounds = bias + MAX_ABS_INPUT * weights.sum(axis=1)
    worst = int(bounds.max())
    if worst > 2 ** 31 - 1:
        raise ValueError(f'{node.name}: accumulation bound {worst} exceeds int32 '
                         f'(contract v1.3 section 6); widen the accumulator or split channels')
    return worst


class PackBuilder:
    def __init__(self, graph: Graph, calib: dict):
        self.graph = graph
        self.scales = {int(k): v for k, v in calib['tensor_scales'].items()}
        self.calib = calib

    def build(self):
        g, nodes, tensors = self.graph, [], dict(self.graph.tensor_shapes)
        new_scale = dict(self.scales)
        tid_map = {t: t for t in tensors}

        def requant(src_tid, dst_scale, name):
            nonlocal requant_count
            out_tid = len(tensors)
            tensors[out_tid] = tensors[src_tid]
            new_scale[out_tid] = dst_scale
            m, n, rel = multiplier_shift(new_scale[src_tid] / dst_scale)
            if rel > 2 ** -29:
                raise AssertionError(f'{name}: multiplier relative error {rel:.2e}')
            nodes.append(Node(name=name, op='requantize', op_class='PS_INT', inputs=[tid_map[src_tid]],
                              output=out_tid, source_module=name))
            params[name] = {'M': m, 'shift': n}
            return out_tid

        params: dict = {}
        requant_count = 0
        for node in g.nodes:
            inputs = [tid_map[t] for t in node.inputs]
            if node.op == 'concat':
                dst = new_scale[tid_map[node.output]] if tid_map[node.output] in new_scale else self.scales[node.output]
                resolved = []
                for i, src_tid in enumerate(node.inputs):
                    src = self.scales[src_tid]
                    if src == dst:
                        resolved.append(inputs[i])
                    else:
                        requant_count += 1
                        resolved.append(requant(src_tid, dst, f'{node.name}.req{i}'))
                inputs = resolved
            elif node.op == 'add' and self.scales[node.inputs[1]] != self.scales[node.inputs[0]]:
                requant_count += 1
                inputs = [inputs[0], requant(node.inputs[1], self.scales[node.inputs[0]], f'{node.name}.req1')]
            out = tid_map[node.output] if node.output in tid_map else node.output
            out2 = node.output2
            if node.output2 != -1 and node.output2 not in tid_map:
                tid_map[node.output2] = node.output2
            new_node = Node(name=node.name, op=node.op, op_class=node.op_class, inputs=inputs,
                            output=out, cin=node.cin, cout=node.cout, kh=node.kh, kw=node.kw,
                            stride=node.stride, pads=node.pads, act=node.act,
                            source_module=node.source_module, output2=out2)
            nodes.append(new_node)
            if node.op.startswith('conv'):
                sx = self.scales[node.inputs[0]]
                sy = self.scales[node.output]
                qw, sw = quantize_weights(g.weights[node.name][0])
                b_fold = g.weights[node.name][1].numpy().astype(np.float64)
                denom = sx * sw.astype(np.float64)
                qb32 = round_half_up(b_fold / denom).astype(np.int64)
                if np.abs(qb32).max() > 2 ** 31 - 1:
                    raise AssertionError(f'{node.name}: bias exceeds int32')
                m_rows, rels = [], []
                r = sx * sw.astype(np.float64) / sy
                for o in range(node.cout):
                    m_o, n_o, rel = multiplier_shift(float(r[o]))
                    m_rows.append((m_o, n_o))
                    rels.append(rel)
                bound = accumulation_bound(node, qw, qb32)  # raises on int32 violation
                params[node.name] = {'qw': qw, 'qb': qb32.astype(np.int32), 'sw': sw,
                                     'M': np.array([x[0] for x in m_rows], dtype=np.int32),
                                     'shift': np.array([x[1] for x in m_rows], dtype=np.int32),
                                     'bound': bound,
                                     'silu_lut': silu_lut(sy).tobytes() if node.act == 'silu' else None,
                                     'max_multiplier_rel_err': max(rels)}
        self.int_graph = Graph(nodes=nodes, tensor_shapes=tensors, input_shape=g.input_shape)
        self.int_graph.scope_outputs = [tid_map[t] for t in g.scope_outputs]
        self.params = params
        self.scales_int = new_scale
        return self.int_graph, params

    # -- serialization --------------------------------------------------------
    def emit_binary(self, path: Path, meta: dict):
        with path.open('wb') as f:
            f.write(MAGIC)
            f.write(struct.pack('<I', len(self.int_graph.nodes)))
            for node in self.int_graph.nodes:
                name_bytes = node.name.encode('utf-8')
                blob = self.params.get(node.name)
                param_offset, param_len = 0, 0
                if blob is not None:
                    payload = self._param_bytes(node, blob)
                    param_offset, param_len = len(self._body), len(payload)
                    self._body.extend(payload)
                head = struct.pack('<4B', OP_CODES[node.op], ACT_CODES[node.act],
                                   CLASS_CODES[node.op_class], 0)
                head += struct.pack('<5I', node.cin, node.cout, node.kh, node.kw, node.stride)
                head += struct.pack('<4i', *node.pads)
                head += struct.pack('<I', len(node.inputs))
                head += b''.join(struct.pack('<i', t) for t in node.inputs)
                head += struct.pack('<ii', node.output, node.output2)
                head += struct.pack('<II', param_offset, param_len)
                head += struct.pack('<I', len(name_bytes)) + name_bytes
                f.write(head)
            f.write(struct.pack('<I', len(self.int_graph.tensor_shapes)))
            for tid in sorted(self.int_graph.tensor_shapes):
                f.write(struct.pack('<3I', *self.int_graph.tensor_shapes[tid]))
            meta_bytes = json.dumps(meta, ensure_ascii=False).encode('utf-8')
            f.write(struct.pack('<I', len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack('<I', len(self._body)))
            f.write(bytes(self._body))

    def _param_bytes(self, node, blob):
        out = b''
        if node.op.startswith('conv'):
            out += blob['qw'].tobytes()
            out += blob['qb'].astype('<i4').tobytes()
            out += blob['M'].astype('<i4').tobytes()
            out += blob['shift'].astype('<i4').tobytes()
            if blob['silu_lut'] is not None:
                out += blob['silu_lut']
        elif node.op == 'requantize':
            out += struct.pack('<i', blob['M']) + struct.pack('<i', blob['shift'])
        return out

    @classmethod
    def create(cls, weights_path, calib_path):
        from training.exporter import CONTRACT_PATH
        model = load_detection_model(weights_path)
        graph = Tracer(model.model).trace()
        calib = json.loads(Path(calib_path).read_text(encoding='utf-8'))
        builder = cls(graph, calib)
        builder.root_model = model
        builder._body = bytearray()
        return builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='experiments/p2_expanded_train/fit/weights/best.pt')
    parser.add_argument('--calibration', default='experiments/p4_ptq/report.json')
    parser.add_argument('--output', default='training/export/p4_baseline128_v1')
    parser.add_argument('--model-id', default='p4-baseline128')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    builder = PackBuilder.create(args.weights, args.calibration)
    int_graph, params = builder.build()
    # body offsets need a second pass once all payloads are known
    builder._body = bytearray()
    silu_count = sum(1 for n in int_graph.nodes if n.act == 'silu')
    requant_nodes = [n.name for n in int_graph.nodes if n.op == 'requantize']
    meta = {'model_id': args.model_id, 'format_version': FORMAT_VERSION,
            'contract_version': contract['contract_version'],
            'weights_sha256': file_hash(args.weights),
            'calibration_report_sha256': file_hash(args.calibration),
            'graph': {'nodes': len(int_graph.nodes), 'tensors': len(int_graph.tensor_shapes),
                      'requantize_nodes': len(requant_nodes), 'scope_outputs': int_graph.scope_outputs},
            'quantization': {'conv_layers': sum(1 for n in int_graph.nodes if n.op.startswith('conv')),
                             'silu_luts': silu_count,
                             'worst_multiplier_rel_err': max(
                                 (p.get('max_multiplier_rel_err', 0.0) for p in params.values()), default=0.0),
                             'worst_accumulation_bound_ratio': max(
                                 (p['bound'] / (2 ** 31 - 1) for p in params.values() if 'bound' in p), default=0.0)},
            'layer_index': [{'name': n.name, 'op': n.op, 'class': n.op_class, 'act': n.act}
                            for n in int_graph.nodes]}
    bin_path = output / 'model_pack.bin'
    builder.emit_binary(bin_path, meta)
    write_json(output / 'model_pack.json', {**meta, 'pack_sha256': file_hash(bin_path),
                                            'requantize_node_names': requant_nodes})
    worst_ratio = meta['quantization']['worst_accumulation_bound_ratio']
    print(f"pack: {len(int_graph.nodes)} nodes ({meta['quantization']['conv_layers']} conv, "
          f"{len(requant_nodes)} requant, {silu_count} silu LUTs), "
          f"worst bound ratio {worst_ratio:.4f}, {bin_path.stat().st_size // 1024} KiB")


if __name__ == '__main__':
    main()
