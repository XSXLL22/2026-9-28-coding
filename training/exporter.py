"""P4.1 exporter: BN folding, symbolic graph tracing, op audit, float verification, weight quantization.

Contract: configs/quantization_contract.json (p4-contract-1.1) and docs/定点格式说明.md.
Node names mirror the checkpoint's module paths; verification compares executor outputs
against forward-hook captures of the original model, node by node. Composites (C2f, SPPF,
Detect) emit their children in eval-forward order, so hook fire order matches node order.
Nodes without a source module (chunk/add/internal concat) are verified transitively:
every Conv output and every scope output carries a hook.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from runtime.common import ROOT, file_hash, write_json

CONTRACT_PATH = ROOT / 'configs' / 'quantization_contract.json'
OPS_PL_V1 = {'conv3x3_s1', 'conv3x3_s2', 'conv1x1_s1', 'conv1x1_s2'}
OPS_PS_INT = {'silu', 'add', 'maxpool_5x5', 'upsample_nearest_2x', 'concat', 'split_chunk', 'requantize'}

OP_CODES = {'conv3x3_s1': 0, 'conv3x3_s2': 1, 'conv1x1_s1': 2, 'conv1x1_s2': 3,
            'maxpool_5x5': 4, 'upsample_nearest_2x': 5, 'concat': 6, 'split_chunk': 7,
            'add': 8, 'relu': 9, 'requantize': 10}
ACT_CODES = {'none': 0, 'relu': 1, 'silu': 2}
CLASS_CODES = {'PL_V1': 0, 'PS_INT': 1, 'PS_FLOAT': 2}


def round_half_up(x):
    """floor(x + 0.5): the single project-wide rounding rule (contract §4)."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def saturate_int8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def fold_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d | None):
    """Contract §2: w_fold = w*gamma/sqrt(var+eps); b_fold = beta + (b0-mean)*gamma/sqrt(var+eps)."""
    w = conv.weight.detach().float().clone()
    b = conv.bias.detach().float().clone() if conv.bias is not None else torch.zeros(w.shape[0])
    if bn is None:
        return w, b
    gamma, beta = bn.weight.detach().float(), bn.bias.detach().float()
    mean, var = bn.running_mean.detach().float(), bn.running_var.detach().float()
    denom = torch.sqrt(var + bn.eps)
    w_fold = w * (gamma / denom).view(-1, 1, 1, 1)
    b_fold = beta + (b - mean) * gamma / denom
    return w_fold, b_fold


def quantize_weights(w_fold: torch.Tensor):
    """Per-output-channel symmetric int8; sw[o]=max|w[o]|/127; round_half_up; max maps to +/-127."""
    w = w_fold.detach().numpy().astype(np.float64)
    sw = np.max(np.abs(w), axis=(1, 2, 3)) / 127.0
    sw = np.where(sw <= 0, 1e-12, sw)
    qw = saturate_int8(round_half_up(w / sw[:, None, None, None]))
    return qw.astype(np.int8), sw.astype(np.float32)


@dataclass
class Node:
    name: str
    op: str
    op_class: str
    inputs: list
    output: int
    cin: int = 0
    cout: int = 0
    kh: int = 0
    kw: int = 0
    stride: int = 1
    pads: tuple = (0, 0, 0, 0)
    act: str = 'none'
    source_module: str = ''
    output2: int = -1            # split_chunk only: second chunk tensor id


@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    tensor_shapes: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    scope_outputs: list = field(default_factory=list)
    input_shape: tuple = (3, 128, 128)

    def new_tensor(self, shape, producer):
        tid = len(self.tensor_shapes)
        self.tensor_shapes[tid] = tuple(shape)
        return tid

    def add(self, **kwargs):
        node = Node(**kwargs)
        self.nodes.append(node)
        return node

    def node_by_name(self, name):
        return next(n for n in self.nodes if n.name == name)


class Tracer:
    """Symbolic trace with static shapes; emits nodes in ultralytics eval-forward order."""

    def __init__(self, layers, input_shape=(3, 128, 128)):
        self.layers = list(layers)
        self.graph = Graph(input_shape=input_shape)
        self.x0_tid = self.graph.new_tensor(input_shape, 'input')
        # ultralytics index i stores layer i's output at y[i+1]; y[0] is the network input.
        self.y = [self.x0_tid]

    def _module(self, path):
        obj = self.root
        for part in path.split('.'):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    def _conv(self, path, conv2d, bn, in_tid, act):
        if len(conv2d.padding) != 2 or conv2d.padding[0] != conv2d.padding[1]:
            raise ValueError(f'{path}: asymmetric padding {conv2d.padding}')
        w_fold, b_fold = fold_conv_bn(conv2d, bn)
        kh, kw = conv2d.kernel_size
        if (kh, kw) not in ((3, 3), (1, 1)):
            raise ValueError(f'{path}: unsupported kernel {kh}x{kw}')
        if conv2d.groups != 1 or conv2d.dilation != (1, 1) or conv2d.stride[0] != conv2d.stride[1]:
            raise ValueError(f'{path}: unsupported conv geometry')
        stride = conv2d.stride[0]
        op = f'conv{kh}x{kw}_s{stride}'
        if op not in OPS_PL_V1:
            raise ValueError(f'{path}: op {op} not in PL_V1')
        c_in, h, w = self.graph.tensor_shapes[in_tid]
        pad = conv2d.padding[0]
        h_out = (h + 2 * pad - kh) // stride + 1
        w_out = (w + 2 * pad - kw) // stride + 1
        node = self.graph.add(name=path, op=op, op_class='PL_V1', inputs=[in_tid], output=-1,
                              cin=c_in, cout=conv2d.out_channels, kh=kh, kw=kw, stride=stride,
                              pads=(pad, pad, pad, pad), act=act, source_module=path)
        node.output = self.graph.new_tensor((conv2d.out_channels, h_out, w_out), path)
        self.graph.weights[path] = (w_fold, b_fold)
        return node

    def _walk_seq(self, seq, path, in_tid):
        cur = in_tid
        for j, sub in enumerate(seq):
            cur = self._walk(sub, f'{path}.{j}', cur)
        return cur

    def _walk(self, m, path, in_tid):
        kind = type(m).__name__
        if kind == 'Conv':
            return self._conv(path, m.conv, m.bn, in_tid, act='silu').output
        if kind == 'Conv2d':
            if m.bias is None:
                raise ValueError(f'{path}: bare Conv2d without bias is not in the audited contract')
            return self._conv(path, m, None, in_tid, act='none').output
        if kind == 'C2f':
            cv1_tid = self._walk(m.cv1, f'{path}.cv1', in_tid)
            c, h, w = self.graph.tensor_shapes[cv1_tid]
            if c % 2:
                raise ValueError(f'{path}: chunk channels {c} not even')
            split = self.graph.add(name=f'{path}.chunk', op='split_chunk', op_class='PS_INT',
                                   inputs=[cv1_tid], output=-1, source_module=f'{path}.cv1(chunk)')
            split.output = self.graph.new_tensor((c // 2, h, w), split.name)
            split.output2 = self.graph.new_tensor((c // 2, h, w), split.name)
            ys = [split.output, split.output2]
            cur = split.output2
            for i, bottleneck in enumerate(m.m):
                cur = self._walk(bottleneck, f'{path}.m.{i}', cur)
                ys.append(cur)
            cat_tid = self._concat(f'{path}.cat', ys, hookable=False)
            return self._walk(m.cv2, f'{path}.cv2', cat_tid)
        if kind == 'Bottleneck':
            y1_tid = self._walk(m.cv1, f'{path}.cv1', in_tid)
            y2_tid = self._walk(m.cv2, f'{path}.cv2', y1_tid)
            if m.add:
                if self.graph.tensor_shapes[y2_tid] != self.graph.tensor_shapes[in_tid]:
                    raise ValueError(f'{path}: residual shape mismatch')
                node = self.graph.add(name=f'{path}.add', op='add', op_class='PS_INT',
                                      inputs=[in_tid, y2_tid], output=-1, source_module=f'{path}(residual)')
                node.output = self.graph.new_tensor(self.graph.tensor_shapes[in_tid], node.name)
                return node.output
            return y2_tid
        if kind == 'SPPF':
            cv1_tid = self._walk(m.cv1, f'{path}.cv1', in_tid)
            ys, cur = [cv1_tid], cv1_tid
            for i in range(3):
                c, h, w = self.graph.tensor_shapes[cur]
                node = self.graph.add(name=f'{path}.m.{i}', op='maxpool_5x5', op_class='PS_INT',
                                      inputs=[cur], output=-1, kh=5, kw=5, stride=1, pads=(2, 2, 2, 2),
                                      source_module=f'{path}.m')
                node.output = self.graph.new_tensor((c, h, w), node.name)
                cur = node.output
                ys.append(cur)
            cat_tid = self._concat(f'{path}.cat', ys, hookable=False)
            return self._walk(m.cv2, f'{path}.cv2', cat_tid)
        if kind == 'Upsample':
            c, h, w = self.graph.tensor_shapes[in_tid]
            node = self.graph.add(name=path, op='upsample_nearest_2x', op_class='PS_INT',
                                  inputs=[in_tid], output=-1, source_module=path)
            node.output = self.graph.new_tensor((c, h * int(m.scale_factor), w * int(m.scale_factor)), path)
            return node.output
        if kind == 'Detect':
            for i in range(m.nl):
                xi_tid = self.y[self.detect_from[i] + 1]
                box_tid = self._walk_seq(m.cv2[i], f'{path}.cv2.{i}', xi_tid)
                cls_tid = self._walk_seq(m.cv3[i], f'{path}.cv3.{i}', xi_tid)
                self.graph.scope_outputs.extend([box_tid, cls_tid])
            return -1
        raise ValueError(f'Unsupported module {kind} at {path}')

    def _concat(self, path, in_tids, hookable=True):
        shapes = [self.graph.tensor_shapes[t] for t in in_tids]
        if any(s[1:] != shapes[0][1:] for s in shapes):
            raise ValueError(f'{path}: concat spatial mismatch {shapes}')
        node = self.graph.add(name=path, op='concat', op_class='PS_INT', inputs=list(in_tids),
                              output=-1, source_module=path)
        node.output = self.graph.new_tensor((sum(s[0] for s in shapes),) + shapes[0][1:], path)
        if hookable and not path.endswith('.cat'):
            self.hookable_paths.add(path)
        return node.output

    def trace(self):
        self.hookable_paths = set()
        for i, layer in enumerate(self.layers):
            kind = type(layer).__name__
            f = getattr(layer, 'f', -1)
            if kind == 'Concat':
                in_tids = [(self.y[i] if j == -1 else self.y[j + 1]) for j in f]
                out = self._concat(f'model.{i}', in_tids)
            elif kind == 'Detect':
                self.detect_from = list(f)
                out = self._walk(layer, f'model.{i}', None)
            else:
                in_tid = self.y[i] if f == -1 else self.y[f + 1]
                out = self._walk(layer, f'model.{i}', in_tid)
            self.y.append(out)
        if not self.graph.scope_outputs:
            raise ValueError('No detect-head scope outputs found')
        return self.graph


def _module_refs(root, graph):
    """(path, module) for every node source that resolves to a real module, in node order."""
    refs = []
    for node in graph.nodes:
        path = node.source_module
        if not path or path.endswith('(chunk)') or path.endswith('(residual)') or path.endswith('.cat'):
            continue
        try:
            obj = root
            for part in path.split('.'):
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        except (AttributeError, IndexError):
            continue
        if isinstance(obj, torch.nn.Module):
            refs.append((path, obj))
    return refs


def capture_original_outputs(detection_model, module_refs, x):
    """One forward pass with hooks; outputs keyed by module path, in fire order."""
    captured, handles = {}, []

    def make(name):
        def hook(_module, _inp, output):
            captured.setdefault(name, []).append(output.detach().float().clone())
        return hook

    seen = set()
    for path, module in module_refs:
        if path in seen:
            continue
        seen.add(path)
        handles.append(module.register_forward_hook(make(path)))
    with torch.no_grad():
        detection_model(x)
    for h in handles:
        h.remove()
    return captured


def run_float_executor(graph, x):
    """Execute the traced graph with folded float weights; returns {node_name: tensor}."""
    cache = {0: x}
    outputs = {}
    for node in graph.nodes:
        ins = [cache[t] for t in node.inputs]
        if node.op.startswith('conv'):
            w, b = graph.weights[node.name]
            pt, pb, pl, pr = node.pads
            y = F.conv2d(F.pad(ins[0], (pl, pr, pt, pb)), w, b, stride=node.stride)
            if node.act == 'silu':
                y = F.silu(y)
            elif node.act == 'relu':
                y = F.relu(y)
            cache[node.output] = y
            outputs[node.name] = y
        elif node.op == 'maxpool_5x5':
            # Implicit -inf padding (standard max-pool semantics), matching the framework's
            # MaxPool2d(padding=2); zero-padding would corrupt all-negative border windows.
            y = F.max_pool2d(ins[0], node.kh, stride=node.stride, padding=2)
            cache[node.output] = y
            outputs[node.name] = y
        elif node.op == 'upsample_nearest_2x':
            y = F.interpolate(ins[0], scale_factor=2.0, mode='nearest')
            cache[node.output] = y
            outputs[node.name] = y
        elif node.op == 'concat':
            y = torch.cat(ins, 1)
            cache[node.output] = y
            outputs[node.name] = y
        elif node.op == 'split_chunk':
            a, b = torch.chunk(ins[0], 2, 1)
            cache[node.output] = a
            cache[node.output2] = b
            outputs[node.name] = a
        elif node.op == 'add':
            y = ins[0] + ins[1]
            cache[node.output] = y
            outputs[node.name] = y
        else:
            raise ValueError(f'Unknown op {node.op}')
    return outputs, cache


def verify_against_original(detection_model, graph, x, tolerance=1e-4):
    """Fold+trace verification: executor node outputs vs original module hook outputs."""
    refs = _module_refs(detection_model, graph)
    captured = capture_original_outputs(detection_model, refs, x)
    outputs, _ = run_float_executor(graph, x)
    rows, worst, compared = [], 0.0, 0
    for node in graph.nodes:
        ref_list = captured.get(node.source_module)
        if not ref_list:
            continue
        ref = ref_list.pop(0)
        got = outputs[node.name]
        diff = float((ref - got).abs().max())
        worst = max(worst, diff)
        compared += 1
        rows.append({'node': node.name, 'max_abs_diff': diff})
    if compared == 0:
        raise AssertionError('No hook comparisons matched any node')
    if worst > tolerance:
        raise AssertionError(f'Fold/trace verification failed: worst diff {worst} > {tolerance}')
    return {'max_abs_diff': worst, 'tolerance': tolerance, 'compared_nodes': compared,
            'nodes_total': len(graph.nodes), 'rows': rows,
            'note': 'chunk/add/internal-concat nodes have no source module and are verified transitively via every hooked Conv output.'}


def op_audit(graph):
    counts, problems = {}, []
    for node in graph.nodes:
        counts[node.op] = counts.get(node.op, 0) + 1
        if node.op not in OPS_PL_V1 and node.op not in OPS_PS_INT:
            problems.append(f'{node.name}: unclassified op {node.op}')
        if node.op.startswith('conv') and node.op not in OPS_PL_V1:
            problems.append(f'{node.name}: conv op {node.op} not in PL_V1')
    silu_layers = [n.name for n in graph.nodes if n.act == 'silu']
    scope_nodes = [n.name for n in graph.nodes if n.output in set(graph.scope_outputs)]
    return {'op_counts': counts, 'nodes_total': len(graph.nodes),
            'conv_nodes': sum(1 for n in graph.nodes if n.op.startswith('conv')),
            'silu_activations': len(silu_layers),
            'student_model_replace': {'op': 'silu', 'count': len(silu_layers), 'examples': silu_layers[:5]},
            'scope_output_nodes': scope_nodes, 'problems': problems}


def load_detection_model(weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = checkpoint['model'].float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_audit(weights_path, output_dir, seed=20260928):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    model = load_detection_model(weights_path)
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, 128, 128, generator=generator)
    tracer = Tracer(model.model)
    graph = tracer.trace()
    verification = verify_against_original(model, graph, x)
    audit = op_audit(graph)
    if audit['problems']:
        raise AssertionError(f'Op audit problems: {audit["problems"]}')
    weight_rows = []
    for name, (w, _b) in graph.weights.items():
        qw, sw = quantize_weights(w)
        deq = qw.astype(np.float64) * sw.astype(np.float64)[:, None, None, None]
        weight_rows.append({'node': name, 'weight_l2': float(np.linalg.norm((w.numpy() - deq).ravel())),
                            'weight_max_abs': float(np.abs(w.numpy() - deq).max())})
    report = {'stage': 'p4.1', 'status': 'complete', 'contract_version': contract['contract_version'],
              'weights_sha256': file_hash(weights_path), 'seed': seed,
              'graph': {'nodes_total': len(graph.nodes), 'tensor_count': len(graph.tensor_shapes),
                        'input_shape': list(graph.input_shape),
                        'scope_output_count': len(graph.scope_outputs),
                        'scope_output_nodes': audit['scope_output_nodes']},
              'fold_verification': {k: v for k, v in verification.items() if k != 'rows'},
              'op_audit': audit,
              'weight_quantization': {'layers': len(weight_rows),
                                      'worst_l2': max(r['weight_l2'] for r in weight_rows),
                                      'worst_max_abs': max(r['weight_max_abs'] for r in weight_rows),
                                      'examples': weight_rows[:5]},
              'limitations': ['Verification input is seeded synthetic data in [0,1]; fold error is structural, not data dependent.',
                              'Weight quantization errors are reported, not accepted: PTQ (P4.2) decides.']}
    write_json(output / 'report.json', report)
    write_json(output / 'op_audit.json', audit)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='experiments/p2_expanded_train/fit/weights/best.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    report = run_audit(args.weights, args.output, seed=args.seed)
    a = report['op_audit']
    print(f"nodes={a['nodes_total']} conv={a['conv_nodes']} silu={a['silu_activations']} "
          f"fold_worst_diff={report['fold_verification']['max_abs_diff']:.2e} "
          f"compared={report['fold_verification']['compared_nodes']} "
          f"scope_outputs={report['graph']['scope_output_count']}")


if __name__ == '__main__':
    main()
