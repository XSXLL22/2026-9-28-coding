"""Independent Python integer reference model (P4.3), executing a P4PKB01 pack bit-exactly.

Contract: p4-contract-1.3 (docs/定点格式说明.md). Rules implemented here must be
re-implemented identically in C (reference/c) and later in RTL:

  conv      acc = sum(qx*qw) + qb, int32-safe (verified at pack build);
            per output channel: y = (acc*M + half) >> shift, round-half-up,
            saturate int8; padding value 0.
  requant   same rounding on a single int8 tensor, scalar M/shift.
  silu      256-entry LUT lookup, no arithmetic.
  maxpool   integer max, border positions ignored (-inf semantics); pad value -128
            is exact for int8 because it can never win the max.
  add       int8 operands widened to int16, sum saturates to int8.
  concat/split/upsample  pure readdressing (upsample = nearest replication).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pack_reader import read_pack


def saturate_int8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def requant_channel(acc, M, shift):
    """acc int64 (cout, H, W); per-channel M/shift; round-half-up via arithmetic shift.
    shift == 0 is legal and means y = t; negative shift is a contract violation."""
    shift = np.asarray(shift, dtype=np.int64)
    if np.any(shift < 0):
        raise ValueError('negative shift')
    safe = np.maximum(shift, 1)
    half = (1 << (safe - 1)).astype(np.int64)[:, None, None] * (shift > 0)[:, None, None]
    t = acc * M[:, None, None] + half
    t = t >> np.maximum(shift, 0)[:, None, None]
    return saturate_int8(t)


def requant_scalar(x, M, shift):
    t = x.astype(np.int64) * M
    half = (1 << int(shift) - 1) if shift > 0 else 0
    return saturate_int8((t + half) >> int(shift))


def conv2d_int8_accumulate(x, node):
    """Raw int64 accumulator for every output pixel: sum(qx*qw) + qb (contract §3).

    Exact integer arithmetic: no rounding happens here, so this stage is a valid
    comparison point for the RTL accumulator (P5.0 step A).
    """
    pt, pb, pl, pr = node.pads
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)))  # contract: padding value 0
    c_in, h_in, w_in = xp.shape
    h_out = (h_in - node.kh) // node.stride + 1
    w_out = (w_in - node.kw) // node.stride + 1
    acc = np.zeros((node.cout, h_out, w_out), dtype=np.int64)
    for i in range(node.kh):
        for j in range(node.kw):
            window = xp[:, i:i + h_out * node.stride:node.stride, j:j + w_out * node.stride:node.stride]
            w = node.qw[:, :, i, j].astype(np.int64)
            acc += np.einsum('oc,cij->oij', w, window.astype(np.int64), optimize=True)
    acc += node.qb.astype(np.int64)[:, None, None]
    return acc


def conv2d_int8_requant(acc, node):
    """PL output boundary: requantize + saturate to int8, BEFORE any activation.

    This is what the P5 RTL core emits (SiLU stays in PS_INT), so P5 vectors must
    compare against this function and NOT against conv2d_int8's post-activation output.
    """
    return requant_channel(acc, node.M, node.shift)


def conv2d_int8(x, node):
    y = conv2d_int8_requant(conv2d_int8_accumulate(x, node), node)
    if node.act == 'silu':
        y = node.silu_lut[y.astype(np.int16) + 128]
    elif node.act == 'relu':
        y = np.maximum(y, 0)
    return y.astype(np.int8)


def maxpool_5x5(x, node):
    pt, pb, pl, pr = node.pads
    # -inf border semantics: int8 minimum never wins the max (contract 1.2)
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), constant_values=-128)
    h_out = (x.shape[1] + pt + pb - node.kh) // node.stride + 1
    w_out = (x.shape[2] + pl + pr - node.kw) // node.stride + 1
    out = np.full((x.shape[0], h_out, w_out), -128, dtype=np.int8)
    for i in range(node.kh):
        for j in range(node.kw):
            window = xp[:, i:i + h_out * node.stride:node.stride, j:j + w_out * node.stride:node.stride]
            out = np.maximum(out, window)
    return out


def upsample_nearest_2x(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def run_pack(pack, x_input):
    """x_input: int8 (C,H,W) already input-quantized. Returns {node_name: int8 tensor}."""
    buffers = {0: x_input}
    outputs = {}
    for node in pack.nodes:
        ins = [buffers[t] for t in node.inputs]
        if node.op.startswith('conv'):
            y = conv2d_int8(ins[0], node)
            buffers[node.output] = y
            outputs[node.name] = y
        elif node.op == 'requantize':
            y = requant_scalar(ins[0], node.M, node.shift)
            buffers[node.output] = y
            outputs[node.name] = y
        elif node.op == 'maxpool_5x5':
            y = maxpool_5x5(ins[0], node)
            buffers[node.output] = y
            outputs[node.name] = y
        elif node.op == 'upsample_nearest_2x':
            y = upsample_nearest_2x(ins[0])
            buffers[node.output] = y
            outputs[node.name] = y
        elif node.op == 'concat':
            y = np.concatenate(ins, axis=0)
            buffers[node.output] = y
            outputs[node.name] = y
        elif node.op == 'split_chunk':
            c = ins[0].shape[0] // 2
            buffers[node.output] = ins[0][:c]
            buffers[node.output2] = ins[0][c:]
            outputs[node.name] = ins[0][:c]
        elif node.op == 'add':
            y = saturate_int8(ins[0].astype(np.int16) + ins[1].astype(np.int16))
            buffers[node.output] = y
            outputs[node.name] = y
        else:
            raise ValueError(f'unsupported op {node.op}')
    return outputs, buffers


def quantize_input(image_uint8_rgb):
    """Contract §1: q = round_half_up(pixel/255 * 127), stored int8 0..127."""
    x = image_uint8_rgb.astype(np.float64) / 255.0
    return np.clip(np.floor(x * 127.0 + 0.5), 0, 127).astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', required=True)
    parser.add_argument('--input', required=True, help='.npy uint8/float image (C,H,W) or int8 quantized tensor')
    parser.add_argument('--output', required=True, help='output .npz for scope outputs')
    parser.add_argument('--dump-dir', default=None, help='optional: dump every node output as .npy')
    args = parser.parse_args()
    pack = read_pack(args.pack)
    raw = np.load(args.input)
    raw = np.asarray(raw)
    if raw.dtype == np.int8:
        x = raw
    elif raw.dtype == np.uint8:
        x = quantize_input(raw)
    else:
        x = quantize_input(np.clip(raw * 255.0, 0, 255).astype(np.uint8))
    if tuple(x.shape) != pack.tensor_shapes[0]:
        raise ValueError(f'input shape {x.shape} != pack input {pack.tensor_shapes[0]}')
    outputs, _ = run_pack(pack, x)
    scope_names = [n.name for n in pack.nodes if n.output in set(pack.meta['graph']['scope_outputs'])]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{name: outputs[name] for name in scope_names})
    if args.dump_dir:
        dump = Path(args.dump_dir)
        dump.mkdir(parents=True, exist_ok=True)
        for name, tensor in outputs.items():
            np.save(dump / f'{name}.npy', tensor)
    print(json.dumps({'scope_outputs': scope_names,
                      'shapes': [list(outputs[n].shape) for n in scope_names]}))


if __name__ == '__main__':
    main()
