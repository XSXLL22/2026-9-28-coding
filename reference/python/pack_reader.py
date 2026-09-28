"""P4PKB01 pack reader for the independent integer reference implementations.

Shared file-format contract only; no code shared with training/. All integers
little-endian. Layout (docs/定点格式说明.md §8 and model_pack.json):

  magic(8) | u32 num_nodes | node records (variable) | u32 num_tensors |
  tensor shapes (3*u32 each) | u32 meta_len | meta JSON | u32 body_len | body

Node record:
  u8 op_code, u8 act_code, u8 class_code, u8 flags |
  u32 cin, cout, kh, kw, stride | i32 pad_t, pad_b, pad_l, pad_r |
  u32 num_inputs, i32 inputs[num_inputs] | i32 output, i32 output2 |
  u32 param_offset, u32 param_len (relative to body start) |
  u32 name_len, name bytes

Conv param blob: int8 qw[cout*cin*kh*kw] | i32 qb[cout] | i32 M[cout] |
i32 shift[cout] | (u8 silu_lut[256] when act=silu).
Requantize param blob: i32 M, i32 shift.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b'P4PKB01\n'
OP_NAMES = {0: 'conv3x3_s1', 1: 'conv3x3_s2', 2: 'conv1x1_s1', 3: 'conv1x1_s2',
            4: 'maxpool_5x5', 5: 'upsample_nearest_2x', 6: 'concat', 7: 'split_chunk',
            8: 'add', 9: 'relu', 10: 'requantize'}
ACT_NAMES = {0: 'none', 1: 'relu', 2: 'silu'}
CLASS_NAMES = {0: 'PL_V1', 1: 'PS_INT', 2: 'PS_FLOAT'}


@dataclass
class PackNode:
    name: str
    op: str
    act: str
    op_class: str
    cin: int = 0
    cout: int = 0
    kh: int = 0
    kw: int = 0
    stride: int = 1
    pads: tuple = (0, 0, 0, 0)
    inputs: list = field(default_factory=list)
    output: int = -1
    output2: int = -1
    qw: object = None      # int8 (cout, cin, kh, kw), conv only
    qb: object = None      # int32 (cout,)
    M: object = None       # int64 (cout,) conv; scalar for requantize
    shift: object = None   # int64 (cout,) conv; scalar for requantize
    silu_lut: object = None  # int8 (256,)


@dataclass
class Pack:
    nodes: list = field(default_factory=list)
    tensor_shapes: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def node(self, name):
        return next(n for n in self.nodes if n.name == name)


def read_pack(path):
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError('bad pack magic')
    pos = 8
    (num_nodes,) = struct.unpack_from('<I', data, pos)
    pos += 4
    nodes = []
    for _ in range(num_nodes):
        op_code, act_code, class_code, _flags = struct.unpack_from('<4B', data, pos)
        pos += 4
        cin, cout, kh, kw, stride = struct.unpack_from('<5I', data, pos)
        pos += 20
        pt, pb, pl, pr = struct.unpack_from('<4i', data, pos)
        pos += 16
        (num_inputs,) = struct.unpack_from('<I', data, pos)
        pos += 4
        inputs = list(struct.unpack_from(f'<{num_inputs}i', data, pos))
        pos += 4 * num_inputs
        output, output2 = struct.unpack_from('<ii', data, pos)
        pos += 8
        param_offset, param_len = struct.unpack_from('<II', data, pos)
        pos += 8
        (name_len,) = struct.unpack_from('<I', data, pos)
        pos += 4
        name = data[pos:pos + name_len].decode('utf-8')
        pos += name_len
        nodes.append({'name': name, 'op': OP_NAMES[op_code], 'act': ACT_NAMES[act_code],
                      'class': CLASS_NAMES[class_code], 'cin': cin, 'cout': cout, 'kh': kh,
                      'kw': kw, 'stride': stride, 'pads': (pt, pb, pl, pr), 'inputs': inputs,
                      'output': output, 'output2': output2,
                      'param': (param_offset, param_len)})
    (num_tensors,) = struct.unpack_from('<I', data, pos)
    pos += 4
    shapes = {}
    for tid in range(num_tensors):
        c, h, w = struct.unpack_from('<3I', data, pos)
        pos += 12
        shapes[tid] = (c, h, w)
    (meta_len,) = struct.unpack_from('<I', data, pos)
    pos += 4
    meta = json.loads(data[pos:pos + meta_len].decode('utf-8'))
    pos += meta_len
    (body_len,) = struct.unpack_from('<I', data, pos)
    pos += 4
    body = data[pos:pos + body_len]
    if body_len != len(body):
        raise ValueError('pack truncated')

    pack_nodes = []
    for rec in nodes:
        node = PackNode(name=rec['name'], op=rec['op'], act=rec['act'], op_class=rec['class'],
                        cin=rec['cin'], cout=rec['cout'], kh=rec['kh'], kw=rec['kw'],
                        stride=rec['stride'], pads=rec['pads'], inputs=rec['inputs'],
                        output=rec['output'], output2=rec['output2'])
        off, ln = rec['param']
        if ln:
            blob = body[off:off + ln]
            if node.op.startswith('conv'):
                n_w = node.cout * node.cin * node.kh * node.kw
                qw = np.frombuffer(blob, dtype=np.int8, count=n_w, offset=0)
                cursor = n_w
                node.qw = qw.reshape(node.cout, node.cin, node.kh, node.kw)
                node.qb = np.frombuffer(blob, dtype='<i4', count=node.cout, offset=cursor)
                cursor += 4 * node.cout
                node.M = np.frombuffer(blob, dtype='<i4', count=node.cout, offset=cursor).astype(np.int64)
                cursor += 4 * node.cout
                node.shift = np.frombuffer(blob, dtype='<i4', count=node.cout, offset=cursor).astype(np.int64)
                cursor += 4 * node.cout
                if node.act == 'silu':
                    node.silu_lut = np.frombuffer(blob, dtype=np.uint8, count=256, offset=cursor).astype(np.int8)
            elif node.op == 'requantize':
                m, s = struct.unpack_from('<ii', blob, 0)
                node.M, node.shift = np.int64(m), np.int64(s)
        pack_nodes.append(node)
    return Pack(nodes=pack_nodes, tensor_shapes=shapes, meta=meta)
