"""P5.0 vector generator for the minimal 3x3 convolution RTL.

Expectations come ONLY from the Python integer reference's PRE-SiLU boundary
(`conv2d_int8_requant` over `conv2d_int8_accumulate`), i.e. exactly the PL boundary defined
in hardware/rtl/INTERFACE.md. Nothing here consults the RTL: vectors are generated before
simulation and must never be regenerated to match RTL output (that would invert the
verification). The generator is deterministic: same seed -> byte-identical vectors.

Crafted vectors pin contract behaviour that a random stream would rarely hit, and their
expected values are derived by hand from the contract tables:

  syn_round_table   M=2^30, shift=32 makes y = floor(acc/4 + 0.5), i.e. the contract's
                    round-half-up table for n=2 (docs/定点格式说明.md §4):
                    acc  9 -> 2, 7 -> 2, -5 -> -1, -6 -> -1, -7 -> -2
  syn_sat_half      same M/shift; acc +-762 saturate, acc +-254 are exact half-way cases
  syn_shift0_sat    shift=0 is legal: y = t, so acc 1/0/-1 -> 127/0/-128
  syn_neg128        weights and inputs at -128 exercise the int8 abs(-128) widening path
  syn_rnd_*         asymmetric shapes/random values so channel-stride and row/column index
                    errors cannot be masked by uniform data

A real-layer vector is emitted from the actual pack (real intermediate int8 activation of a
3x3/stride=1/pad=1 layer), which is the strongest index/scale check available without
hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import (  # noqa: E402
    conv2d_int8_accumulate,
    conv2d_int8_requant,
    quantize_input,
    run_pack,
)
from pack_reader import PackNode, read_pack  # noqa: E402

from runtime.common import file_hash, write_json  # noqa: E402
from tools.build_golden_vectors import letterbox_128_u8, real_image_vectors  # noqa: E402

VECTOR_VERSION = 'p5v-1'
GENERATOR = 'tools/build_p5_vectors.py'
M_MIN, M_MAX = 2 ** 30, 2 ** 31
MAX_SHIFT = 62


def check_contract(M, n, where):
    """Contract v1.3 §4 legality, asserted independently of whoever produced (M, n)."""
    M = np.asarray(M, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    if not np.all((M >= M_MIN) & (M < M_MAX)):
        raise ValueError(f'{where}: M outside [2^30, 2^31): {M.tolist()}')
    if not np.all((n >= 0) & (n <= MAX_SHIFT)):
        raise ValueError(f'{where}: shift outside [0, {MAX_SHIFT}]: {n.tolist()}')


def hex_bytes(arr):
    return ''.join(f'{int(v) & 0xFF:02x}\n' for v in np.asarray(arr).ravel())


def hex_words(arr):
    return ''.join(f'{int(v) & 0xFFFFFFFF:08x}\n' for v in np.asarray(arr).ravel())


def weight_stream(qw):
    """OIHW flattening used by the RTL: index o*cin*9 + ci*9 + i*3 + j.

    C-order flattening of the (cout, cin, 3, 3) array gives exactly that ordering, so no
    transpose is involved; the function exists to name the convention.
    """
    return np.asarray(qw).reshape(-1)


def centre_tap(qw):
    """Zero 3x3 kernel with the given values on the centre tap (i=j=1).

    With pad=1, stride=1 and a 1x1 input, only the centre tap is in bounds, so output
    channel o sees exactly qw[o, ci, 1, 1] as its weight -- this is what makes the crafted
    vectors hand-computable.
    """
    v = np.asarray(qw, dtype=np.int8)
    out = np.zeros((v.shape[0], v.shape[1], 3, 3), dtype=np.int8)
    out[:, :, 1, 1] = v.reshape(v.shape[0], v.shape[1])
    return out


def make_node(cin, cout, qw, qb, M, shift, name='vector'):
    return PackNode(name=name, op='conv3x3_s1', act='none', op_class='PL_V1',
                    cin=cin, cout=cout, kh=3, kw=3, stride=1, pads=(1, 1, 1, 1),
                    inputs=[0], output=1,
                    qw=qw.reshape(cout, cin, 3, 3), qb=np.asarray(qb, dtype=np.int64),
                    M=np.asarray(M, dtype=np.int64), shift=np.asarray(shift, dtype=np.int64))


def emit(out_root, name, kind, x, node, note, seed, source=None, manifest=None):
    x = np.asarray(x, dtype=np.int8)
    cin, h, w = x.shape
    qw = node.qw.astype(np.int8)
    qb = node.qb.astype(np.int64)
    M = node.M.astype(np.int64)
    n = node.shift.astype(np.int64)
    check_contract(M, n, name)

    acc = conv2d_int8_accumulate(x, node)
    expected = conv2d_int8_requant(acc, node)

    vdir = out_root / name
    if vdir.exists():
        raise FileExistsError(vdir)
    vdir.mkdir(parents=True)

    files = {
        'input.mem': hex_bytes(x),
        'weight.mem': hex_bytes(weight_stream(qw)),
        'param.mem': hex_words(np.stack([qb, M, n], axis=1).reshape(-1)),
        'expected.mem': hex_bytes(expected),
    }
    hashes = {}
    for fname, text in files.items():
        (vdir / fname).write_text(text, encoding='utf-8', newline='\n')
        hashes[fname] = file_hash(vdir / fname)

    info = {
        'vector_version': VECTOR_VERSION,
        'name': name,
        'kind': kind,
        'note': note,
        'seed': seed,
        'source': source or {'origin': 'synthetic', 'generator': GENERATOR},
        'layout': {'input': 'CHW', 'weight': 'OIHW', 'output': 'CHW'},
        'geometry': {'cin': cin, 'cout': node.cout, 'h': h, 'w': w,
                     'kernel': 3, 'stride': 1, 'pad': 1,
                     'h_out': h, 'w_out': w, 'output_count': int(node.cout * h * w)},
        'accumulator_expected': {
            'min': int(acc.min()), 'max': int(acc.max()),
            'note': 'int64 sum(qx*qw)+qb; the RTL accumulator is internal and is not dumped, '
                    'but crafted vectors pin it through the requantization',
        },
        'multiplier_shift': {'M': M.tolist(), 'shift': n.tolist()},
        'output_expected': {'count': int(expected.size), 'sha256': hashes['expected.mem'],
                            'note': 'pre-SiLU requantized int8 (PL boundary), CHW'},
        'files': hashes,
        'boundary': 'pre_silu_requantized_int8',
    }
    write_json(vdir / 'vector.json', info)
    if manifest is not None:
        manifest['vectors'][name] = {
            'kind': kind, 'note': note, 'seed': seed,
            'geometry': info['geometry'], 'source': info['source'],
            'output_expected': info['output_expected'], 'files': hashes,
        }
    print(f'{name}: cin={cin} cout={node.cout} {h}x{w} -> {expected.size} outputs')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', default='training/export/p4_baseline128_v1/model_pack.bin')
    parser.add_argument('--output', default='experiments/p5_minimal_conv_v1/vectors')
    parser.add_argument('--layer', default='model.6.m.0.cv1')
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    out_root = Path(args.output)
    if out_root.exists():
        raise FileExistsError(f'{out_root} exists; P5 vectors are frozen, use a new directory')
    out_root.mkdir(parents=True)

    manifest = {'vector_version': VECTOR_VERSION, 'pack': args.pack,
                'pack_sha256': file_hash(ROOT / args.pack), 'seed': args.seed,
                'generator': GENERATOR, 'generator_sha256': file_hash(ROOT / GENERATOR),
                'expected_source': 'reference/python/int_reference.py '
                                   'conv2d_int8_requant(conv2d_int8_accumulate(x, node)) '
                                   '-- the PL boundary, SiLU NOT applied',
                'comparison': 'RTL output must equal expected.mem byte-for-byte; '
                              'vectors are generated before simulation and never regenerated '
                              'to match RTL results',
                'vectors': {}}

    rng = np.random.RandomState(args.seed)

    # ---- crafted: contract rounding table, n_eff = 2 (M=2^30, shift=32)
    qw = centre_tap(np.array([9, 7, -5, -6, -7], dtype=np.int8).reshape(5, 1, 1, 1))
    node = make_node(1, 5, qw, np.zeros(5), np.full(5, 2 ** 30), np.full(5, 32))
    emit(out_root, 'syn_round_table', 'crafted', np.array([[[1]]], dtype=np.int8), node,
         'unit input, centre weights = accumulator; M=2^30/shift=32 -> floor(acc/4+0.5); '
         'expects the contract table 9->2, 7->2, -5->-1, -6->-1, -7->-2', args.seed,
         manifest=manifest)

    # ---- crafted: saturation at both ends and exact +-0.5 half-way cases
    x = np.full((3, 1, 1), 127, dtype=np.int8)
    qw = centre_tap(np.array([[[2], [2], [2]], [[-2], [-2], [-2]],
                              [[1], [1], [0]], [[-1], [-1], [0]]], dtype=np.int8))
    node = make_node(3, 4, qw, np.zeros(4), np.full(4, 2 ** 30), np.full(4, 32))
    emit(out_root, 'syn_sat_half', 'crafted', x, node,
         'acc = +762/-762 (saturate) and +254/-254 (exact .5 rounding); expects '
         '127, -128, 64, -63', args.seed, manifest=manifest)

    # ---- crafted: shift = 0 is legal, y = t, everything saturates immediately
    qw = centre_tap(np.array([1, 0, -1], dtype=np.int8).reshape(3, 1, 1, 1))
    node = make_node(1, 3, qw, np.zeros(3), np.full(3, 2 ** 30), np.zeros(3))
    emit(out_root, 'syn_shift0_sat', 'crafted', np.array([[[1]]], dtype=np.int8), node,
         'shift=0: y = acc*M with half forced to 0; acc 1/0/-1 -> 127/0/-128', args.seed,
         manifest=manifest)

    # ---- crafted: int8 abs(-128) widening path (weights and inputs at -128)
    x = np.full((2, 3, 3), -128, dtype=np.int8)
    qw = np.zeros((1, 2, 3, 3), dtype=np.int8)
    qw[0, 0, 1, 1] = -128          # centre tap of channel 0 only
    node = make_node(2, 1, qw, np.zeros(1), np.array([2 ** 30]), np.array([38]))
    # acc = (-128)*(-128) = 16384; shift=38 with M=2^30 gives floor(acc/256 + 0.5) = 64.
    # A sign or widening error would land on -64 instead, so the value is observable
    # (an identity requant would have saturated and hidden the difference).
    emit(out_root, 'syn_neg128', 'crafted', x, node,
         'product (-128)*(-128)=16384 at the centre tap, observed as floor(16384/256+0.5)=64 '
         '-- not saturated, so a sign/abs(-128) error shows up as -64', args.seed,
         manifest=manifest)

    # ---- random, asymmetric shapes (index coverage)
    for name, cin, cout, h, w in [('syn_rnd_3x3_c1_o1', 1, 1, 3, 3),
                                  ('syn_rnd_5x7_c3_o2', 3, 2, 5, 7),
                                  ('syn_rnd_7x3_c5_o3', 5, 3, 7, 3)]:
        qw = rng.randint(-128, 128, size=(cout, cin, 3, 3)).astype(np.int8)
        qb = rng.randint(-1000, 1001, size=cout).astype(np.int64)
        # legal (M, n) per channel, derived the same way the contract prescribes. No clipping:
        # check_contract below must pass on the derivation itself, not on a masked value.
        real_scale = rng.uniform(0.02, 0.6, size=cout)
        n = np.ceil(np.log2(M_MIN / real_scale)).astype(np.int64)
        M = np.floor(real_scale * 2.0 ** n + 0.5).astype(np.int64)
        check_contract(M, n, name)
        node = make_node(cin, cout, qw, qb, M, n)
        emit(out_root, name, 'synthetic_random', rng.randint(-128, 128, size=(cin, h, w)).astype(np.int8),
             node, f'random asymmetric {cin}x{cout} {h}x{w}, full int8 domain', args.seed, manifest=manifest)

    # ---- degenerate: all-zero weights and bias (also used as task B in the two-task test)
    node = make_node(2, 2, np.zeros((2, 2, 3, 3), dtype=np.int8), np.zeros(2),
                     np.full(2, 2 ** 30), np.full(2, 30))
    emit(out_root, 'syn_zero', 'degenerate', np.zeros((2, 4, 4), dtype=np.int8), node,
         'all-zero weights/bias: every output must be 0; catches stale-state leakage between '
         'consecutive tasks', args.seed, manifest=manifest)

    # ---- real layer from the actual pack
    pack = read_pack(ROOT / args.pack)
    layer = pack.node(args.layer)
    if not (layer.kh == 3 and layer.kw == 3 and layer.stride == 1 and tuple(layer.pads) == (1, 1, 1, 1)):
        raise ValueError(f'{args.layer} is not a 3x3/stride=1/pad=1 layer')
    tid = layer.inputs[0]
    image_path, stem = real_image_vectors(1, args.seed)[0]
    import cv2
    bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'cannot decode {image_path}')
    x_img = quantize_input(letterbox_128_u8(bgr))
    _, buffers = run_pack(pack, x_img)
    x_layer = buffers[tid]
    if tuple(x_layer.shape) != tuple(pack.tensor_shapes[tid]):
        raise ValueError('layer input shape mismatch')
    emit(out_root, f'real_{args.layer}', 'real_layer', x_layer, layer,
         f'real intermediate int8 activation feeding {args.layer} for frozen dev image '
         f'{image_path.name}; weights/bias/M/shift are the pack\'s own', args.seed,
         source={'origin': 'pack_layer', 'pack': args.pack,
                 'pack_sha256': file_hash(ROOT / args.pack), 'layer': args.layer,
                 'layer_input_shape': list(x_layer.shape), 'image': image_path.name,
                 'image_stem': stem},
         manifest=manifest)

    write_json(out_root / 'manifest.json', manifest)
    print(f'manifest: {out_root / "manifest.json"} ({len(manifest["vectors"])} vectors)')


if __name__ == '__main__':
    main()
