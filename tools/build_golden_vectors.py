"""P4.5 golden vector generator: inputs + per-node expected outputs from the Python reference.

Vector kinds (阶段任务清单 P4): normal, boundary, negative, saturation, tail-channel, random,
plus real-image end-to-end vectors. The real pack exercises everything reachable with legal
inputs; a hand-built mini-pack (cin=5, no power-of-two channels) exercises the tail-channel
rule, and full-range signed inputs exercise negative/rounding/saturation paths. Expected
outputs come ONLY from reference/python/int_reference.py; C must reproduce them bit-exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import quantize_input, run_pack, saturate_int8  # noqa: E402
from pack_reader import read_pack  # noqa: E402

from runtime.common import file_hash, write_json  # noqa: E402

CONTRACT_VERSION = 'p4-contract-1.2'
DUMP_OPS = {'conv3x3_s1', 'conv3x3_s2', 'conv1x1_s1', 'conv1x1_s2', 'requantize', 'add', 'maxpool_5x5'}
GENERATOR = 'tools/build_golden_vectors.py'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_outputs(outputs, pack, out_dir):
    files = {}
    for node in pack.nodes:
        if node.op in DUMP_OPS:
            raw = outputs[node.name].astype(np.int8).tobytes()
            rel = f'expected/{node.name}.bin'
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / rel).write_bytes(raw)
            files[node.name] = {'path': rel, 'shape': list(outputs[node.name].shape), 'sha256': sha(raw)}
    return files


def real_image_vectors(count, seed):
    """Frozen selection from the 109-image development set (seeded, documented)."""
    val_dir = ROOT / 'datasets' / 'public_expanded_comparison' / 'images' / 'val'
    files = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    rng = np.random.RandomState(seed)
    chosen = sorted(rng.choice(len(files), count, replace=False).tolist())
    return [(files[i], files[i].stem) for i in chosen]


def letterbox_128_u8(bgr):
    import cv2
    h, w = bgr.shape[:2]
    scale = min(128 / h, 128 / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((128, 128, 3), 114, dtype=np.uint8)
    top, left = (128 - nh) // 2, (128 - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # CHW RGB uint8


def build_mini_pack(out_dir):
    """Hand-built tail-channel pack: conv1x1(cin=5, cout=3, silu) on a 4x4 input."""
    cin, cout, spatial = 5, 3, 4
    rng = np.random.RandomState(20260928)
    qw = saturate_int8(np.floor(rng.randn(cout, cin, 1, 1) * 20 + 0.5)).astype(np.int8)
    sw = np.abs(qw).max(axis=(1, 2, 3)).astype(np.float64) / 127.0
    sw = np.where(sw == 0, 1e-12, sw)
    qw = saturate_int8(np.floor(qw.astype(np.float64) / sw[:, None, None, None] + 0.5))
    b = np.round(rng.randn(cout) * 0.3).astype(np.int64)
    sx, sy = 1.0 / 127.0, 0.02
    qb = np.floor(b / (sx * sw) + 0.5).astype(np.int64)
    r = sx * sw / sy
    n = np.ceil(np.log2(2 ** 30 / r)).astype(np.int64)
    M = np.floor(r * 2.0 ** n + 0.5).astype(np.int64)
    v = np.arange(-128, 129, dtype=np.float64) * sy
    lut = np.clip(np.floor((v / (1 + np.exp(-v))) / sy + 0.5), -128, 127).astype(np.uint8)

    nodes = [{'name': 'mini.conv', 'op': 2, 'act': 2, 'cls': 0, 'cin': cin, 'cout': cout,
              'kh': 1, 'kw': 1, 'stride': 1, 'pads': (0, 0, 0, 0), 'inputs': [0],
              'output': 1, 'output2': -1,
              'param': qw.tobytes() + qb.astype('<i4').tobytes() + M.astype('<i4').tobytes()
                       + n.astype('<i4').tobytes() + lut.tobytes()}]
    tensors = [(cin, spatial, spatial), (cout, spatial, spatial)]
    body = nodes[0]['param']
    out = bytearray()
    out += b'P4PKB01\n'
    out += struct.pack('<I', 1)
    name_b = nodes[0]['name'].encode()
    rec = struct.pack('<4B', nodes[0]['op'], nodes[0]['act'], nodes[0]['cls'], 0)
    rec += struct.pack('<5I', cin, cout, 1, 1, 1)
    rec += struct.pack('<4i', 0, 0, 0, 0)
    rec += struct.pack('<I', 1) + struct.pack('<i', 0)
    rec += struct.pack('<ii', 1, -1)
    rec += struct.pack('<II', 0, len(body))
    rec += struct.pack('<I', len(name_b)) + name_b
    out += rec
    out += struct.pack('<I', len(tensors))
    for t in tensors:
        out += struct.pack('<3I', *t)
    meta = json.dumps({'model_id': 'p4-mini-tail', 'format_version': 'p4pkb-1',
                       'contract_version': CONTRACT_VERSION, 'note': 'synthetic tail-channel vector pack'})
    meta_b = meta.encode('utf-8')
    out += struct.pack('<I', len(meta_b)) + meta_b
    out += struct.pack('<I', len(body)) + body
    path = out_dir / 'mini_pack.bin'
    path.write_bytes(bytes(out))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', default='training/export/p4_baseline128_v1/model_pack.bin')
    parser.add_argument('--output', default='tests/golden')
    parser.add_argument('--real-images', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    out_root = Path(args.output)
    if out_root.exists():
        raise FileExistsError(out_root)
    out_root.mkdir(parents=True)
    pack = read_pack(args.pack)
    input_shape = pack.tensor_shapes[0]
    rng = np.random.RandomState(args.seed)
    manifest = {'contract_version': CONTRACT_VERSION, 'pack': args.pack,
                'pack_sha256': file_hash(args.pack), 'seed': args.seed,
                'generator': GENERATOR, 'generator_sha256': file_hash(ROOT / GENERATOR),
                'expected_source': 'reference/python/int_reference.py (P4.3)',
                'comparison': 'reference/c/int_reference must reproduce every expected/*.bin byte-for-byte',
                'vectors': {}}

    def emit(name, kind, x_int8, note):
        vdir = out_root / name
        (vdir / 'expected').mkdir(parents=True, exist_ok=True)
        raw = x_int8.astype(np.int8).tobytes()
        (vdir / 'input.bin').write_bytes(raw)
        outputs, _ = run_pack(pack, x_int8)
        files = dump_outputs(outputs, pack, vdir)
        manifest['vectors'][name] = {'kind': kind, 'note': note,
                                     'input': {'shape': list(x_int8.shape), 'sha256': sha(raw)},
                                     'expected': files}
        print(f'{name}: {len(files)} expected tensors')

    c, h, w = input_shape
    # 1 normal: legal-domain smooth-ish random (pixels only use 0..127 per input contract)
    emit('v01_normal_random', 'normal', rng.randint(0, 128, size=input_shape).astype(np.int8),
         'contract-legal input domain 0..127, uniform random')
    # 2 boundary: extreme corners of the legal domain and the full int8 domain
    b = np.zeros(input_shape, dtype=np.int8)
    b[:, : h // 2, : w // 2] = 127
    b[:, h // 2:, w // 2:] = -128
    emit('v02_boundary_extremes', 'boundary', b,
         'quadrants at int8 extremes; 127 is the legal input maximum, -128 is synthetic')
    # 3 negative: full signed range (synthetic: violates the image-domain contract on purpose)
    emit('v03_negative_fullrange', 'negative', rng.randint(-128, 128, size=input_shape).astype(np.int8),
         'full int8 range; exercises negative accumulator paths and negative SiLU LUT entries')
    # 4 saturation: all-max input drives every layer toward int8 clipping
    emit('v04_saturation_max', 'saturation', np.full(input_shape, 127, dtype=np.int8),
         'saturates int8 outputs through the depth of the network; int32 accumulation is '
         'guarded by the per-layer bound check (max |qx|=127 worst case), not by this vector')
    emit('v05_saturation_min', 'saturation', np.full(input_shape, -128, dtype=np.int8),
         'all-minimum signed input; maximum-magnitude negative accumulators and rounding')
    # 5 random (second seed)
    emit('v06_random_seed_b', 'random', rng.randint(-128, 128, size=input_shape).astype(np.int8),
         'full-range random, different stream from v03')
    # 6 tail-channel mini pack
    mini_path = build_mini_pack(out_root)
    mini = read_pack(mini_path)
    mini_x = rng.randint(-128, 128, size=mini.tensor_shapes[0]).astype(np.int8)
    vdir = out_root / 'v07_tail_channel'
    (vdir / 'expected').mkdir(parents=True, exist_ok=True)
    raw = mini_x.tobytes()
    (vdir / 'input.bin').write_bytes(raw)
    mini_out, _ = run_pack(mini, mini_x)
    files = {}
    for node in mini.nodes:
        blob = mini_out[node.name].astype(np.int8).tobytes()
        rel = f'expected/{node.name}.bin'
        (vdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (vdir / rel).write_bytes(blob)
        files[node.name] = {'path': rel, 'shape': list(mini_out[node.name].shape), 'sha256': sha(blob)}
    (vdir / 'pack.bin').write_bytes(mini_path.read_bytes())
    manifest['vectors']['v07_tail_channel'] = {
        'kind': 'tail_channel', 'note': 'synthetic mini-pack with cin=5 exercises non-power-of-two '
                                       'channel handling; C uses this vector pack instead of the main pack',
        'pack': 'pack.bin', 'pack_sha256': file_hash(mini_path),
        'input': {'shape': list(mini_x.shape), 'sha256': sha(raw)}, 'expected': files}
    print(f"v07_tail_channel: {len(files)} expected tensors")
    # 7 real images
    for i, (path, stem) in enumerate(real_image_vectors(args.real_images, args.seed)):
        import cv2
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'cannot decode {path}')
        u8 = letterbox_128_u8(bgr)
        emit(f'v{i + 8:02d}_real_{stem}', 'real_image', quantize_input(u8),
             f'frozen dev image {path.name}, letterboxed 128, input-quantized per contract')

    write_json(out_root / 'manifest.json', manifest)
    print(f'manifest: {out_root / "manifest.json"} ({len(manifest["vectors"])} vectors)')


if __name__ == '__main__':
    main()
