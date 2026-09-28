"""P5.0 step-A evidence: accumulator overflow headroom for every frozen P5 vector.

Contract v1.3 §6 bounds each output channel's accumulator before requantization by

    B[o] = |qb[o]| + 128 * sum_ci,i,j |qw[o,ci,i,j]|

(the 128 is the largest magnitude an int8 activation can take). This script recomputes B
straight from the frozen weight/bias files -- not from anything the generator recorded --
and reports the observed accumulator range from the Python integer reference next to it, so
the "no accumulator overflow" claim has a checked, reproducible basis rather than an
assertion.

It fails (non-zero exit) if a vector's observed accumulator exceeds its own bound, or if a
bound does not fit in int32, since the PL datapath and the C/RTL references all rely on
that headroom.

Usage:
    python tools/check_p5_accum_bound.py [--vectors DIR] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import conv2d_int8_accumulate  # noqa: E402
from pack_reader import PackNode  # noqa: E402
from runtime.common import file_hash, write_json  # noqa: E402

INT32_MAX = 2 ** 31 - 1
MAX_ABS_INPUT = 128     # contract v1.3 §6


def read_hex_bytes(path):
    return np.array([int(line, 16) for line in
                     path.read_text(encoding='utf-8').split()], dtype=np.int64)


def read_hex_words(path):
    return np.array([int(line, 16) for line in
                     path.read_text(encoding='utf-8').split()], dtype=np.int64)


def load_vector(vdir):
    info = json.loads((vdir / 'vector.json').read_text(encoding='utf-8'))
    g = info['geometry']
    cin, cout, h, w = g['cin'], g['cout'], g['h'], g['w']
    x = read_hex_bytes(vdir / 'input.mem')
    # stored as int8 two's complement
    x = np.where(x >= 128, x - 256, x).astype(np.int8).reshape(cin, h, w)
    qw = read_hex_bytes(vdir / 'weight.mem')
    qw = np.where(qw >= 128, qw - 256, qw).astype(np.int8).reshape(cout, cin, 3, 3)
    words = read_hex_words(vdir / 'param.mem').reshape(cout, 3)
    qb = np.where(words[:, 0] >= 2 ** 31, words[:, 0] - 2 ** 32, words[:, 0]).astype(np.int64)
    M, shift = words[:, 1], words[:, 2]
    node = PackNode(name=info['name'], op='conv3x3_s1', act='none', op_class='PL_V1',
                    cin=cin, cout=cout, kh=3, kw=3, stride=1, pads=(1, 1, 1, 1),
                    inputs=[0], output=1, qw=qw, qb=qb, M=M, shift=shift)
    return info, x, node


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vectors', default=str(ROOT / 'experiments' / 'p5_minimal_conv_v1'
                                                  / 'vectors'))
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    vec_dir = Path(args.vectors)

    rows, failures = [], []
    for name in sorted(p.name for p in vec_dir.iterdir() if p.is_dir()):
        info, x, node = load_vector(vec_dir / name)
        # contract bound, computed from the frozen weights and bias only
        bound = np.abs(node.qb) + MAX_ABS_INPUT * np.abs(node.qw.astype(np.int64)).sum(axis=(1, 2, 3))
        acc = conv2d_int8_accumulate(x, node)
        obs = int(np.abs(acc).max())
        worst = int(bound.max())
        ok = obs <= worst and worst <= INT32_MAX and int(np.abs(node.qb).max()) <= INT32_MAX
        rows.append({
            'name': name, 'kind': info['kind'], 'cin': node.cin, 'cout': node.cout,
            'h': info['geometry']['h'], 'w': info['geometry']['w'],
            'observed_abs_acc_max': obs,
            'bound_max': worst,
            'bound_max_channel': int(np.argmax(bound)),
            'headroom_ratio': round(worst / obs, 2) if obs else None,
            'int32_margin': INT32_MAX - worst,
            'within_bound': bool(obs <= worst),
            'bound_fits_int32': bool(worst <= INT32_MAX),
            'ok': bool(ok),
        })
        if not ok:
            failures.append(name)
        print(f"{name:24s} |acc|max={obs:>8d}  B_max={worst:>10d}  "
              f"headroom={worst / obs if obs else float('inf'):>9.1f}x  "
              f"int32_margin={INT32_MAX - worst:>10d}  {'OK' if ok else 'FAIL'}")

    print(f"\naccumulation bound (contract v1.3 §6, max |input| = {MAX_ABS_INPUT}): "
          f"B[o] = |qb[o]| + {MAX_ABS_INPUT}*sum|qw[o,:]|")
    print(f"{len(rows) - len(failures)}/{len(rows)} vectors within their own bound; "
          f"largest bound leaves {min(r['int32_margin'] for r in rows)} of int32 headroom")
    if failures:
        print('FAILED: ' + ', '.join(failures))
    if args.json:
        write_json(Path(args.json), {
            'phase': 'P5.0', 'check': 'accumulator overflow headroom',
            'contract': 'p4-contract-1.3 §6',
            'formula': 'B[o] = |qb[o]| + 128*sum_ci,i,j |qw[o,ci,i,j]|',
            'note': 'the bound is recomputed from the frozen weight/bias files, so it is '
                    'independent of the vector generator\'s own recording',
            'vectors_manifest_sha256': file_hash(vec_dir / 'manifest.json'),
            'vectors': rows, 'failed': failures},
            )
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
