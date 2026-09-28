"""P4.5 acceptance runner: Python and C integer references must match every golden vector.

For each vector: (1) the Python reference re-executes the pack and must reproduce the
frozen expected files byte-for-byte (self-check), (2) the compiled C reference must do the
same. Any mismatch fails the run; the report records per-node hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import numpy as np
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import run_pack  # noqa: E402
from pack_reader import read_pack  # noqa: E402

from runtime.common import file_hash, write_json  # noqa: E402

MAIN_PACK = 'training/export/p4_baseline128_v1/model_pack.bin'


def node_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_vector(name, info, golden_dir, c_binary, work_dir):
    pack_rel = info.get('pack')
    pack_path = (golden_dir / pack_rel) if pack_rel else ROOT / MAIN_PACK
    pack = read_pack(pack_path)
    x = np.frombuffer((golden_dir / 'input.bin').read_bytes(), dtype=np.int8)
    x = x.reshape(pack.tensor_shapes[0])
    outputs, _ = run_pack(pack, x)

    results, ok = {}, True
    for node_name, meta in info['expected'].items():
        expected = (golden_dir / meta['path']).read_bytes()
        mine = outputs[node_name].astype(np.int8).tobytes()
        py_ok = node_digest(mine) == meta['sha256'] == node_digest(expected)
        results[node_name] = {'python': py_ok, 'sha256': meta['sha256']}
        ok &= py_ok
    if not ok:
        return {'python_matches_expected': False, 'c_matches_expected': False, 'nodes': results}

    out_dir = work_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(c_binary), '--pack', str(pack_path), '--input', str(golden_dir / 'input.bin'),
           '--output-dir', str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'C reference failed on {name}: {proc.stderr}')
    c_ok_all = True
    for node_name, meta in info['expected'].items():
        got = (out_dir / f'{node_name}.bin').read_bytes()
        c_ok = node_digest(got) == meta['sha256']
        results[node_name]['c'] = c_ok
        c_ok_all &= c_ok
    return {'python_matches_expected': True, 'c_matches_expected': c_ok_all, 'nodes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--golden', default='tests/golden')
    parser.add_argument('--c-binary', default='reference/c/int_reference.exe')
    parser.add_argument('--output', default='experiments/p4_golden_verification/report.json')
    args = parser.parse_args()
    golden = Path(args.golden)
    manifest = json.loads((golden / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['pack_sha256'] != file_hash(MAIN_PACK):
        raise AssertionError('manifest pack hash does not match the current model pack')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / 'c_outputs'

    vectors = {}
    all_ok = True
    for name, info in manifest['vectors'].items():
        result = verify_vector(name, info, golden / name, Path(args.c_binary), work)
        vectors[name] = {'kind': info['kind'], 'nodes': len(result['nodes']),
                         'python_matches_expected': result['python_matches_expected'],
                         'c_matches_expected': result['c_matches_expected']}
        all_ok &= result['python_matches_expected'] and result['c_matches_expected']
        print(f"{name}: python={'PASS' if result['python_matches_expected'] else 'FAIL'} "
              f"c={'PASS' if result['c_matches_expected'] else 'FAIL'} ({len(result['nodes'])} nodes)")
    report = {'status': 'passed' if all_ok else 'failed',
              'contract_version': manifest['contract_version'], 'pack_sha256': manifest['pack_sha256'],
              'c_binary_sha256': file_hash(args.c_binary),
              'vectors': vectors,
              'acceptance': 'Python and C integer references bit-exact on every golden vector '
                            '(P4 acceptance criterion); C timing is not a performance claim.'}
    write_json(output, report)
    if not all_ok:
        sys.exit(1)
    print(f"all {len(vectors)} vectors bit-exact: {output}")


if __name__ == '__main__':
    main()
