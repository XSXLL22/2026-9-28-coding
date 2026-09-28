"""P5.0 simulation runner: compile the frozen RTL with the P5 testbench, replay every vector
through it and compare the emitted bytes against the pre-simulation expectations.

Failure model: a scenario passes only if the simulator exits 0, the testbench prints its
pass marker, and the output file is byte-identical to expected.mem. The runner never
regenerates expectations -- it only reads the vector directory produced by
tools/build_p5_vectors.py before any simulation existed.

Vectors are frozen: this script refuses to run if the vector directory has been modified
since the manifest was written (hash check), so a failing simulation cannot be "fixed" by
quietly editing a vector.

Usage:
    python tools/run_p5_sim.py                     # full suite
    python tools/run_p5_sim.py --only basic_real   # one or more scenario names
    python tools/run_p5_sim.py --list              # show the scenario table
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import file_hash, write_json  # noqa: E402

TB = ROOT / 'hardware' / 'tb' / 'tb_conv3x3.v'
RTL = [ROOT / 'hardware' / 'rtl' / 'mac_array.v',
       ROOT / 'hardware' / 'rtl' / 'requant_sat.v',
       ROOT / 'hardware' / 'rtl' / 'conv3x3_core.v']
DEFAULT_VECTORS = ROOT / 'experiments' / 'p5_minimal_conv_v1' / 'vectors'
DEFAULT_WORK = ROOT / 'experiments' / 'p5_minimal_conv_v1'

REAL = 'real_model.6.m.0.cv1'
SEED = 20260928

# Scenario ids interpreted by the testbench (see hardware/tb/tb_conv3x3.v header).
SC_BASIC, SC_STALL, SC_RESET, SC_TWO, SC_BADCFG, SC_LOCKED, SC_UNDERFLOW = range(7)


def scenario_table():
    """Every row is one vvp invocation. `vcd` rows additionally dump a small waveform."""
    crafted = ['syn_round_table', 'syn_sat_half', 'syn_shift0_sat', 'syn_neg128',
               'syn_rnd_3x3_c1_o1', 'syn_rnd_5x7_c3_o2', 'syn_rnd_7x3_c5_o3', 'syn_zero']
    rows = [
        dict(name='basic_' + v, sid=SC_BASIC, vec=v,
             note='plain run, every vector in the set')
        for v in crafted
    ]
    rows.append(dict(name='basic_real', sid=SC_BASIC, vec=REAL,
                     note='plain run on a real pack layer (64x64 channels, 8x8, 4096 outputs)'))

    # waveform coverage: the crafted vectors are short enough that a full VCD stays openable
    for v in ('syn_sat_half', 'syn_rnd_7x3_c5_o3'):
        rows.append(dict(name=f'wave_{v}', sid=SC_BASIC, vec=v, vcd=True,
                         note='same run as basic_%s, with a waveform dump for inspection' % v))

    rows += [
        dict(name='stall_syn_rnd_5x7_c3_o2', sid=SC_STALL, vec='syn_rnd_5x7_c3_o2', vcd=True,
             note='random input and output backpressure, with a waveform dump'),
        dict(name='stall_real', sid=SC_STALL, vec=REAL,
             note='random input and output backpressure on the real layer'),
        dict(name='reset_mid_syn_rnd_5x7_c3_o2', sid=SC_RESET, vec='syn_rnd_5x7_c3_o2', vcd=True,
             note='synchronous reset in the middle of the input load, then a clean re-submit, '
                  'with a waveform dump'),
        dict(name='reset_mid_real', sid=SC_RESET, vec=REAL,
             note='mid-task reset and re-submit on the real layer'),
        dict(name='two_tasks', sid=SC_TWO, vec='syn_rnd_3x3_c1_o1', vec2='syn_zero',
             note='two consecutive tasks with different geometry; the second is all-zero, so '
                  'any state carried over from the first shows up as a non-zero output'),
        dict(name='bad_config_real', sid=SC_BADCFG, vec=REAL,
             note='cin=0 rejected with UNSUPPORTED_CONFIG and no output, then the same vector '
                  'runs normally (recovery)'),
        dict(name='locked_config_real', sid=SC_LOCKED, vec=REAL,
             note='a config write during BUSY is rejected with CONFIG_LOCKED and the running '
                  'task still produces the correct bytes'),
        dict(name='param_underflow_real', sid=SC_UNDERFLOW, vec=REAL,
             note='one param word short: the core must give up by itself with PARAM_UNDERFLOW, '
                  'emit nothing, and stay usable'),
    ]
    return rows


def rel(p):
    """Plusarg path form: project-relative with forward slashes.

    The project directory name is not ASCII; iverilog opens plusarg paths through the
    C runtime, so a short relative path under a known cwd is the most portable form.
    """
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def load_vectors(vec_dir):
    manifest_path = vec_dir / 'manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'{manifest_path} missing; run tools/build_p5_vectors.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    # freeze check: a vector edited after generation would silently invalidate the run
    for name, info in manifest['vectors'].items():
        for fname, want in info['files'].items():
            got = file_hash(vec_dir / name / fname)
            if got != want:
                raise SystemExit(f'vector {name}/{fname} changed since generation '
                                 f'({got[:12]} != {want[:12]}); P5 vectors are frozen')
    return manifest


def parse_ok(stdout):
    m = re.search(r'TB_SCEN=(\d+) OUT_COUNT=(\d+) DONE=(\d+) ERRORS=(\d+) FIRST_ERR=(-?\d+) '
                  r'CYCLES=(\d+) WATCHDOG_CYC=(\d+)', stdout)
    if not m:
        return None
    keys = ['scen', 'out_count', 'done', 'errors', 'first_err', 'cycles', 'watchdog_cyc']
    return {k: int(v) for k, v in zip(keys, m.groups())}


def run_scenario(row, vec_dir, work, sim, log_dir, out_dir):
    name = row['name']
    vec = vec_dir / row['vec']
    out_file = out_dir / f'{name}.out'
    args = ['vvp', rel(sim),
            f"+SCEN={row['sid']}", f"+SEED={SEED}",
            f"+IN={rel(vec / 'input.mem')}", f"+WT={rel(vec / 'weight.mem')}",
            f"+PARAM={rel(vec / 'param.mem')}", f"+OUT={rel(out_file)}"]
    geom = json.loads((vec / 'vector.json').read_text(encoding='utf-8'))['geometry']
    args += [f"+CIN={geom['cin']}", f"+COUT={geom['cout']}",
             f"+H={geom['h']}", f"+W={geom['w']}"]

    expected_files = [vec / 'expected.mem']
    if 'vec2' in row:
        vec2 = vec_dir / row['vec2']
        g2 = json.loads((vec2 / 'vector.json').read_text(encoding='utf-8'))['geometry']
        args += [f"+IN2={rel(vec2 / 'input.mem')}", f"+WT2={rel(vec2 / 'weight.mem')}",
                 f"+PARAM2={rel(vec2 / 'param.mem')}",
                 f"+OUT2={rel(out_dir / (name + '_task2.out'))}",
                 f"+CIN2={g2['cin']}", f"+COUT2={g2['cout']}",
                 f"+H2={g2['h']}", f"+W2={g2['w']}"]
        expected_files.append(vec2 / 'expected.mem')

    vcd = None
    if row.get('vcd'):
        vcd = out_dir / f'{name}.vcd'
        args.append(f'+VCD={rel(vcd)}')

    log_path = log_dir / f'{name}.log'
    t0 = time.time()
    # bytes, not text: iverilog/vvp emit messages in the console codepage, and a decode
    # failure must not be able to turn a real run into a crash of the runner itself
    proc = subprocess.run(args, capture_output=True, cwd=str(ROOT))
    elapsed = time.time() - t0
    stdout = (proc.stdout or b'').decode('utf-8', errors='replace')
    stderr = (proc.stderr or b'').decode('utf-8', errors='replace')
    log_path.write_text(
        '# command: ' + ' '.join(args) + '\n'
        f'# exit={proc.returncode} wall={elapsed:.1f}s\n\n'
        '--- stdout ---\n' + stdout + '\n--- stderr ---\n' + stderr,
        encoding='utf-8', newline='\n')

    result = dict(name=name, sid=row['sid'], vec=row['vec'], note=row['note'],
                  exit=proc.returncode, wall_s=round(elapsed, 1), log=str(log_path.relative_to(ROOT)))
    stats = parse_ok(stdout)
    result['dut'] = stats
    # the TB's own pass marker; required so an early $finish cannot look like a pass
    result["tb_pass"] = "TB_PASS" in stdout

    outputs = [out_file] + ([out_dir / (name + '_task2.out')] if 'vec2' in row else [])
    result['compare'] = []
    for produced, expected in zip(outputs, expected_files):
        if not produced.exists():
            result['compare'].append({'file': produced.name, 'match': False,
                                      'reason': 'no output file produced'})
            continue
        got, want = produced.read_bytes(), expected.read_bytes()
        result['compare'].append({
            'file': produced.name, 'expected': str(expected.relative_to(ROOT)),
            'bytes': len(got), 'match': got == want,
            'reason': None if got == want else
                      (f'{len(got)} bytes vs {len(want)} expected' if len(got) != len(want)
                       else 'same length, differing bytes'),
        })
    if vcd is not None:
        result['vcd'] = {'path': str(vcd.relative_to(ROOT)),
                         'bytes': vcd.stat().st_size if vcd.exists() else 0}
    result['ok'] = (result['exit'] == 0 and result['tb_pass']
                    and all(c['match'] for c in result['compare']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vectors', default=str(DEFAULT_VECTORS))
    parser.add_argument('--work', default=str(DEFAULT_WORK))
    parser.add_argument('--only', nargs='*', default=None,
                        help='scenario names to run (default: all)')
    parser.add_argument('--list', action='store_true', help='print the scenario table and exit')
    parser.add_argument('--keep-sim', action='store_true', help='do not recompile')
    args = parser.parse_args()

    rows = scenario_table()
    if args.list:
        for r in rows:
            print(f"{r['name']:34s} scen={r['sid']} vec={r['vec']:24s} {r['note']}")
        return 0
    if args.only:
        unknown = set(args.only) - {r['name'] for r in rows}
        if unknown:
            raise SystemExit(f'unknown scenario(s): {sorted(unknown)}')
        rows = [r for r in rows if r['name'] in args.only]

    vec_dir = Path(args.vectors)
    work = Path(args.work)
    log_dir = work / 'logs'
    out_dir = work / 'rtl_out'
    build = work / 'build'
    for d in (log_dir, out_dir, build):
        d.mkdir(parents=True, exist_ok=True)
    manifest = load_vectors(vec_dir)

    sim = build / 'conv3x3_sim.vvp'
    compile_log = log_dir / 'iverilog_compile.log'
    if not args.keep_sim or not sim.exists():
        cmd = ['iverilog', '-g2001', '-o', str(sim), str(TB)] + [str(f) for f in RTL]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        compile_log.write_text('# ' + ' '.join(cmd) + f'\n# exit={proc.returncode}\n\n'
                               + proc.stdout + proc.stderr,
                               encoding='utf-8', newline='\n')
        if proc.returncode != 0:
            print(proc.stdout + proc.stderr)
            raise SystemExit('iverilog compile failed; see ' + str(compile_log))
        iverilog_version = subprocess.run(['iverilog', '-V'], capture_output=True, text=True
                                          ).stdout.splitlines()[0]
    else:
        iverilog_version = 'reused existing simulation (--keep-sim)'

    results = []
    for row in rows:
        res = run_scenario(row, vec_dir, work, sim, log_dir, out_dir)
        results.append(res)
        mark = 'PASS' if res['ok'] else 'FAIL'
        cyc = res['dut']['cycles'] if res['dut'] else -1
        extra = ''
        if not res['ok']:
            why = []
            if res['exit'] != 0:
                why.append(f"exit={res['exit']}")
            if not res['tb_pass']:
                why.append('no TB_PASS')
            why += [f"{c['file']}: {c['reason']}" for c in res['compare'] if not c['match']]
            extra = '  <- ' + '; '.join(why)
        print(f'{mark}  {res["name"]:34s} out={cyc:>9d}  {res["wall_s"]:>6.1f}s  '
              f'bytes={[c["bytes"] for c in res["compare"]]}{extra}')
        if res.get('vcd'):
            print(f'      waveform: {res["vcd"]["path"]} ({res["vcd"]["bytes"]} bytes)')

    report = {
        'phase': 'P5.0',
        'description': 'minimal 3x3/stride=1/pad=1 convolution core, bit-exact against the '
                       'pre-SiLU Python integer reference boundary',
        'vectors': {'dir': str(vec_dir.relative_to(ROOT)),
                    'manifest_sha256': file_hash(vec_dir / 'manifest.json'),
                    'vector_version': manifest['vector_version'],
                    'count': len(manifest['vectors']),
                    'generator': manifest['generator'],
                    'generator_sha256': manifest['generator_sha256'],
                    'pack': manifest['pack'], 'pack_sha256': manifest['pack_sha256'],
                    'seed': manifest['seed']},
        'rtl': {str(p.relative_to(ROOT)): file_hash(p) for p in RTL},
        'testbench': {str(TB.relative_to(ROOT)): file_hash(TB)},
        'runner': {str(Path(__file__).resolve().relative_to(ROOT)): file_hash(Path(__file__))},
        'toolchain': {'iverilog': iverilog_version, 'seed': SEED},
        'comparison': 'RTL output file vs expected.mem, raw bytes (no line-ending normalisation)',
        'note': 'cycle counts are simulation measurements from the DUT cycle_count register, '
                'not synthesis or board results; no synthesis, timing or hardware claim is made',
        'scenarios': results,
        'passed': sum(1 for r in results if r['ok']),
        'failed': sum(1 for r in results if not r['ok']),
    }
    write_json(work / 'sim_report.json', report)

    print(f"\n{report['passed']}/{len(results)} scenarios passed; report: "
          f"{work / 'sim_report.json'}")
    if report['failed']:
        sys.exit(1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
