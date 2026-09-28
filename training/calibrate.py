"""P4.2 PTQ calibration: frozen 64-image train subset, per-layer activation scales and error stats.

Scales attach to TENSORS (each producer node's output); the pack builder converts them into
per-conv multiplier/shift pairs and requant nodes. The 109-image development set is never
touched here; scales must be reproducible from this manifest alone (contract §3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from runtime.common import file_hash, write_json
from training.exporter import CONTRACT_PATH, Tracer, load_detection_model, quantize_weights, run_float_executor

CALIB_SEED = 20260928
CALIB_COUNT = 64
TRAIN_DIR = 'datasets/public_expanded_v1_prepared/images/train'
INPUT_SIZE = 128
HIST_BINS = 8192
PERCENTILE = 0.999


def letterbox_128(bgr):
    """Ultralytics-style letterbox: aspect-fit into 128x128, centered, pad value 114, then RGB/255."""
    h, w = bgr.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def frozen_subset():
    files = sorted(p for p in Path(TRAIN_DIR).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    if len(files) < CALIB_COUNT:
        raise ValueError(f'train split has only {len(files)} images')
    rng = np.random.RandomState(CALIB_SEED)
    chosen = sorted(rng.choice(len(files), CALIB_COUNT, replace=False).tolist())
    return [(files[i], files[i].as_posix()) for i in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='experiments/p2_expanded_train/fit/weights/best.pt')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))

    subset = frozen_subset()
    manifest = {'seed': CALIB_SEED, 'count': len(subset), 'source_dir': TRAIN_DIR,
                'letterbox': f'aspect-fit {INPUT_SIZE}x{INPUT_SIZE}, centered, pad 114, RGB, /255',
                'files': [{'path': rel, 'sha256': file_hash(abs_path)} for abs_path, rel in subset]}
    write_json(output / 'calibration_manifest.json', manifest)

    model = load_detection_model(args.weights)
    graph = Tracer(model.model).trace()
    # Conv outputs get their own scales; concat output scales are calibration artifacts
    # of the concatenated tensor (contract §3). Pool/upsample/split/add inherit by rule.
    scale_nodes = [n for n in graph.nodes if n.op.startswith('conv') or n.op == 'concat']
    maxima = {n.name: 0.0 for n in scale_nodes}
    inputs = []
    for abs_path, _rel in subset:
        bgr = cv2.imdecode(np.fromfile(str(abs_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'Cannot decode {abs_path}')
        inputs.append(letterbox_128(bgr))

    def run_pass(collect):
        for x in inputs:
            outputs, _ = run_float_executor(graph, x)
            collect(outputs)

    def collect_max(outputs):
        for node in scale_nodes:
            v = float(outputs[node.name].abs().max())
            if v > maxima[node.name]:
                maxima[node.name] = v

    def make_hist_collector(store):
        def collect(outputs):
            for node in scale_nodes:
                # |values| only: the scale quantizes magnitude; signed values below 0
                # (e.g. all-negative cls logits) would silently fall outside [0, max].
                store.setdefault(node.name, np.zeros(HIST_BINS, dtype=np.int64))[...] += np.histogram(
                    np.abs(outputs[node.name].detach().numpy().ravel()), bins=HIST_BINS,
                    range=(0.0, maxima[node.name]))[0]
        return collect

    run_pass(collect_max)
    store: dict = {}
    run_pass(make_hist_collector(store))

    input_scale = 1.0 / 127.0
    tensor_scales = {0: input_scale}
    rows = []
    for node in scale_nodes:
        hist = store[node.name]
        total = int(hist.sum())
        edges = np.linspace(0.0, maxima[node.name], HIST_BINS + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        cum = np.cumsum(hist) / total
        p999 = float(centers[min(int(np.searchsorted(cum, PERCENTILE)), HIST_BINS - 1)])
        p999 = max(p999, maxima[node.name] / HIST_BINS)

        def l2_error(clip):
            s = clip / 127.0
            inside = centers <= clip
            excess = np.maximum(centers[~inside] - clip, 0.0)
            return float(np.sqrt(np.sum(hist[inside]) * (s ** 2 / 12.0) + np.sum(hist[~inside] * excess ** 2)))

        def sat(clip):
            return float(hist[centers > clip].sum() / total)

        chosen = 'max_abs' if l2_error(maxima[node.name]) <= l2_error(p999) else 'percentile_abs_99.9'
        sx = (maxima[node.name] if chosen == 'max_abs' else p999) / 127.0
        tensor_scales[node.output] = sx
        if node.output2 != -1:
            tensor_scales[node.output2] = sx
        rows.append({'node': node.name, 'max_abs': maxima[node.name], 'p999_abs': p999,
                     'sat_ratio_max': sat(maxima[node.name]), 'sat_ratio_p999': sat(p999),
                     'l2_max': l2_error(maxima[node.name]), 'l2_p999': l2_error(p999),
                     'chosen_rule': chosen, 'scale': sx})

    # Scale-preserving ops inherit (SPPF pools, upsample, chunk halves); add outputs take
    # the first operand's scale per contract. Concat scales were calibrated above.
    for node in graph.nodes:
        if node.op in ('maxpool_5x5', 'upsample_nearest_2x'):
            tensor_scales[node.output] = tensor_scales[node.inputs[0]]
        elif node.op == 'split_chunk':
            src = tensor_scales[node.inputs[0]]
            tensor_scales[node.output] = src
            if node.output2 != -1:
                tensor_scales[node.output2] = src
        elif node.op == 'add':
            tensor_scales[node.output] = tensor_scales[node.inputs[0]]
    missing = [n.name for n in graph.nodes if n.output not in tensor_scales]
    if missing:
        raise AssertionError(f'Nodes without a scale: {missing[:5]}')

    weight_summary = []
    for name, (w, _b) in graph.weights.items():
        qw, sw = quantize_weights(w)
        deq = qw.astype(np.float64) * sw.astype(np.float64)[:, None, None, None]
        weight_summary.append({'node': name, 'l2': float(np.linalg.norm((w.numpy() - deq).ravel())),
                               'max_abs': float(np.abs(w.numpy() - deq).max())})

    report = {'stage': 'p4.2', 'status': 'complete', 'contract_version': contract['contract_version'],
              'weights_sha256': file_hash(args.weights),
              'calibration': {'manifest': 'calibration_manifest.json', 'seed': CALIB_SEED,
                              'count': len(subset), 'percentile': PERCENTILE, 'hist_bins': HIST_BINS,
                              'selection_rule': 'per node: lower estimated L2 error; tie -> max_abs'},
              'input_scale': input_scale,
              'tensor_scales': {str(k): v for k, v in tensor_scales.items()},
              'per_layer': rows,
              'weight_quantization': {'layers': len(weight_summary),
                                      'worst_l2': max(r['l2'] for r in weight_summary),
                                      'worst_max_abs': max(r['max_abs'] for r in weight_summary)},
              'limitations': ['Histogram-based percentile is quantized to 8192 bins; reproducible from manifest + contract.',
                              'Calibration images are train-split only; the 109-image development set was not used for scales.',
                              'Estimated L2 error is a within-bin uniform-error approximation, not an exact error.']}
    write_json(output / 'report.json', report)
    print(f"calibrated {len(rows)} layers (conv+concat) from {len(subset)} images; "
          f"max sat_ratio_p999 = {max(r['sat_ratio_p999'] for r in rows):.4%}; "
          f"worst weight L2 = {report['weight_quantization']['worst_l2']:.3f}")


if __name__ == '__main__':
    main()
