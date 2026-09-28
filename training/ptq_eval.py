"""P4.7 PTQ accuracy evaluation (E0-corrected): one coordinate space, verified provenance.

E0 corrections over the superseded version (experiments/p4_ptq_eval, see CORRECTION_NOTICE):
- GT lives in ORIGINAL image pixel space (normalized labels x original dims); predictions
  are inverse-mapped from the letterbox canvas via the transform metadata produced by the
  SAME forward routine. The pre-fix code stretched GT to 128x128, so on every non-square
  image (all 109) GT and predictions lived in different frames.
- Provenance is enforced before any inference: weights/pack/calibration hashes, contract
  version, input shape, six scope scales bound to output nodes (finite, positive), and the
  109 dev images+labels against the frozen data audit.
- F0 (framework float network, hook capture) and F1 (folded float executor) are compared
  per node and end-to-end (AP50 within 0.1pp) before INT8 (Q0) is compared against them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python'))
from int_reference import quantize_input, run_pack  # noqa: E402
from pack_reader import read_pack  # noqa: E402

from training.error_analysis import match  # noqa: E402
from training.exporter import CONTRACT_PATH, Tracer, load_detection_model, run_float_executor  # noqa: E402
from runtime.common import file_hash, write_json  # noqa: E402

REG_MAX = 16
STRIDES = (8, 16, 32)
CONF_WORKING_POINT = 0.25
NMS_IOU_AP = 0.7
NMS_IOU_WORKING_POINT = 0.45
CONF_CANDIDATE = 0.001
DEFAULT_CALIBRATION = 'experiments/p4_ptq/report.json'
DEFAULT_DATA_AUDIT = 'experiments/p2_expanded_new_evaluation/data_audit.json'
AP_DIAGNOSTIC_TOLERANCE_PP = 0.1


class LetterboxTransform:
    """Single source of the image<->canvas geometry (E0.2). Forward and inverse share it.

    Convention = ultralytics LetterBox(new_shape, auto=False, center=True): scale = min of
    the two sides, new size rounded once, padding split with floor (equivalent to the
    framework's round(d/2 - 0.1)/round(d/2 + 0.1) pair). Realized per-axis gains are the
    ROUNDED sizes over the original sizes, matching what cv2.resize actually did.
    """

    def __init__(self, orig_w, orig_h, canvas=128):
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError('non-positive original size')
        self.orig_w, self.orig_h, self.canvas = int(orig_w), int(orig_h), int(canvas)
        scale = min(self.canvas / self.orig_h, self.canvas / self.orig_w)
        self.nw = int(round(self.orig_w * scale))
        self.nh = int(round(self.orig_h * scale))
        self.gain_x = self.nw / self.orig_w
        self.gain_y = self.nh / self.orig_h
        self.pad_left = (self.canvas - self.nw) // 2
        self.pad_top = (self.canvas - self.nh) // 2

    def apply(self, bgr):
        """Letterbox a BGR image; returns (uint8 CHW RGB canvas, self)."""
        import cv2
        resized = cv2.resize(bgr, (self.nw, self.nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.canvas, self.canvas, 3), 114, dtype=np.uint8)
        canvas[self.pad_top:self.pad_top + self.nh, self.pad_left:self.pad_left + self.nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1), self

    def original_box_to_canvas(self, box):
        out = np.array([box[0] * self.gain_x + self.pad_left, box[1] * self.gain_y + self.pad_top,
                        box[2] * self.gain_x + self.pad_left, box[3] * self.gain_y + self.pad_top],
                       dtype=np.float64)
        return np.clip(out, 0, self.canvas)

    def canvas_box_to_original(self, box):
        out = np.array([(box[0] - self.pad_left) / self.gain_x, (box[1] - self.pad_top) / self.gain_y,
                        (box[2] - self.pad_left) / self.gain_x, (box[3] - self.pad_top) / self.gain_y],
                       dtype=np.float64)
        out[0] = min(max(out[0], 0.0), float(self.orig_w))
        out[2] = min(max(out[2], 0.0), float(self.orig_w))
        out[1] = min(max(out[1], 0.0), float(self.orig_h))
        out[3] = min(max(out[3], 0.0), float(self.orig_h))
        return out

    def metadata(self):
        return {'orig_w': self.orig_w, 'orig_h': self.orig_h, 'canvas': self.canvas,
                'nw': self.nw, 'nh': self.nh, 'gain_x': self.gain_x, 'gain_y': self.gain_y,
                'pad_left': self.pad_left, 'pad_top': self.pad_top}


def decode_head(raws, scales=None):
    """raws: per scale [box(64,h,w), cls(2,h,w)] ascending stride. scales: per-tensor
    dequantization scales (int8 path) or None (float path). Returns canvas-space xyxy and
    post-sigmoid class scores."""
    boxes, scores = [], []
    for si, stride in enumerate(STRIDES):
        box_raw, cls_raw = raws[2 * si], raws[2 * si + 1]
        if scales is not None:
            box_raw = box_raw.astype(np.float64) * scales[2 * si]
            cls_raw = cls_raw.astype(np.float64) * scales[2 * si + 1]
        c, h, w = box_raw.shape
        dist = box_raw.reshape(4, REG_MAX, h, w).astype(np.float64)
        dist = dist - dist.max(axis=1, keepdims=True)
        e = np.exp(dist)
        softmax = e / e.sum(axis=1, keepdims=True)
        expect = (softmax * np.arange(REG_MAX, dtype=np.float64)[None, :, None, None]).sum(axis=1)
        l, t, r, b = expect
        ax = (np.arange(w, dtype=np.float64) + 0.5) * stride
        ay = (np.arange(h, dtype=np.float64) + 0.5) * stride
        cx, cy = np.meshgrid(ax, ay)
        x1 = (cx - l * stride).ravel()
        y1 = (cy - t * stride).ravel()
        x2 = (cx + r * stride).ravel()
        y2 = (cy + b * stride).ravel()
        boxes.append(np.stack([x1, y1, x2, y2], 1))
        logits = cls_raw.reshape(2, -1).T
        scores.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(boxes), np.concatenate(scores)


def nms_per_class(boxes, scores, num_classes, iou_threshold, conf=CONF_CANDIDATE):
    keep_boxes, keep_scores, keep_classes = [], [], []
    for cls in range(num_classes):
        s = scores[:, cls].copy()
        order = np.argsort(-s)
        b = boxes[order]
        s = s[order]
        alive = s >= conf
        for i in range(len(s)):
            if not alive[i]:
                continue
            keep_boxes.append(b[i])
            keep_scores.append(s[i])
            keep_classes.append(cls)
            xx1 = np.maximum(b[i, 0], b[i + 1:, 0])
            yy1 = np.maximum(b[i, 1], b[i + 1:, 1])
            xx2 = np.minimum(b[i, 2], b[i + 1:, 2])
            yy2 = np.minimum(b[i, 3], b[i + 1:, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_j = (b[i + 1:, 2] - b[i + 1:, 0]) * (b[i + 1:, 3] - b[i + 1:, 1])
            iou = inter / np.maximum(area_i + area_j - inter, 1e-9)
            alive[i + 1:] &= ~(iou > iou_threshold)
    if not keep_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.array(keep_boxes), np.array(keep_scores), np.array(keep_classes)


def iou_box(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50_pooled(predictions, gt_per_image, num_classes, iou=0.5):
    """Pooled AP50: a class's predictions across all images compete for one GT pool;
    AP = all-point interpolated PR AUC."""
    aps = {}
    for cls in range(num_classes):
        n_gt = sum(len(gt['boxes'][gt['classes'] == cls]) for gt in gt_per_image.values())
        preds = [(p['scores'][i], img, p['boxes'][i])
                 for img, p in predictions.items() for i in np.where(p['classes'] == cls)[0]]
        preds.sort(key=lambda t: -t[0])
        matched = {img: np.zeros(len(gt['boxes'][gt['classes'] == cls]), dtype=bool)
                   for img, gt in gt_per_image.items()}
        tp, fp = np.zeros(len(preds)), np.zeros(len(preds))
        for rank, (_s, img, box) in enumerate(preds):
            gt = gt_per_image[img]
            gboxes = gt['boxes'][gt['classes'] == cls]
            if len(gboxes) == 0:
                fp[rank] = 1
                continue
            ious = np.array([iou_box(box, g) for g in gboxes])
            best = int(np.argmax(ious))
            if ious[best] >= iou and not matched[img][best]:
                matched[img][best] = True
                tp[rank] = 1
            else:
                fp[rank] = 1
        if n_gt == 0:
            aps[cls] = None
            continue
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        mrec = np.concatenate([[0], recall, [1]])
        mpre = np.concatenate([[1], precision, [0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps[cls] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return aps


def load_gt_normalized(label_path):
    """Raw normalized labels; scaling happens once the transform is known (E0.2)."""
    boxes, classes = [], []
    for line in Path(label_path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        c, xc, yc, bw, bh = map(float, line.split())
        boxes.append([xc, yc, bw, bh])
        classes.append(int(c))
    return boxes, classes


def load_gt(label_path, meta):
    """YOLO normalized center-format labels -> ORIGINAL pixel space xyxy (E0.2)."""
    boxes_n, classes = load_gt_normalized(label_path)
    boxes = np.array([[(b[0] - b[2] / 2) * meta.orig_w, (b[1] - b[3] / 2) * meta.orig_h,
                       (b[0] + b[2] / 2) * meta.orig_w, (b[1] + b[3] / 2) * meta.orig_h]
                      for b in boxes_n]).reshape(-1, 4)
    return {'boxes': boxes, 'classes': np.asarray(classes, dtype=int)}


def verify_sources(args, pack, model, expected_val_count=109):
    """E0.3: fail before any expensive inference when identities do not line up."""
    problems = []
    pack_json_path = Path(args.pack).with_suffix('.json')
    weights_sha = file_hash(args.weights)
    if pack.meta['weights_sha256'] != weights_sha:
        problems.append(f"pack weights_sha256 {pack.meta['weights_sha256'][:12]} != weights file {weights_sha[:12]}")
    calib = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    if calib.get('weights_sha256') != weights_sha:
        problems.append('calibration report was built from different weights')
    if pack_json_path.exists():
        manifest = json.loads(pack_json_path.read_text(encoding='utf-8'))
        if manifest.get('pack_sha256') != file_hash(args.pack):
            problems.append('model_pack.bin does not match its own JSON manifest')
        if manifest.get('calibration_report_sha256') and manifest['calibration_report_sha256'] != file_hash(args.calibration):
            problems.append('calibration report changed since the pack was built')
    else:
        problems.append('model pack JSON manifest missing')
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if pack.meta['contract_version'] != contract['contract_version']:
        problems.append(f"contract version mismatch: pack {pack.meta['contract_version']} vs current {contract['contract_version']}")
    if tuple(pack.tensor_shapes[0]) != (3, 128, 128):
        problems.append(f'unexpected pack input shape {pack.tensor_shapes[0]}')
    scale_nodes = [n for n in pack.nodes if n.output in set(pack.meta['graph']['scope_outputs'])]
    calib_scales = {int(k): v for k, v in calib['tensor_scales'].items()}
    scope_scales = []
    for node in scale_nodes:
        s = calib_scales.get(node.output)
        if s is None or not np.isfinite(s) or s <= 0:
            problems.append(f'scale for {node.name} (tid {node.output}) missing/non-finite/non-positive: {s}')
        scope_scales.append(s)
    if len(scope_scales) != 6:
        problems.append(f'expected 6 scope scales, bound {len(scope_scales)}')
    audit = json.loads((ROOT / DEFAULT_DATA_AUDIT).read_text(encoding='utf-8'))
    audit_entries = {Path(row['image']).name: row for row in audit['files'] if row['split'] == 'val'}
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    if len(images) != expected_val_count:
        problems.append(f'development set has {len(images)} images, expected {expected_val_count}')
    for p in images:
        row = audit_entries.get(p.name)
        if row is None:
            problems.append(f'{p.name} missing from frozen data audit')
            continue
        label = Path(args.labels) / (p.stem + '.txt')
        if file_hash(p) != row['image_sha256'] or file_hash(label) != row['label_sha256']:
            problems.append(f'{p.name} or its label changed against the frozen audit')
    return problems, calib, scope_scales, images


def _working_point_match(pred_boxes, pred_scores, pred_classes, ci, gt):
    sel = pred_classes == ci
    plist = [{'xyxy_original_pixels': b, 'confidence': s}
             for b, s in zip(pred_boxes[sel], pred_scores[sel]) if s >= CONF_WORKING_POINT]
    return match(plist, [list(b) for b in gt['boxes'][gt['classes'] == ci]], 0.5)


def _validate_decoder(model, u8, raws_f):
    """Framework head decode emits xywh; convert to xyxy, then compare geometry/scores."""
    with torch.no_grad():
        y = model(torch.from_numpy(u8.astype(np.float32) / 255.0).unsqueeze(0))[0][0].numpy()
    boxes, scores = decode_head(raws_f)
    n = y.shape[1]
    if n != len(boxes):
        raise AssertionError(f'anchor count mismatch: mine {len(boxes)} vs framework {n}')
    cx, cy, bw, bh = y[0], y[1], y[2], y[3]
    lib_boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    lib_scores = y[4:6].T
    box_diff = float(np.abs(boxes - lib_boxes).max())
    score_diff = float(np.abs(scores - lib_scores).max())
    if box_diff > 0.1 or score_diff > 1e-4:
        raise AssertionError(f'decoder validation failed: box diff {box_diff:.4f}px, score diff {score_diff:.2e}')
    return {'anchors': n, 'box_max_abs_diff_px': box_diff, 'score_max_abs_diff': score_diff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', default='training/export/p4_baseline128_v1/model_pack.bin')
    parser.add_argument('--weights', default='experiments/p2_expanded_train/fit/weights/best.pt')
    parser.add_argument('--calibration', default=DEFAULT_CALIBRATION)
    parser.add_argument('--images', default='datasets/public_expanded_comparison/images/val')
    parser.add_argument('--labels', default='datasets/public_expanded_comparison/labels/val')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke-count', type=int, default=0, help='engineering smoke check on the first N images')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    pack = read_pack(args.pack)
    model = load_detection_model(args.weights)
    graph = Tracer(model.model).trace()
    scope_nodes = [n for n in graph.nodes if n.output in set(graph.scope_outputs)]
    scope_names = [n.name for n in scope_nodes]
    problems, calib, scope_scales, images = verify_sources(args, pack, model)
    if problems:
        write_json(output / 'source_problems.json', {'status': 'rejected', 'problems': problems})
        raise SystemExit(f'source verification failed: {problems}')
    if args.smoke_count:
        images = images[:args.smoke_count]

    hook_targets = {}
    for n in scope_nodes:
        obj = model
        for part in n.source_module.split('.'):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        hook_targets[n.name] = obj
    hook_outputs = {}
    handles = []
    for name, module in hook_targets.items():
        def make_hook(name):
            def hook(_m, _i, out):
                hook_outputs[name] = out.detach().float().numpy()[0]
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))

    preds = {'fp32_f0': {}, 'fp32_f1': {}, 'int8': {}}
    gt_cache = {}
    per_node_diffs = []
    per_image = []
    decoder_validation = None
    for path in images:
        import cv2
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'cannot decode {path}')
        meta = LetterboxTransform(bgr.shape[1], bgr.shape[0])
        u8, _ = meta.apply(bgr)
        x_float = torch.from_numpy(u8.astype(np.float32) / 255.0).unsqueeze(0)
        with torch.no_grad():
            model(x_float)  # F0: framework forward; hooks capture raw head outputs
        raws_f0 = [np.ascontiguousarray(hook_outputs[name]) for name in scope_names]
        fout, _ = run_float_executor(graph, x_float)
        raws_f1 = [fout[name][0].numpy() for name in scope_names]
        x_int = quantize_input(u8)
        iout, _ = run_pack(pack, x_int)
        raws_q = [iout[name] for name in scope_names]
        if decoder_validation is None:
            decoder_validation = _validate_decoder(model, u8, raws_f0)
        for name, a, b in zip(scope_names, raws_f0, raws_f1):
            per_node_diffs.append({'image': path.name, 'node': name, 'max_abs_diff': float(np.abs(a - b).max())})
        gt = load_gt(Path(args.labels) / (path.stem + '.txt'), meta)
        gt_cache[path.name] = gt
        decoded = {}
        for tag, raws, scales_used in (('fp32_f0', raws_f0, None), ('fp32_f1', raws_f1, None),
                                       ('int8', raws_q, scope_scales)):
            boxes, scores = decode_head(raws, scales_used)
            nb, ns, nc = nms_per_class(boxes, scores, num_classes=2, iou_threshold=NMS_IOU_AP)
            # E0.2: NMS stays in canvas space; matching/AP happen in ORIGINAL space
            nb_orig = np.array([meta.canvas_box_to_original(b) for b in nb]).reshape(-1, 4)
            preds[tag][path.name] = {'boxes': nb_orig, 'scores': ns, 'classes': nc}
            decoded[tag] = (nb_orig, ns, nc)
        int_nb, int_ns, int_nc = decoded['int8']
        f0_nb, f0_ns, f0_nc = decoded['fp32_f0']
        wp_int = {cls: _working_point_match(int_nb, int_ns, int_nc, ci, gt) for ci, cls in ((0, 'smoke'), (1, 'fire'))}
        wp_fp = {cls: _working_point_match(f0_nb, f0_ns, f0_nc, ci, gt) for ci, cls in ((0, 'smoke'), (1, 'fire'))}
        per_image.append({'image': path.name, 'transform': meta.metadata(),
                          'working_point_int8': wp_int, 'working_point_fp32': wp_fp,
                          'negative_with_detection': bool(len(gt['boxes']) == 0 and (int_ns >= CONF_WORKING_POINT).any()),
                          'negative_with_detection_fp32': bool(len(gt['boxes']) == 0 and (f0_ns >= CONF_WORKING_POINT).any())})
    for h in handles:
        h.remove()

    ap_results = {}
    for tag in ('fp32_f0', 'fp32_f1', 'int8'):
        aps = ap50_pooled(preds[tag], gt_cache, 2)
        ap_results[tag] = {'ap50_smoke': aps[0], 'ap50_fire': aps[1],
                           'map50': (aps[0] + aps[1]) / 2 if None not in aps.values() else None}
    f0_map, f1_map, q_map = (ap_results[t]['map50'] for t in ('fp32_f0', 'fp32_f1', 'int8'))
    f0_f1_diff_pp = None
    if args.smoke_count:
        gate = {'delta_map50_pp': None, 'note': 'smoke run: AP/gate undefined when a class has no GT in the subset'}
    else:
        if f0_map is None or f1_map is None or q_map is None:
            raise AssertionError('a class had no GT; AP undefined')
        f0_f1_diff_pp = round((f1_map - f0_map) * 100, 3)
        if abs(f1_map - f0_map) > AP_DIAGNOSTIC_TOLERANCE_PP / 100:
            raise AssertionError(f'F0/F1 AP50 differ by {f0_f1_diff_pp}pp > {AP_DIAGNOSTIC_TOLERANCE_PP}pp tolerance')
        delta = q_map - f0_map
        gate = {'delta_map50_pp': round(delta * 100, 2), 'exploration_target_pp': -3.0,
                'f0_f1_diff_pp': f0_f1_diff_pp,
                'gate': 'PASS' if delta >= -0.03 else 'EXCEEDED: propose QAT plan for user decision'}

    neg_int = sum(1 for r in per_image if r['negative_with_detection'])
    neg_fp32 = sum(1 for r in per_image if r['negative_with_detection_fp32'])
    tp_int = {c: sum(r['working_point_int8'][c]['tp'] for r in per_image) for c in ('smoke', 'fire')}
    fp_int = {c: sum(r['working_point_int8'][c]['fp'] for r in per_image) for c in ('smoke', 'fire')}
    fn_int = {c: sum(r['working_point_int8'][c]['fn'] for r in per_image) for c in ('smoke', 'fire')}
    tp_fp32 = {c: sum(r['working_point_fp32'][c]['tp'] for r in per_image) for c in ('smoke', 'fire')}
    fp_fp32 = {c: sum(r['working_point_fp32'][c]['fp'] for r in per_image) for c in ('smoke', 'fire')}
    fn_fp32 = {c: sum(r['working_point_fp32'][c]['fn'] for r in per_image) for c in ('smoke', 'fire')}

    worst_nodes = sorted(per_node_diffs, key=lambda r: -r['max_abs_diff'])[:12]
    report = {'stage': 'p4.7-E0', 'status': 'complete' if not args.smoke_count else 'smoke-ok',
              'smoke_run': bool(args.smoke_count),
              'contract_version': contract['contract_version'],
              'images': len(images),
              'coordinate_note': 'GT and predictions both live in ORIGINAL image pixel space; predictions '
                                 'inverse-mapped through LetterboxTransform metadata (E0.2).',
              'decoder_validation': decoder_validation,
              'f0_f1': {'per_node_max_abs_diff': max(d['max_abs_diff'] for d in per_node_diffs),
                        'ap50_diff_pp': f0_f1_diff_pp, 'tolerance_pp': AP_DIAGNOSTIC_TOLERANCE_PP,
                        'worst_nodes': worst_nodes if not args.smoke_count else worst_nodes,
                        'ap_diff_evaluated': not args.smoke_count},
              'ap50': ap_results,
              'working_point': {'int8': {'conf': CONF_WORKING_POINT, 'nms_iou': NMS_IOU_AP, 'match_iou': 0.5,
                                         'tp': tp_int, 'fp': fp_int, 'fn': fn_int,
                                         'negative_images_with_detection': neg_int},
                                'fp32_f0': {'tp': tp_fp32, 'fp': fp_fp32, 'fn': fn_fp32,
                                            'negative_images_with_detection': neg_fp32},
                                'note': 'both paths here use the AP candidate setting NMS 0.7; the R1.3 fixed '
                                        'point used NMS 0.45 and is not comparable in absolute terms'},
              'qat_gate': gate,
              'per_image': per_image,
              'source_hashes': {'weights_sha256': file_hash(args.weights), 'pack_sha256': file_hash(args.pack),
                                'calibration_sha256': file_hash(args.calibration)},
              'limitations': ['109 reused development images; not independent test or campus acceptance.',
                              'Box metrics are not event alarm rates.',
                              'Pooled AP50 from the shared in-repo decoder/NMS is not comparable to library mAP numbers.']}
    write_json(output / 'report.json', report)
    def _r(v):
        return round(v, 4) if v is not None else None
    print(json.dumps({'fp32_f0_map50': _r(f0_map), 'fp32_f1_map50': _r(f1_map),
                      'int8_map50': _r(q_map), 'f0_f1_diff_pp': f0_f1_diff_pp,
                      'delta_pp': gate['delta_map50_pp'], 'gate': gate.get('gate'),
                      'int8_wp': {'tp': tp_int, 'fp': fp_int, 'neg_img': neg_int},
                      'fp32_wp': {'tp': tp_fp32, 'fp': fp_fp32, 'neg_img': neg_fp32}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
