"""Fixed-threshold image-level detection errors, distinct from event alarms."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from runtime.common import write_json


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_details(predictions, targets, threshold):
    """Return original prediction/target indices using the shared greedy policy."""
    used = set()
    pairs, false_positives = [], []
    for prediction_index, prediction in sorted(enumerate(predictions), key=lambda item: item[1]["confidence"], reverse=True):
        options = [(iou(prediction["xyxy_original_pixels"], target), index) for index, target in enumerate(targets) if index not in used]
        score, index = max(options, default=(0.0, -1))
        if index >= 0 and score >= threshold:
            used.add(index)
            pairs.append({"prediction_index": prediction_index, "target_index": index, "iou": score})
        else:
            false_positives.append(prediction_index)
    return {"pairs": pairs, "fp_indices": false_positives,
            "fn_indices": [index for index in range(len(targets)) if index not in used]}


def match(predictions, targets, threshold):
    details = match_details(predictions, targets, threshold)
    return {"tp": len(details["pairs"]), "fp": len(details["fp_indices"]), "fn": len(details["fn_indices"])}


def analyze(predictions_path, labels_directory, threshold=0.5):
    totals = {name: Counter() for name in ("smoke", "fire", "leaf_pile")}
    classes = list(totals)
    images, negative_images, negative_with_detection = 0, 0, 0
    seen = set()
    for line in Path(predictions_path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        source = Path(record["source"])
        if source.name in seen:
            raise ValueError("Image analysis does not accept repeated filenames or video frames")
        seen.add(source.name)
        label = Path(labels_directory) / (source.stem + ".txt")
        targets = {name: [] for name in classes}
        for row in label.read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            c, x, y, w, h = map(float, row.split())
            width, height = record["image_width"], record["image_height"]
            targets[classes[int(c)]].append([(x - w / 2) * width, (y - h / 2) * height,
                                             (x + w / 2) * width, (y + h / 2) * height])
        project_predictions = [p for p in record["detections"] if p["project_class_id"] is not None]
        for index, name in enumerate(classes):
            totals[name].update(match([p for p in project_predictions if p["project_class_id"] == index], targets[name], threshold))
        if not any(targets.values()):
            negative_images += 1
            negative_with_detection += bool(project_predictions)
        images += 1
    if not images:
        raise ValueError("Empty predictions file")
    result = {}
    for name, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        result[name] = dict(values, precision=tp / (tp + fp) if tp + fp else None,
                            recall=tp / (tp + fn) if tp + fn else None, ground_truth_boxes=tp + fn)
    return {"images": images, "iou_threshold": threshold, "per_class": result,
            "negative_images": negative_images, "negative_images_with_detection": negative_with_detection,
            "note": "Uses confidence-filtered JSONL; null means undefined. Not event-level fire-alarm rates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, analyze(args.predictions, args.labels))


if __name__ == "__main__":
    main()
