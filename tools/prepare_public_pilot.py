"""Create an auditable training copy; clip finite valid boxes to visible image bounds."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import file_hash, write_json


def clip_yolo_row(line):
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("Expected five YOLO fields")
    class_id = int(fields[0])
    x, y, width, height = map(float, fields[1:])
    if class_id not in (0, 1) or not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError("Invalid class or non-finite coordinates")
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("Invalid normalized coordinates; not auto-repairable")
    bounds = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
    clipped = tuple(min(1.0, max(0.0, v)) for v in bounds)
    if bounds == clipped:
        return line, False
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box has no visible area")
    return f"{class_id} {(x1+x2)/2:.12g} {(y1+y2)/2:.12g} {x2-x1:.12g} {y2-y1:.12g}", True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "datasets" / "public_pilot"))
    parser.add_argument("--output", default=str(ROOT / "datasets" / "public_pilot_prepared"))
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_provenance_sha256": file_hash(source / "provenance.json"),
              "policy": "clip boxes to image bounds; preserve raw source files and splits", "changes": [], "files": []}
    try:
        for split, names in provenance["selection"].items():
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
            for name in names:
                image = source / "images" / split / Path(name).name
                label = source / "labels" / split / (image.stem + ".txt")
                target_image = output / "images" / split / image.name
                target_label = output / "labels" / split / label.name
                shutil.copy2(image, target_image)
                rows = []
                for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    cleaned, changed = clip_yolo_row(line)
                    rows.append(cleaned)
                    if changed:
                        report["changes"].append({"label": str(label), "line": number, "before": line, "after": cleaned})
                target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                report["files"].append({"image": str(target_image), "sha256": file_hash(target_image),
                                        "source_label_sha256": file_hash(label), "label_sha256": file_hash(target_label)})
        (output / "data.yaml").write_text("path: " + json.dumps(str(output).replace("\\", "/")) +
                                           "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: smoke\n  1: fire\n", encoding="utf-8")
        from training.baseline import audit_dataset
        audit = audit_dataset(output / "data.yaml")
        write_json(output / "audit.json", audit)
        report.update(status="complete", splits=audit["splits"])
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "preparation.json", report)
    print(json.dumps({"clipped_boxes": len(report["changes"]), "splits": report["splits"]}))


if __name__ == "__main__":
    main()
