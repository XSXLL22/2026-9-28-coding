"""Train YOLOv8n and compare 96/128 validation on a local, audited dataset."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from runtime.common import ROOT, environment, file_hash, write_json, configure_ultralytics

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))


def augmentation_state(dataset):
    """Inspect transform probabilities without invoking transforms or consuming RNG."""
    rows, visited = [], set()

    def walk(transform, path):
        if transform is None or id(transform) in visited:
            return
        visited.add(id(transform))
        name = type(transform).__name__
        if name in ('Mosaic', 'MixUp', 'CutMix', 'CopyPaste'):
            rows.append({'path': path, 'type': name, 'p': float(transform.p)})
        children = getattr(transform, 'transforms', None)
        if isinstance(children, (list, tuple)):
            for index, child in enumerate(children):
                walk(child, f'{path}.transforms[{index}]')
        walk(getattr(transform, 'pre_transform', None), f'{path}.pre_transform')

    walk(dataset.transforms, 'dataset.transforms')
    return rows


def augmentation_observer(path):
    """Record at first batch AFTER the framework's epoch-start close/reset action."""
    last_epoch = None

    def observe(trainer):
        nonlocal last_epoch
        epoch = int(trainer.epoch) + 1
        if epoch == last_epoch:
            return
        row = {'epoch': epoch, 'callback': 'on_train_batch_start',
               'configured_close_mosaic': int(trainer.args.close_mosaic),
               'transforms': augmentation_state(trainer.train_loader.dataset)}
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        last_epoch = epoch

    return observe


def audit_dataset(yaml_path):
    import yaml
    from PIL import Image
    from tools.validate_dataset import check_label
    from collections import Counter

    yaml_path = Path(yaml_path).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    base = Path(config.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    names = config["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names not in ({0: "smoke", 1: "fire"}, {0: "smoke", 1: "fire", 2: "leaf_pile"}):
        raise ValueError("Dataset must use canonical smoke/fire[/leaf_pile] names and order")
    errors, hashes, summary = [], {}, {}
    sample_hashes = []
    for split in ("train", "val", "test"):
        directory = base / config[split]
        if not directory.is_dir():
            raise ValueError(f"This entry point requires an image directory for {split}: {directory}")
        paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and p.is_file())
        if not paths:
            raise ValueError(f"Empty split: {split}")
        counts, negatives = Counter(), 0
        for path in paths:
            with Image.open(path) as image:
                image.verify()
            parts = list(path.parts)
            if "images" not in parts:
                raise ValueError(f"Missing images path component: {path}")
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            check_label(label, len(names), errors, counts)
            if not label.read_text(encoding="utf-8").strip():
                negatives += 1
            digest = file_hash(path)
            if digest in hashes and hashes[digest] != split:
                errors.append(f"Identical image crosses splits: {path}")
            hashes[digest] = split
            sample_hashes.append({"split": split, "image": str(path), "image_sha256": digest,
                                  "label_sha256": file_hash(label)})
        summary[split] = {"images": len(paths), "negative_images": negatives, "boxes_per_class": dict(counts)}
        if split in ("train", "val") and any(counts[class_id] == 0 for class_id in names):
            errors.append(f"Every configured class needs at least one annotated box in {split}")
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors[:20]))
    return {"yaml_sha256": file_hash(yaml_path), "names": names, "splits": summary,
            "files": sample_hashes, "limitation": "Exact duplicate check only; video groups and perceptual duplicates require separate review."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True, help="Existing local .pt file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, choices=(96, 128, 320), default=128)
    parser.add_argument("--eval-sizes", type=int, nargs="+", choices=(96, 128, 256, 320), default=[96, 128],
                        help="Validation sizes; 256/320 are software diagnostic only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--warmup-bias-lr", type=float, default=None,
                        help="Explicit bias warmup learning rate; omission preserves the library default")
    parser.add_argument("--close-mosaic", type=int, default=None,
                        help="Disable Mosaic during the final N epochs; omission preserves the project's 0")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.threads) <= 0:
        parser.error("epochs, batch, threads must be positive")
    if args.close_mosaic is not None:
        if not 0 <= args.close_mosaic <= args.epochs:
            parser.error("close-mosaic must be between 0 and epochs")
        if args.mode != 'train':
            parser.error("close-mosaic applies only to train")
    if args.warmup_bias_lr is not None:
        if not math.isfinite(args.warmup_bias_lr) or not 0 <= args.warmup_bias_lr <= 1:
            parser.error("warmup-bias-lr must be finite and in [0,1]")
        if args.mode != "train":
            parser.error("warmup-bias-lr applies only to train")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    configure_ultralytics()
    from ultralytics import YOLO
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "args": vars(args), "environment": environment(),
              "initial_weights_sha256": file_hash(weights), "purpose": "software_baseline_not_fpga_model",
              "entrypoint_sha256": file_hash(__file__)}
    write_json(output / "report.json", report)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    try:
        audit = audit_dataset(args.data)
        write_json(output / "data_audit.json", audit)
        model = YOLO(str(weights))
        if args.mode == "train":
            training_options = dict(data=str(Path(args.data).resolve()), imgsz=args.imgsz, epochs=args.epochs,
                        batch=args.batch, device=args.device, workers=0, seed=args.seed,
                        deterministic=True, amp=False, cache=False, plots=False,
                        project=str(output), name="fit", exist_ok=False, verbose=False,
                        optimizer="AdamW", lr0=0.001,
                        close_mosaic=args.close_mosaic if args.close_mosaic is not None else 0)
            if args.warmup_bias_lr is not None:
                training_options['warmup_bias_lr'] = args.warmup_bias_lr
            report['training_options'] = training_options
            write_json(output / 'report.json', report)
            model.add_callback('on_train_batch_start', augmentation_observer(output / 'augmentation_epochs.jsonl'))
            model.train(**training_options)
            best = Path(model.trainer.best)
            if not best.is_file():
                raise RuntimeError("Training returned without best.pt")
            report.update(best_weights=str(best), best_weights_sha256=file_hash(best))
            model = YOLO(str(best))
        if model.names != audit["names"]:
            raise ValueError(f"Evaluation class mismatch: model={model.names}, dataset={audit['names']}")
        results = {}
        for size in dict.fromkeys(args.eval_sizes):
            metrics = model.val(data=str(Path(args.data).resolve()), split="val", imgsz=size,
                                batch=args.batch, device=args.device, workers=0, plots=False,
                                conf=0.001, iou=0.7, half=False, rect=False,
                                project=str(output), name=f"val_{size}", verbose=False)
            per_class = {}
            for position, class_id in enumerate(metrics.box.ap_class_index):
                precision, recall, ap50, ap = metrics.box.class_result(position)
                per_class[model.names[int(class_id)]] = dict(precision=float(precision), recall=float(recall),
                                                          ap50=float(ap50), ap50_95=float(ap))
            results[str(size)] = {"aggregate": {key: float(value) for key, value in metrics.results_dict.items()},
                                  "per_class": per_class, "speed_ms_per_image": metrics.speed}
        report.update(status="complete", validation=results, duration_seconds=time.perf_counter() - started,
                      metric_note="Ultralytics precision/recall at its F1-selected confidence; AP sweeps confidence. Not fixed-threshold alarm metrics.",
                      unsupported_classes=["leaf_pile"] if len(audit["names"]) == 2 else [])
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "report.json", report)
    print(f"Saved baseline report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
