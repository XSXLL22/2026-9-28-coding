"""Offline YOLO inference for images, frame directories, and video files.

Run: python -m runtime.vision --weights model.pt --source video.mp4 --output experiments/run01
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from runtime.common import ROOT, class_mapping, environment, file_hash, write_json, configure_ultralytics

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}


def roi_contour(config, width, height):
    import cv2
    import numpy as np
    points = config["roi"]["polygon"]
    if len(points) < 3 or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in points):
        raise ValueError("ROI must contain at least 3 finite normalized points")
    contour = np.array([[x * width, y * height] for x, y in points], dtype=np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 0:
        raise ValueError("Baseline ROI must be a nondegenerate convex polygon")
    return contour


def roi_overlap(box, contour):
    import cv2
    import numpy as np
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    rectangle = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(rectangle, contour)
    return min(1.0, max(0.0, float(intersection) / area))


def read_image(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def frames(source, sequence_fps=None):
    import cv2
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGES and p.is_file())
        if not files:
            raise ValueError(f"No images in directory: {source}")
        for index, path in enumerate(files):
            timestamp = index * 1000.0 / sequence_fps if sequence_fps else None
            yield index, timestamp, "assumed_sequence_fps" if sequence_fps else "unavailable", path, read_image(path)
    elif source.suffix.lower() in IMAGES and source.is_file():
        yield 0, None, "unavailable", source, read_image(source)
    elif source.suffix.lower() in VIDEOS and source.is_file():
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        index, previous = 0, -1.0
        try:
            while True:
                valid, image = capture.read()
                if not valid:
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                basis = "container_timestamp"
                if not math.isfinite(timestamp) or timestamp < 0 or (index > 0 and timestamp <= previous):
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    timestamp = index * 1000 / fps if fps > 0 and math.isfinite(fps) else None
                    basis = "nominal_fps_estimate" if timestamp is not None else "unavailable"
                previous = timestamp if timestamp is not None else previous
                yield index, timestamp, basis, source, image
                index += 1
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if index == 0 or (expected > 0 and index < expected):
                raise ValueError(f"Video decode ended early: decoded {index}, declared {expected}")
        finally:
            capture.release()
    else:
        raise ValueError(f"Unsupported or missing input: {source}")


def run(args):
    import cv2
    import numpy as np
    import torch
    configure_ultralytics()
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "args": vars(args), "environment": environment()}
    write_json(output / "run.json", report)
    try:
        weights = Path(args.weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Weights must exist locally: {weights}")
        config_path = Path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        write_json(output / "config_snapshot.json", config)
        model = YOLO(str(weights))
        mapping = class_mapping(model.names, config.get("model_class_map"))
        if not args.allow_unmapped_model and not any(v is not None for v in mapping.values()):
            raise ValueError("Model has no project classes. Use a trained smoke/fire model, or explicitly --allow-unmapped-model for plumbing tests.")
        model_hash = file_hash(weights)
        report.update(model_sha256=model_hash, model_names=model.names, class_mapping=mapping,
                      unsupported_project_classes=sorted({0, 1, 2} - set(mapping.values())),
                      config_sha256=file_hash(config_path))
        # Warm-up is excluded from per-frame timing, retained in total run wall time.
        model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8), imgsz=args.imgsz,
                      device=args.device, verbose=False, rect=False)
        count, inference_ms, frame_ms = 0, [], []
        source_hashes = {}
        iterator = iter(frames(args.source, args.sequence_fps))
        with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            while args.max_frames is None or count < args.max_frames:
                frame_start = time.perf_counter()
                try:
                    index, timestamp, basis, path, frame = next(iterator)
                except StopIteration:
                    break
                if str(path) not in source_hashes:
                    source_hashes[str(path)] = file_hash(path)
                height, width = frame.shape[:2]
                contour = roi_contour(config, width, height)
                infer_start = time.perf_counter()
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                       device=args.device, verbose=False, rect=False)[0]
                elapsed = (time.perf_counter() - infer_start) * 1000
                boxes = result.boxes.cpu()
                detections = []
                for xyxy, score, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                    class_id = int(class_id)
                    detections.append({"model_class_id": class_id, "model_class_name": model.names[class_id],
                                       "project_class_id": mapping[class_id], "confidence": score,
                                       "xyxy_original_pixels": xyxy, "roi_box_area_fraction": roi_overlap(xyxy, contour)})
                record = {"schema_version": 1, "node_id": config["node_id"], "frame_id": index,
                          "source": str(path), "source_timestamp_ms": timestamp, "timestamp_basis": basis,
                          "image_width": width, "image_height": height, "model_sha256": model_hash,
                          "detections": detections, "predict_wall_ms": elapsed, "model_speed_ms": result.speed}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if args.save_images:
                    canvas = result.plot()
                    cv2.polylines(canvas, [contour.astype(np.int32)], True, (0, 255, 255), 2)
                    success, encoded = cv2.imencode(".jpg", canvas)
                    if not success:
                        raise RuntimeError("Annotated image encoding failed")
                    encoded.tofile(str(output / f"frame_{index:06d}.jpg"))
                inference_ms.append(elapsed)
                frame_ms.append((time.perf_counter() - frame_start) * 1000)
                count += 1
        iterator.close()
        if count == 0:
            raise ValueError("Input produced no frames")
        report.update(status="complete", frame_count=count, input_hashes=source_hashes,
                      predict_mean_ms=float(np.mean(inference_ms)), predict_p95_ms=float(np.percentile(inference_ms, 95)),
                      frame_loop_mean_ms=float(np.mean(frame_ms)), frame_loop_fps=1000 / float(np.mean(frame_ms)),
                      total_wall_seconds=time.perf_counter() - started,
                      timing_note="Loop includes read, first-use input hashing, prediction, ROI, serialization, optional JPEG; excludes model loading and warm-up.")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "run.json", report)
    print(json.dumps({"output": str(output), "frames": count, "predict_mean_ms": report["predict_mean_ms"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "vision.json"))
    parser.add_argument("--imgsz", type=int, choices=(96, 128, 256, 320), default=128,
                        help="256/320 are software diagnostic size, not the FPGA target")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--sequence-fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--allow-unmapped-model", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("conf and iou must be in [0,1]")
    if args.sequence_fps is not None and (not math.isfinite(args.sequence_fps) or args.sequence_fps <= 0):
        parser.error("sequence-fps must be finite and positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    if args.threads <= 0:
        parser.error("threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
