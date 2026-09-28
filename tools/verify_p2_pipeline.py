"""Run real model inference through all offline input modes and check repeatability."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import write_json
from runtime.vision import read_image


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if len(files) < 3:
        raise ValueError("At least 3 real images required")
    # This short video is a generated input-mode fixture, not a real fire sequence.
    video = output / "fixture_from_stills.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    for path in files[:3]:
        writer.write(cv2.resize(read_image(path), (320, 240)))
    writer.release()
    commands = []
    for name, source, size in (("image", files[0], 128), ("frames128", Path(args.images), 128),
                               ("repeat128", Path(args.images), 128), ("frames96", Path(args.images), 96),
                               ("video", video, 128)):
        command = [sys.executable, "-m", "runtime.vision", "--weights", str(Path(args.weights).resolve()),
                   "--source", str(source.resolve()), "--output", str(output / name), "--imgsz", str(size), "--device", "cpu"]
        if name == "image":
            command.append("--save-images")
        result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
        (output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        commands.append(command)
        if result.returncode:
            raise RuntimeError(f"Input mode {name} failed; see log")
        print(f"PASS {name}", flush=True)
    def detections(name):
        return [json.loads(line)["detections"] for line in (output / name / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
    first_detections = detections("frames128")
    if first_detections != detections("repeat128"):
        raise AssertionError("Repeatability failed: detections differ")
    from training.error_analysis import analyze
    for size in (96, 128):
        write_json(output / f"errors{size}.json", analyze(output / f"frames{size}" / "detections.jsonl", args.labels))
    records = [json.loads(line) for line in (output / "video" / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 3 or abs(records[-1]["source_timestamp_ms"] - 400) > 1:
        raise AssertionError("Video frame count or timestamp mismatch")
    write_json(output / "verification.json", {"status": "passed", "input_modes": ["image", "frame_directory", "video"],
                                               "repeat_detections_exact": True,
                                               "repeat_detection_count": sum(map(len, first_detections)),
                                               "repeat_note": "Equality of empty detections alone does not validate nonempty box stability.",
                                               "commands": commands,
                                               "video_note": "Three resized still images, input-mode fixture only; not temporal validation."})


if __name__ == "__main__":
    main()
