"""Download a small, versioned D-Fire pilot using the publisher-linked Kaggle mirror.

This is a pipeline pilot, not a leakage-audited research benchmark.
Existing upstream splits are preserved. No campus or leaf-pile data is fabricated.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import threading
import urllib.parse
import urllib.request
import zipfile
import io
import shutil
from pathlib import Path

BASE = "https://www.kaggle.com/api/v1/datasets/"
DATASET = "sayedgamal99/smoke-fire-detection-yolo"
ROOT = Path(__file__).resolve().parents[1]
LOCAL = threading.local()


def get(url):
    import requests
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(6):
        try:
            response = LOCAL.session.get(url, timeout=(20, 60))
            response.raise_for_status()
            return response.content
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))


def download(name, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = BASE + "download/" + DATASET + "/" + urllib.parse.quote(name, safe="") + "?datasetVersionNumber=1"
        data = get(url)
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                files = [item for item in archive.infolist() if not item.is_dir()]
                if len(files) != 1:
                    raise ValueError(f"Unexpected archive for {name}")
                data = archive.read(files[0])
        if data.lstrip().startswith((b"<!DOCTYPE", b"<html")):
            raise ValueError(f"Server returned HTML instead of {name}")
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return {"upstream_path": name, "local_path": destination.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--val", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "public_pilot")
    parser.add_argument("--reuse", type=Path, help="Reuse verified files and catalog from an earlier completed download")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)  # provenance uses project-relative paths
    output.mkdir(parents=True, exist_ok=True)
    selection_config = {"dataset": DATASET, "version": 1, "counts": {s: getattr(args, s) for s in ("train", "val", "test")}}
    previous_path = output / "provenance.json"
    if previous_path.exists():
        completed = json.loads(previous_path.read_text(encoding="utf-8"))
        if {s: len(completed["selection"][s]) for s in ("train", "val", "test")} != selection_config["counts"]:
            raise ValueError("Use a new output directory for different sample counts")
        for record in completed["files"]:
            if hashlib.sha256((ROOT / record["local_path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Completed download changed: {record['local_path']}")
    config_path = output / "download_request.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != selection_config:
        raise ValueError("Cannot change selection counts in an existing download directory")
    config_path.write_text(json.dumps(selection_config, indent=2), encoding="utf-8")
    reusable = {}
    if args.reuse:
        reuse = args.reuse.resolve()
        previous_manifest = json.loads((reuse / "provenance.json").read_text(encoding="utf-8"))
        if previous_manifest["dataset"] != DATASET or previous_manifest["version"] != 1:
            raise ValueError("Reuse dataset/version mismatch")
        reusable = {row["upstream_path"]: row for row in previous_manifest["files"]}
        if not (output / "catalog.json").exists():
            shutil.copy2(reuse / "catalog.json", output / "catalog.json")
    metadata = json.loads(get(BASE + "view/" + DATASET))
    (output / "upstream_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # Persist the original class declaration as provenance before making a local YAML.
    download("data.yaml", output / "upstream_data.yaml")
    catalog_path = output / "catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    else:
        checkpoint = output / "catalog_partial.json"
        previous = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
        catalog, token = previous.get("files", []), previous.get("next_token", "")
        for page in range(500):
            query = urllib.parse.urlencode({"datasetVersionNumber": 1, "pageSize": 1000, "pageToken": token})
            response = json.loads(get(BASE + "list/" + DATASET + "?" + query))
            catalog.extend(item["name"] for item in response.get("datasetFiles", []))
            token = response.get("nextPageToken", "")
            checkpoint.write_text(json.dumps({"files": catalog, "next_token": token}), encoding="utf-8")
            print(f"catalog page {page + 1}: {len(catalog)} files", flush=True)
            if not token:
                break
        if token:
            raise RuntimeError("Incomplete catalog; stopped at page limit")
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    catalog_set = set(catalog)
    jobs, selected = [], {}
    for split in ("train", "val", "test"):
        candidates = [name for name in catalog if f"/{split}/images/" in name and name.lower().endswith((".jpg", ".png", ".jpeg"))]
        # Fixed hash order spreads selected files over filenames, unlike first-N frames.
        candidates.sort(key=lambda name: hashlib.sha256(("p2-pilot-v1:" + name).encode()).hexdigest())
        count = getattr(args, split)
        if count <= 0 or len(candidates) < count:
            raise ValueError(f"Insufficient files or invalid count in {split}")
        selected[split] = candidates[:count]
        for name in selected[split]:
            label = str(Path(name.replace("/images/", "/labels/")).with_suffix(".txt")).replace("\\", "/")
            if label not in catalog_set:
                raise ValueError(f"Missing upstream label: {label}")
            jobs.extend([(name, output / "images" / split / Path(name).name),
                         (label, output / "labels" / split / Path(label).name)])
    for name, destination in jobs:
        if not destination.exists() and name in reusable:
            record = reusable[name]
            cached = ROOT / record["local_path"]
            if hashlib.sha256(cached.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Reusable source hash mismatch: {cached}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, destination)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, name, dest) for name, dest in jobs]
        records = []
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records) % 20 == 0:
                print(f"downloaded {len(records)}/{len(jobs)}", flush=True)
    manifest = {"dataset": DATASET, "version": 1, "license": metadata.get("licenseName"),
                "selection": selected, "files": sorted(records, key=lambda x: x["upstream_path"]),
                "purpose": "pipeline_pilot_only", "leakage_audit": "upstream groups unknown; near-duplicate audit pending"}
    (output / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data.yaml").write_text("path: " + json.dumps(str(output).replace("\\", "/")) + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: smoke\n  1: fire\n", encoding="utf-8")
    print(f"Pilot ready: {output}", flush=True)


if __name__ == "__main__":
    main()
