"""Shared provenance and explicit project class mapping."""
from __future__ import annotations

import hashlib
import json
import platform
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"smoke": 0, "fire": 1, "leaf_pile": 2}


def restore_windows_architecture():
    """Restore missing Windows process metadata from the OS, keeping CPU checks active.

    Restricted launchers may omit PROCESSOR_ARCHITECTURE. Python then reports an
    empty machine string, causing Polars to skip CPUID and reject every feature.
    Never guess from Python bitness (which cannot distinguish ARM64 and AMD64).
    """
    if os.name != 'nt' or os.environ.get('PROCESSOR_ARCHITECTURE'):
        return
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsWow64Process2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
    kernel.IsWow64Process2.restype = ctypes.c_int
    process, native = ctypes.c_ushort(), ctypes.c_ushort()
    if not kernel.IsWow64Process2(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
        raise ctypes.WinError(ctypes.get_last_error())
    machine = process.value or native.value
    names = {0x8664: 'AMD64', 0x014c: 'x86', 0xaa64: 'ARM64'}
    if machine not in names:
        raise RuntimeError(f'Unknown Windows process machine: {machine:#x}')
    os.environ['PROCESSOR_ARCHITECTURE'] = names[machine]
    # platform.uname() caches results, possibly including the earlier empty value.
    if hasattr(platform, 'invalidate_caches'):
        platform.invalidate_caches()
    else:
        platform._uname_cache = None


def configure_ultralytics():
    restore_windows_architecture()
    directory = ROOT / ".cache" / "ultralytics"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(directory)
    from ultralytics import settings
    settings.update({"sync": False})


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def environment():
    restore_windows_architecture()
    import torch
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python")},
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def class_mapping(names, configured=None):
    """Never reinterpret a COCO class ID as a project class ID."""
    mapping = {}
    for source_id, name in names.items():
        target = (configured or {}).get(name, name.lower())
        mapping[int(source_id)] = CLASSES.get(target)
    return mapping
