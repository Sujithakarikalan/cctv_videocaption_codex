"""Report Python and core dependency/device availability without downloading weights."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import sys
from pathlib import Path

import cv2
import torch
import torchvision
import yaml

# Running ``python scripts/verify_environment.py`` puts ``scripts/`` rather than
# the project root on sys.path. Add the root so the script works without install -e.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required.")
        return 1
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PyYAML: {_version('PyYAML')}")
    print(f"Ultralytics: {_version('ultralytics')}")
    print(f"Transformers: {_version('transformers')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Selected device (auto): {get_device('auto')}")
    if cv2.VideoCapture:
        print("OpenCV VideoCapture API: available (codec support depends on this build).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
