"""Compute-device selection with explicit CPU fallback behavior."""

import logging

import torch

LOGGER = logging.getLogger(__name__)


def get_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a PyTorch device.

    Explicit CUDA requests fail with an actionable message if CUDA is unavailable;
    automatic selection warns and falls back to CPU.
    """
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("CUDA is unavailable; using CPU. Video-model inference may be slow.")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in the installed PyTorch build.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device '{requested}'. Expected one of: auto, cpu, cuda.")
