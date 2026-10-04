"""Phase 1 tests for configuration loading and device selection."""

from pathlib import Path

import pytest
import torch

from src.utils.config import load_config
from src.utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_base_configuration_loads() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    assert config["project"]["seed"] == 42
    assert config["models"]["temporal"]["architecture"] == "slowfast_r50"


def test_missing_configuration_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_device_cpu_is_deterministic() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_auto_device_matches_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("auto").type == expected


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        get_device("mps")
