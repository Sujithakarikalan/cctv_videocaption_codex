"""Phase 2 visual encoder tests; all use random weights and perform no downloads."""

import pytest
import torch

from src.features.semantic_keyframes import cosine_similarity
from src.models.resnet18_encoder import ResNet18Encoder


def test_resnet18_returns_l2_normalized_512_dimensional_features() -> None:
    encoder = ResNet18Encoder(pretrained=False).eval()
    with torch.inference_mode():
        features = encoder(torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8))
    assert features.shape == (2, 512)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-5)


def test_cosine_similarity_matches_expected_values() -> None:
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == pytest.approx(-1.0)
