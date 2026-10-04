import numpy as np
import pytest
import torch
from torch import nn

from src.models.slowfast_action_recognizer import (
    ActionPrediction,
    SlowFastActionRecognizer,
    pack_slowfast_pathways,
    preprocess_frames,
)


def test_preprocess_frames_and_pathways():
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(32)]
    video = preprocess_frames(frames)
    assert video.shape == (1, 3, 32, 256, 256)
    slow, fast = pack_slowfast_pathways(video, alpha=4)
    assert slow.shape == (1, 3, 8, 256, 256)
    assert fast.shape == video.shape


def test_classifier_returns_labelled_softmax_topk(tmp_path):
    class ConstantModel(nn.Module):
        def forward(self, pathways):
            assert len(pathways) == 2
            return torch.tensor([[0.0, 2.0, 1.0]])

    recognizer = SlowFastActionRecognizer(
        model=ConstantModel(), labels=["walk", "fight", "run"], device="cpu", project_root=tmp_path
    )
    outputs = recognizer.predict_tensor(torch.zeros((1, 3, 32, 16, 16)), top_k=2)
    assert [item.label for item in outputs] == ["fight", "run"]
    assert all(isinstance(item, ActionPrediction) for item in outputs)
    assert outputs[0].confidence > outputs[1].confidence
    assert sum(item.confidence for item in outputs) < 1.0


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        preprocess_frames([])
    with pytest.raises(ValueError):
        pack_slowfast_pathways(torch.zeros((3, 32, 10, 10)))
