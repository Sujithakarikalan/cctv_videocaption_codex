import pytest
import torch

from src.features.caption_evidence import (
    format_caption_support,
    matches_video_path,
    select_slowfast_hints,
    summarize_yolo_hints,
)
from src.models.qwen3_video_captioner import (
    MODEL_NAME,
    Qwen3VideoCaptioner,
    build_caption_messages,
    resolve_caption_device,
)


def test_resolve_cuda_requires_available_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_caption_device("auto").type == "cpu"
    assert resolve_caption_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_caption_device("cuda")


def test_yolo_evidence_does_not_claim_unique_counts():
    frames = [
        {"detections": [{"label": "person", "confidence": 0.9}, {"label": "kite", "confidence": 0.34}]},
        {"detections": [{"label": "person", "confidence": 0.8}]},
    ]
    hints, metadata = summarize_yolo_hints(frames, min_confidence=0.5)
    assert len(hints) == 1
    assert hints[0]["label"] == "person"
    assert hints[0]["detection_hits_across_frames"] == 2
    assert hints[0]["unique_object_count"] is None
    assert "not a unique-object count" in hints[0]["interpretation"]
    assert metadata["detections_below_confidence_threshold"] == 1


def test_slowfast_hints_are_opt_in_and_weakly_marked():
    result = {"slowfast_action_clips": [{"start_seconds": 0, "end_seconds": 2,
               "predictions": [{"label": "robot dancing", "confidence": 0.95}]}]}
    omitted, omitted_meta = select_slowfast_hints(result, enabled=False)
    assert omitted == []
    assert omitted_meta["usage_status"] == "not requested"
    provided, meta = select_slowfast_hints(result, enabled=True)
    assert provided[0]["label"] == "robot dancing"
    assert "cannot be verified" in meta["usage_status"]
    assert "never abnormality" in meta["score_interpretation"]
    text = format_caption_support([], provided)
    assert "weak, possibly incorrect suggestion" in text


def test_qwen_messages_make_video_primary(tmp_path):
    video = tmp_path / "cctv.mp4"
    video.touch()
    messages = build_caption_messages(video, "- person: seen in sampled frames", max_pixels=1000)
    assert messages[0]["role"] == "system"
    user_video = messages[1]["content"][0]
    assert user_video["type"] == "video"
    assert user_video["video"].startswith("file:")
    user_prompt = messages[1]["content"][1]["text"]
    assert "verify against the video" in user_prompt
    assert "visually supported" in messages[0]["content"]


def test_evidence_results_match_when_project_moved_between_hosts(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    assert matches_video_path(r"C:\old-machine\project\clip.mp4", video)
    assert not matches_video_path(r"C:\old-machine\project\other.mp4", video)


def test_mock_qwen_generation_uses_uniform_num_frames(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder; decoder is mocked")

    class Batch(dict):
        def __init__(self):
            super().__init__({"input_ids": torch.tensor([[1, 2, 3]])})

        def to(self, _device):
            return self

    class Processor:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return Batch()

        def batch_decode(self, sequences, **kwargs):
            assert sequences[0].tolist() == [4, 5]
            return ["A person walks through a public area."]

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

    processor = Processor()
    captioner = Qwen3VideoCaptioner(
        device="cpu", model_name=MODEL_NAME, model=Model(), processor=processor
    )
    output = captioner.caption(video, num_frames=8, supporting_text="person visible")
    assert output.caption == "A person walks through a public area."
    assert output.model_name == MODEL_NAME
    assert output.device == "cpu"
    assert output.num_frames == 8
    assert processor.kwargs["num_frames"] == 8
    assert processor.kwargs["fps"] is None
    assert processor.kwargs["return_tensors"] == "pt"
