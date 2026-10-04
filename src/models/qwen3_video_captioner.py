"""Inference-only Qwen3-VL video caption generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
CAPTION_SYSTEM_PROMPT = (
    "You describe CCTV footage for a factual video captioning system. "
    "Use the video frames as the primary and authoritative evidence. Describe only people, objects, "
    "actions, movement, and scene context that are visually supported. Do not guess intentions, "
    "emotions, crimes, violence, abnormality, or events that are not clearly visible. Supporting "
    "YOLO object detections may be incomplete or wrong; SlowFast action labels are weak suggestions "
    "and may be wrong. Ignore any supporting hint that conflicts with the video. Do not convert "
    "repeated detections across frames into a count of distinct people or objects. Prefer a simple "
    "correct description over a detailed uncertain one. Return one concise natural-language caption."
)


@dataclass(frozen=True)
class CaptionGeneration:
    caption: str
    model_name: str
    device: str
    num_frames: int
    max_pixels: int
    inference_seconds: float


def resolve_caption_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda without silently changing model or explicit device requests."""
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch runtime has no CUDA device. "
                "Use a CUDA-enabled Colab/runtime or pass --device cpu explicitly."
            )
        return torch.device("cuda:0")
    raise ValueError("device must be one of: auto, cpu, cuda")


def build_caption_messages(
    video_path: str | Path,
    supporting_text: str = "",
    max_pixels: int = 151_200,
) -> list[dict[str, Any]]:
    """Build Qwen's native video message; the source video remains the visual source of truth."""
    path = Path(video_path).expanduser().resolve()
    user_text = "Write one concise CCTV caption describing what is visibly happening in this video."
    if supporting_text.strip():
        user_text += "\n\nSupporting hints (may be incomplete or wrong; verify against the video):\n" + supporting_text
    return [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": path.as_uri(),
                    "max_pixels": int(max_pixels),
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]


class Qwen3VideoCaptioner:
    """Load the pretrained Qwen3-VL-4B-Instruct model and produce a single caption."""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = MODEL_NAME,
        model: Any | None = None,
        processor: Any | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.device = resolve_caption_device(device)
        self.model_name = model_name
        if (model is None) != (processor is None):
            raise ValueError("Inject both model and processor together, or neither")
        if model is None:
            model, processor = self._load_pretrained(cache_dir)
        self.model = model
        self.processor = processor

    def _load_pretrained(self, cache_dir: str | Path | None) -> tuple[Any, Any]:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Qwen3-VL requires transformers>=4.57.0. Install the project caption dependencies "
                "with `python -m pip install -r requirements.txt`."
            ) from exc

        if self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            device_map: Mapping[str, str] = {"": str(self.device)}
        else:
            # Explicit CPU mode is supported for capable hosts, but can be very slow and RAM intensive.
            dtype = torch.float32
            device_map = {"": "cpu"}
        kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device_map}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs).eval()
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    f"Could not load {self.model_name} on {self.device}: insufficient memory. "
                    "Use a supported GPU runtime such as the confirmed Colab/T4 setup."
                ) from exc
            raise RuntimeError(f"Could not load pretrained caption model {self.model_name}: {exc}") from exc
        return model, processor

    @torch.inference_mode()
    def caption(
        self,
        video_path: str | Path,
        num_frames: int = 16,
        max_pixels: int = 151_200,
        max_new_tokens: int = 128,
        supporting_text: str = "",
    ) -> CaptionGeneration:
        """Run Qwen's native video processor with uniform num_frames sampling over the source clip."""
        if num_frames < 1 or max_pixels < 1 or max_new_tokens < 1:
            raise ValueError("num_frames, max_pixels, and max_new_tokens must all be positive")
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        messages = build_caption_messages(path, supporting_text=supporting_text, max_pixels=max_pixels)
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                num_frames=num_frames,
                fps=None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-VL video preprocessing failed. Check that transformers>=4.57 and qwen-vl-utils "
                "are installed and that the video can be decoded."
            ) from exc
        inputs = inputs.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        if start is not None:
            start.record()
        import time

        wall_start = time.perf_counter()
        try:
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    f"Qwen3-VL ran out of memory on {self.device}. Reduce --num-frames or --max-pixels, "
                    "or use a larger-memory GPU."
                ) from exc
            raise RuntimeError(f"Qwen3-VL caption generation failed on {self.device}: {exc}") from exc
        if end is not None:
            end.record()
            torch.cuda.synchronize(self.device)
            inference_seconds = float(start.elapsed_time(end) / 1000.0)
        else:
            inference_seconds = float(time.perf_counter() - wall_start)
        prompt_ids = inputs["input_ids"]
        generated_only = [
            output_ids[prompt_ids.shape[-1]:] for output_ids in generated
        ]
        decoded = self.processor.batch_decode(
            generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        caption = decoded[0].strip() if decoded else ""
        if not caption:
            raise RuntimeError("Qwen3-VL returned an empty caption")
        return CaptionGeneration(
            caption=caption,
            model_name=self.model_name,
            device=str(self.device),
            num_frames=num_frames,
            max_pixels=max_pixels,
            inference_seconds=inference_seconds,
        )
