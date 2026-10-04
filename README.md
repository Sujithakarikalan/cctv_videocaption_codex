# Deep Learning-Based CCTV Video Captioning for Intelligent Surveillance

This repository is being implemented in phases. **Phases 1 and 2 are implemented**: environment/configuration setup and motion-aware video keyframe feature extraction. Detection, temporal modeling, caption training, and full inference are later phases and are not implemented yet.

## Phase 1 architecture choices and data requirements

The intended system is modular so each task can be trained and evaluated against the right labels:

| Module | Phase 1 design choice | Required supervision / weights |
|---|---|---|
| Keyframe visual encoder | ImageNet-pretrained ResNet18 with its classifier removed; frozen by default | ImageNet weights; optional later fine-tuning |
| Object detector | Ultralytics YOLOv8n pretrained on COCO as baseline; optional CCTV fine-tuning | COCO weights for baseline; YOLO-format images and bounding boxes for fine-tuning |
| Temporal encoder | Kinetics-pretrained SlowFast-R50; explicit warning if an optional fallback is used | Pretrained Kinetics weights; labeled video clips for action/event fine-tuning |
| Normal/abnormal head | Optional binary classifier on temporal embeddings | Separate normal/abnormal video labels; never substitutes for captions |
| Caption generator | Multimodal Transformer encoder-decoder with a pretrained tokenizer (initially T5 tokenizer); free-form output | Human-written paired video/caption annotations, with video-level split isolation |

The caption model must learn language-to-video correspondence from paired captions. Detection labels, event labels, and generated pseudo-captions are not substitutes for human caption ground truth. Use distinct manifests for each task. Split by original video identity before making clips or multiple-caption rows, so a video's frames/captions cannot leak across train, validation, and test.

Planned caption manifest columns are `video_id,video_path,caption,split`; multiple captions may repeat a `video_id`, but all rows for that ID must use the same split. The planned temporal CSV manifest is `video_path,label,split`. Detection data follows the standard YOLO layout (`images/{train,val,test}`, `labels/{train,val,test}`, and `dataset.yaml`). Keep human captions distinct from any `pseudo_caption` field, and report caption metrics only on held-out videos with human references.

## Environment setup (Phase 1)

Use Python 3.10 or newer. From the project root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch and torchvision pair appropriate for your OS and CUDA version using the selector at [pytorch.org](https://pytorch.org/get-started/locally/). For CPU-only installs, select CPU there. Then install the remaining project dependencies:

```powershell
python -m pip install -r requirements.txt
```

PyTorchVideo/SlowFast support is optional and intentionally not a required Phase 1 install: its dependency stack can be platform-sensitive. Install a compatible PyTorchVideo build only when implementing/running the temporal phase; the temporal module must report clearly if unavailable. Pretrained model downloads are also not performed by the environment check.

Verify core imports and device detection:

```powershell
python scripts/verify_environment.py
pytest -q tests/test_config_device.py
```

The verification script prints Python, PyTorch, torchvision, OpenCV, YAML, Ultralytics, and Transformers versions, CUDA availability, and the automatically selected device. CUDA unavailable is acceptable; automatic mode warns and selects CPU. Video codec availability depends on the installed OpenCV build and the input file.

## Project layout

`configs/` stores YAML settings. `data/raw/` is for original videos/datasets; processed videos and sampled frames belong in `data/processed/`; manifests and human annotations belong in `data/manifests/` and `data/annotations/`; cached features belong in `data/features/`. Model checkpoints go in `checkpoints/`, while logs, metrics, predictions, visualizations, and run artifacts go under `outputs/`. Large data and generated model artifacts are ignored by Git.

`configs/base.yaml` defines shared relative paths and initial model/training defaults. Paths are interpreted from the project root by later pipeline code; there are no machine-specific absolute paths in the config.

## Phase 2: video reading and semantic keyframes

OpenCV reads video metadata (FPS, frame count, duration, dimensions, and codec) and sequentially decodes frames at `data.sample_fps`, including the final decoded frame. A normalized grayscale frame-difference score filters low-change samples while retaining the first and last valid frames and at least one strongest interior motion sample. Pretrained Torchvision ResNet18 has its final `fc` layer replaced with `Identity`, producing 512-dimensional embeddings. ImageNet normalization and L2 feature normalization are applied before cosine comparison. The first and last valid frames are retained; intermediate candidates are retained when their motion score reaches `data.motion_threshold` OR their cosine similarity to the previous retained frame is below `data.semantic_similarity_threshold`, subject to `max_keyframes` and the configured minimum gap for semantic novelty.

Selected JPEG frames, a contact sheet, and `keyframes.json` are written under `data/processed/keyframes/<video-id>/`. Per-video features are stored under `data/features/resnet18/` in schema version 2 `.pt` cache files. Metadata records `encoder_name`, feature dimension, pretrained setting, weights and preprocessing, thresholds, source path/hash, and schema version. Old or incompatible caches are rejected and recomputed. Pretrained weights are stored under the project-local `checkpoints/model_cache/`. The `--no-pretrained` switch exists only for smoke tests; random ResNet18 embeddings are not valid for semantic keyframe selection.

To process one MP4 in PowerShell (from the project root):

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/extract_keyframes.py .\data\raw\your_video.mp4 --no-pretrained
```

For realistic semantic features after approving the model-weight download, omit `--no-pretrained`:

```powershell
python scripts/extract_keyframes.py .\data\raw\your_video.mp4
```

Expected output includes video ID and duration, selected feature shape `(keyframes, 512)`, artifact paths, and cache-hit status. Run the same command again to verify cache reuse. The pretrained command downloads Torchvision's official ResNet18 weights on first use if not already cached; it does not download weights during tests.

Phase 2 unit tests:

```powershell
pytest -q tests/test_keyframes.py tests/test_models.py
```

The tests create a tiny MP4 locally; they skip video-dependent checks if the OpenCV build lacks an MP4V encoder. To check against a real MP4, place it under `data/raw/` (or pass its path directly) and use the command above.

## Phase 3: YOLOv8 detection (optional fine-tuning)

The detector uses Ultralytics YOLOv8n COCO weights (`yolov8n.pt`) for the pretrained baseline; running detection does not require training. `YOLODetectionExtractor` accepts OpenCV BGR frames or saved keyframe paths and returns class IDs/labels, confidence, normalized `xyxy` boxes, and optional track IDs when tracking is enabled. `aggregate_object_features` summarizes per-frame counts, class counts, confidence, mean boxes, and temporal persistence; `ObjectFeatureEncoder` maps those summaries to a learned feature vector for later multimodal fusion.

For optional CCTV fine-tuning, prepare this structure and define `names` (and optionally `nc`) in `dataset.yaml`:

```text
dataset/
  dataset.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
```

Each image needs a same-relative-path `.txt` label (an empty file is valid for an image with no objects). Each non-empty label row is `class_id x_center y_center width height`, with integer class ID and normalized coordinates in `[0,1]`; boxes must have positive dimensions and stay within the image. The validator reports missing directories/labels, orphan labels, class count/ID issues, malformed boxes, and duplicate image stems. Train/validation images and labels are required for fine-tuning; the full validator also checks the test split.

Validate a dataset and optionally fine-tune from COCO weights:

```powershell
python scripts/validate_datasets.py .\path\to\dataset\dataset.yaml
python scripts/train_yolo.py .\path\to\dataset\dataset.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device auto
```

The validator prints a JSON report and exits nonzero on errors. Fine-tuning is an explicit command; it checks the train and validation splits before starting. Ultralytics saves `best.pt`, `last.pt`, and `results.csv` in `outputs/runs/yolo/cctv_yolov8/`. Initial pretrained COCO weights may be downloaded by Ultralytics if absent; detection/fine-tuning code is not invoked by the Phase 3 unit tests.

Run the pretrained detector directly on already extracted keyframes (training is not required):

```powershell
python scripts/detect_keyframes.py .\data\processed\keyframes\<video-id>\keyframe_000.jpg .\data\processed\keyframes\<video-id>\keyframe_001.jpg --track
```

The command writes per-frame labels, confidence scores, normalized boxes, optional track IDs, and aggregate object counts/persistence to `outputs/predictions/detections.json`. It also writes a `.pt` companion containing the `class_ids`, eight-column normalized `stats`, and `object_mask` tensors consumed by `ObjectFeatureEncoder`; this is the next-stage object input. Omit `--track` for detector-only inference. If `yolov8n.pt` is not cached, Ultralytics downloads its pretrained COCO weights on first use.

Phase 3 tests (mocked detector, no YOLO model-weight download):

```powershell
pytest -q tests/test_yolo_phase3.py
```

## Phase 4: temporal encoder and optional event classifier

`TemporalVideoDataset` reads `video_path,label,split` CSV manifests or `data/events/{train,val,test}/{normal,abnormal}/` directories. It checks video-level split isolation and uniformly samples configured clips from variable-duration videos. `SlowFastEncoder` uses PyTorchVideo SlowFast-R50 and exposes its pre-classifier Kinetics embedding when PyTorchVideo is installed. If the dependency is missing and `allow_fallback` is enabled, it emits a warning and uses an explicitly named trainable frame-aggregation CNN; that fallback is **not SlowFast and has no Kinetics weights**. Set `allow_fallback: false` to require SlowFast.

The optional event head predicts only normal/abnormal classes from temporal embeddings. It is separate from caption generation. Training uses inverse-frequency class weights and a balanced video sampler, reports accuracy/balanced accuracy/precision/recall/F1/ROC-AUC/confusion matrix, selects a validation threshold, and saves `best_val_f1.pt`, `last.pt`, and metric history. It requires video-level event labels; YOLO bounding boxes or generated captions are not event labels.

Train the optional binary event model:

```powershell
python scripts/train_event_classifier.py --manifest .\data\manifests\events.csv
```

The CSV needs `video_path,label,split` (optional `video_id`); labels should be `normal` and `abnormal`. All rows/files for an original video must remain in one split. To use the directory adapter instead:

```powershell
python scripts/train_event_classifier.py --directory .\data\events
```

Training uses the configured PyTorchVideo SlowFast-R50 pretrained weights when available and may download them on first use. If PyTorchVideo is unavailable, the configured warning/fallback behavior applies. Event training is optional and is not required for YOLO detection or the later caption pipeline. Phase 4 tests use tiny synthetic clips and the fallback or injected model; they do not download SlowFast weights.

Phase 4 tests:

```powershell
pytest -q tests/test_temporal_phase4.py
```

## Phase 5: pretrained action recognition on CCTV clips

`scripts/recognize_video_actions.py` runs the actual official PyTorchVideo SlowFast-R50 Kinetics-400 classifier on 32-frame video clips. It automatically loads/downloads the pretrained checkpoint and official Kinetics class-name map into `checkpoints/torch_hub/`. If Phase 2 created a matching `keyframes.json`, the command runs pretrained YOLOv8n on those keyframes, aggregates the detections into the existing eight-column object statistics, saves the downstream tensors to a `.pt` file, and reports the SlowFast top-k class labels and softmax scores for uniformly spaced clips. Use `--keyframes-manifest` to specify the manifest explicitly.

The object results and action scores are combined in one output record, but the official pretrained SlowFast classifier is not conditioned on YOLO embeddings. The YOLO features are available for a later trained fusion model; passing untrained object embeddings into SlowFast would give misleading results. A missing Phase 2 manifest is permitted; action inference still runs and reports that YOLO was skipped.

```powershell
python scripts/recognize_video_actions.py .\data\raw\Abuse001_x264.mp4 --top-k 5 --max-clips 3
```

The output JSON is written to `outputs/predictions/<video>_phase5_actions.json`, with a `.pt` companion containing YOLO object tensors. `--device cuda` selects CUDA explicitly and errors if the installed PyTorch build has no CUDA support; `auto` uses CUDA only when available. The current project environment was CPU-only, so inference is expected to be slower.

### What Kinetics-400 can and cannot identify

The pretrained head predicts one of Kinetics-400's general human-action classes. Its label map includes a few fight-like movement labels such as `punching person (boxing)`; these do not mean that the clip is a real-world assault or CCTV fight. It is not trained to identify crime intent or police/security incidents and has no dedicated CCTV labels for theft, shoplifting, robbery, burglary, violence, or road accidents. A top-1 score is a softmax share among these 400 labels, not a calibrated incident probability. Treat the output as generic action suggestions and verify against video context.

For CCTV-specific event recognition, train or fine-tune a clip classifier using labeled surveillance footage from the target camera setting. UCF-Crime is a direct multi-event candidate: it has long untrimmed surveillance videos covering 13 anomaly types, including fighting, road accidents, burglary, robbery, stealing, and shoplifting. Its video-level anomaly labels are weak for exact event timing; for clip-level classification/localization, annotate event start/end times and sample positive and normal clips. RWF-2000 is a closer source for a binary violent/nonviolent CCTV classifier, but it does not cover theft or accidents. Split by original video/camera/source before training and evaluation to avoid leakage. For reliable deployment, add representative target-domain clips and hard negatives for every required class.

Phase 5 unit tests (preprocessing and top-k mapping use an injected test model; no checkpoint download):

```powershell
pytest -q tests/test_slowfast_action_phase5.py
```

## Planned training/evaluation commands

The overall sequence is: inspect manifests and annotations, extract/cache visual keyframes, optionally fine-tune YOLO, optionally train the event/action head, extract remaining modality features, train the caption model on paired human captions, then evaluate on held-out videos and run inference. Temporal modeling, caption training, evaluation, and full inference will be documented in later phases.
