"""
Waggle Dance Annotation Viewer — Flask Server

Serves video frames on-demand, annotation data, and model predictions
for visual inspection of the waggle dance training dataset.

Usage:
    python3 viewer/server.py                                        # localhost:5050
    python3 viewer/server.py --checkpoint ./ckpt/run/best.pth       # with model predictions
    python3 viewer/server.py --host 0.0.0.0 --checkpoint ./ckpt/run/best.pth
    python3 viewer/server.py --checkpoint ./ckpt/run/best.pth --external-videos /path/to/videos/
"""

import argparse
import base64
import re
import os
import sys
import threading
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import yaml
from flask import Flask, render_template, jsonify, Response, request

# Add project root to path so we can import training code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
ANNOTATIONS_PATH = os.path.join(DATA_DIR, "annotations", "fps_multires_clean.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
CROP_SIZE = 224  # must match config
EXTERNAL_VIDEO_DIR = None  # set via --external-videos CLI arg

# Registry: video_name → absolute path (for videos outside VIDEO_DIR)
_external_video_paths = {}

FRAME_CACHE_SIZE = 512   # max decoded JPEG frames in memory
JPEG_QUALITY = 85

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_base_and_variant(video_name: str):
    """Extract base recording ID, resolution, and FPS variant from filename.

    Examples:
        001_2304_1760_60fps.mp4      →  ('001', 2304, 1760, '60fps')
        039_576_440_30fps.mp4        →  ('039', 576, 440, '30fps')
        039_1152_880_30fps_30fps.mp4 →  ('039', 1152, 880, '30fps_30fps')
        001_576_440.mp4              →  ('001', 576, 440, 'native')
        C1_11_03_25_..._480_270.mp4  →  ('C1_11_03_25_...', 480, 270, 'native')
    """
    name = video_name.replace(".mp4", "")
    # Match: <base>_<width>_<height>[_<N>fps[_<N>fps...]]
    match = re.match(r"^(.+?)_(\d+)_(\d+)((?:_\d+fps)*)$", name)
    if match:
        base = match.group(1)
        w, h = int(match.group(2)), int(match.group(3))
        fps_tag = match.group(4).lstrip("_") if match.group(4) else "native"
        return base, w, h, fps_tag
    return name, None, None, "unknown"


def load_annotations():
    """Load the annotation CSV."""
    df = pd.read_csv(ANNOTATIONS_PATH)
    return df


def build_recording_index(df):
    """Build a lookup: base_recording_id → {variants, waggle_runs}.

    Waggle runs are deduplicated per (video_name, waggle_run_id),
    keeping position/direction from the representative row.
    """
    recordings = {}

    for video_name in df["video_name"].unique():
        base, w, h, fps_tag = get_base_and_variant(video_name)

        if base not in recordings:
            recordings[base] = {"base_id": base, "variants": [], "waggle_runs": {}}

        video_path = os.path.join(VIDEO_DIR, video_name)
        variant = {
            "video_name": video_name,
            "width": w,
            "height": h,
            "fps_tag": fps_tag,
            "exists": os.path.exists(video_path),
        }
        recordings[base]["variants"].append(variant)

        # Deduplicate waggle runs for this video
        video_df = df[df["video_name"] == video_name]
        for run_id, run_group in video_df.groupby("waggle_run_id"):
            first = run_group.iloc[0]
            key = f"{video_name}::{run_id}"
            recordings[base]["waggle_runs"][key] = {
                "run_id": int(run_id),
                "video_name": video_name,
                "x": float(first["x1"]),
                "y": float(first["y1"]),
                "dir_x": float(first["direction_x"]),
                "dir_y": float(first["direction_y"]),
                "waggle_start": int(first["waggle_start"]),
                "waggle_end": int(first["waggle_end"]),
            }

    # Sort variants by resolution (lowest first)
    for rec in recordings.values():
        rec["variants"].sort(key=lambda v: (v["width"] or 0, v["fps_tag"]))

    return recordings


# ---------------------------------------------------------------------------
# Frame Serving — thread-safe LRU cache
# ---------------------------------------------------------------------------

class FrameServer:
    """Decode video frames on-demand with per-video locking & LRU caching."""

    def __init__(self, max_cache: int = 512):
        self.max_cache = max_cache
        self._frame_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._caps = {}          # video_name → cv2.VideoCapture
        self._cap_locks = {}     # video_name → Lock  (serialise seek+read)
        self._global_lock = threading.Lock()

    def _get_cap(self, video_name: str):
        """Return (VideoCapture, Lock) for a given video, creating if needed."""
        with self._global_lock:
            if video_name not in self._caps:
                path = resolve_video_path(video_name)
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise FileNotFoundError(f"Cannot open video: {path}")
                self._caps[video_name] = cap
                self._cap_locks[video_name] = threading.Lock()
            return self._caps[video_name], self._cap_locks[video_name]

    def get_frame_jpeg(self, video_name: str, frame_idx: int) -> bytes | None:
        """Return JPEG-encoded frame bytes, or None if frame doesn't exist."""
        key = (video_name, frame_idx)

        with self._cache_lock:
            if key in self._frame_cache:
                self._frame_cache.move_to_end(key)
                return self._frame_cache[key]

        # Decode frame (serialised per video)
        cap, lock = self._get_cap(video_name)
        with lock:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

        if not ret:
            return None

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg_bytes = buf.tobytes()

        with self._cache_lock:
            self._frame_cache[key] = jpeg_bytes
            while len(self._frame_cache) > self.max_cache:
                self._frame_cache.popitem(last=False)

        return jpeg_bytes

    def get_video_info(self, video_name: str) -> dict:
        """Return video metadata without decoding frames."""
        cap, _ = self._get_cap(video_name)
        return {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }


frame_server = FrameServer(FRAME_CACHE_SIZE)


# ---------------------------------------------------------------------------
# Startup — load data
# ---------------------------------------------------------------------------

print("Loading annotations …")
annotations_df = load_annotations()
print(f"  {len(annotations_df):,} rows, {annotations_df['video_name'].nunique()} videos")

print("Building recording index …")
recording_index = build_recording_index(annotations_df)
print(f"  {len(recording_index)} base recordings")

# Compute train/val split (replicate main.py exactly)
print("Computing train/val split …")
from src.utils.video_utils import get_video_category
with open(CONFIG_PATH) as _f:
    _config = yaml.safe_load(_f)
_SEED = 42
_train_ratio = _config['data']['train_ratio']
_video_df = pd.DataFrame({'video_name': annotations_df['video_name'].unique()})
_video_df['category'] = _video_df['video_name'].apply(get_video_category)
train_videos = set()
for _cat, _group in _video_df.groupby('category'):
    _vids = np.random.RandomState(_SEED).permutation(_group['video_name'].values)
    _n_train = int(_train_ratio * len(_vids))
    train_videos.update(_vids[:_n_train])
val_videos = set(_video_df['video_name'].values) - train_videos
print(f"  {len(train_videos)} train / {len(val_videos)} val videos")


# ---------------------------------------------------------------------------
# Prediction Engine — model loading, inference, caching
# ---------------------------------------------------------------------------

class PredictionEngine:
    """Loads a checkpoint and runs inference on video windows from the dataset CSV."""

    def __init__(self):
        self.model = None
        self.device = None
        self.config = None
        self.checkpoint_path = None
        self.checkpoint_meta = {}  # epoch, best_score, etc.
        self._cache = {}  # video_name -> {raw: [...], postprocessed: [...]}
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self.model is not None

    def load_checkpoint(self, checkpoint_path, device_override=None):
        """Load (or reload) a model checkpoint."""
        import torch
        from src.utils.model_utils import load_pretrained_model, EMA
        from src.utils.data_utils import load_config

        with self._lock:
            config = load_config(CONFIG_PATH)
            self.config = config
            if device_override:
                device = torch.device(device_override)
            else:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.device = device

            print(f"  Loading checkpoint: {checkpoint_path}")
            model, checkpoint = load_pretrained_model(checkpoint_path, config, device)

            # Apply EMA weights if available
            ema = EMA(model, decay=config['train']['ema_decay'], device=device)
            if 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                ema.apply_shadow()
                print(f"  Applied EMA weights (updates: {ema.updates})")

            model.eval()
            self.model = model
            self.checkpoint_path = os.path.abspath(checkpoint_path)
            self.checkpoint_meta = {
                'path': self.checkpoint_path,
                'filename': os.path.basename(checkpoint_path),
                'epoch': checkpoint.get('epoch', '?'),
                'best_score': float(checkpoint.get('best_score', 0)),
                'val_loss': float(checkpoint.get('val_loss', 0)) if checkpoint.get('val_loss') else None,
            }
            # Invalidate cache
            self._cache.clear()
            print(f"  Model ready on {device} (epoch {self.checkpoint_meta['epoch']})")

    def get_predictions(self, video_name):
        """Get predictions for a video, using cache if available."""
        if not self.is_loaded:
            return None

        with self._lock:
            if video_name in self._cache:
                return self._cache[video_name]

        # Compute outside lock (GPU work)
        result = self._run_inference(video_name)

        with self._lock:
            self._cache[video_name] = result
        return result

    def _run_inference(self, video_name):
        """Run model inference on all dataset windows for a given video."""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms import ToTensor
        from src.data.video_loader import load_video_frames
        from src.utils.eval_utils import yolo_to_img_space

        config = self.config
        video_rows = annotations_df[annotations_df['video_name'] == video_name]
        if len(video_rows) == 0:
            return {'raw': [], 'postprocessed': [], 'n_windows': 0}

        # Build test transform (same as training eval)
        test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config['augmentations']['width'],
                      config['augmentations']['height'])),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=config['augmentations']['mean'],
                        std=config['augmentations']['std']),
        ])
        to_tensor = ToTensor()

        crop_h, crop_w = config['data']['height'], config['data']['width']
        window_size = config['data']['window_size']

        all_outputs = []
        all_starts = []
        all_ends = []
        all_run_ids = []
        all_crop_origins = []  # (x_min, y_min) per window for coord remapping

        video_path = os.path.join(VIDEO_DIR, video_name)
        batch_tensors = []
        BATCH_SIZE = 16  # process in mini-batches for GPU efficiency

        print(f"  Running inference on {len(video_rows)} windows for {video_name}...")

        for idx, (_, row) in enumerate(video_rows.iterrows()):
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            gt_x, gt_y = float(row['x1']), float(row['y1'])

            # Load frames
            frames = load_video_frames(video_path, start_frame, end_frame, use_cache=False)
            if not frames:
                continue

            first_frame = frames[0]
            if isinstance(first_frame, torch.Tensor):
                original_h, original_w = first_frame.shape[-2:]
            else:
                original_h, original_w = first_frame.shape[:2]

            # Centered crop (no random offset — this is eval mode)
            x_min_ideal = int(gt_x - crop_w / 2)
            y_min_ideal = int(gt_y - crop_h / 2)
            x_min = max(0, min(original_w - crop_w, x_min_ideal))
            y_min = max(0, min(original_h - crop_h, y_min_ideal))
            x_max = x_min + crop_w
            y_max = y_min + crop_h

            # Crop, convert to tensor, apply transforms
            frame_tensors = [to_tensor(f[y_min:y_max, x_min:x_max]) for f in frames]
            frame_tensors = [test_transform(f) for f in frame_tensors]

            # Sample/pad to window_size
            num_frames = len(frame_tensors)
            if num_frames >= window_size:
                idxs_sample = np.linspace(0, num_frames - 1, window_size, dtype=int)
                frame_tensors = [frame_tensors[i] for i in idxs_sample]
            else:
                reps = frame_tensors * (window_size // num_frames + 1)
                frame_tensors = reps[:window_size]

            video_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)  # (C, T, H, W)
            batch_tensors.append(video_tensor)
            all_starts.append(start_frame)
            all_ends.append(end_frame)
            all_run_ids.append(int(row.get('waggle_run_id', -1)))
            all_crop_origins.append((x_min, y_min))  # track crop origin for coord offset

            # Process in batches
            if len(batch_tensors) >= BATCH_SIZE or idx == len(video_rows) - 1:
                batch = torch.stack(batch_tensors).to(self.device)  # (B, C, T, H, W)
                dev_type = self.device.type  # 'cuda' or 'cpu'
                amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16
                with torch.no_grad():
                    with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                        outputs = self.model(batch)
                all_outputs.append(outputs.cpu())
                batch_tensors = []

        if not all_outputs:
            return {'raw': [], 'postprocessed': [], 'n_windows': 0}

        # Concatenate all outputs
        all_outputs = torch.cat(all_outputs, dim=0)
        all_starts_arr = np.array(all_starts)
        all_ends_arr = np.array(all_ends)

        # Convert to crop-space coordinates, then offset to full video space
        crop_size = (config['data']['width'], config['data']['height'])
        raw_preds = yolo_to_img_space(
            all_outputs, all_starts_arr, all_ends_arr,
            window_size=window_size,
            confidence_threshold=config['eval']['confidence_threshold'],
            original_size=crop_size,
            max_dets=config['eval']['max_dets'],
        )

        # Offset positions from crop-space to full video space
        for i, window_preds in enumerate(raw_preds):
            ox, oy = all_crop_origins[i]
            for pred in window_preds:
                pred['position'] = [
                    pred['position'][0] + ox,
                    pred['position'][1] + oy,
                ]

        # Add window metadata to each raw prediction and flatten
        raw_serializable = []
        all_preds_flat = []  # flat pool for cross-window clustering
        for i, window_preds in enumerate(raw_preds):
            for pred in window_preds:
                pred['window_start'] = int(all_starts[i])
                pred['window_end'] = int(all_ends[i])
                pred['waggle_run_id'] = int(all_run_ids[i])
                all_preds_flat.append(pred)
            raw_serializable.extend(window_preds)

        # Run postprocessing: cluster ALL predictions across ALL windows together
        # (not per-window) so overlapping windows merge properly
        from src.utils.postprocess import postprocess_predictions
        post_result = postprocess_predictions(
            all_preds_flat,
            spatial_threshold=config['post_process']['spatial_threshold'],
            temporal_threshold=config['post_process']['temporal_threshold'],
            confidence_threshold=config['post_process']['confidence_threshold'],
            strategy=config['post_process']['strategy'],
            mode=config['post_process']['mode'],
            remove_outliers=config['post_process'].get('outlier_detection', False),
            outlier_method='isolation_forest',
        )
        post_serializable = post_result['filtered_predictions']

        # Convert numpy types to Python native for JSON serialization
        def _to_native(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_native(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        raw_serializable = [_to_native(p) for p in raw_serializable]
        post_serializable = [_to_native(p) for p in post_serializable]

        print(f"  {video_name}: {len(raw_serializable)} raw -> {len(post_serializable)} post-processed")

        return {
            'raw': raw_serializable,
            'postprocessed': post_serializable,
            'n_windows': len(all_starts),
        }


prediction_engine = PredictionEngine()


# ---------------------------------------------------------------------------
# Video path resolution — supports both annotated and external videos
# ---------------------------------------------------------------------------

def resolve_video_path(video_name: str) -> str:
    """Resolve a video name to its absolute path.
    Checks: external registry → VIDEO_DIR → EXTERNAL_VIDEO_DIR.
    """
    # Check external registry first
    if video_name in _external_video_paths:
        return _external_video_paths[video_name]
    # Check standard video directory
    path = os.path.join(VIDEO_DIR, video_name)
    if os.path.exists(path):
        return path
    # Check external directory
    if EXTERNAL_VIDEO_DIR:
        path = os.path.join(EXTERNAL_VIDEO_DIR, video_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Video not found: {video_name}")


def scan_external_videos():
    """Scan EXTERNAL_VIDEO_DIR (recursively) for video files not in the annotation CSV."""
    global _external_video_paths
    if not EXTERNAL_VIDEO_DIR or not os.path.isdir(EXTERNAL_VIDEO_DIR):
        return []

    annotated_names = set(annotations_df['video_name'].unique()) if 'annotations_df' in globals() else set()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV'}
    external = []

    for root, dirs, files in os.walk(EXTERNAL_VIDEO_DIR):
        dirs.sort()
        for fname in sorted(files):
            _, ext = os.path.splitext(fname)
            if ext not in video_extensions:
                continue
            full_path = os.path.join(root, fname)
            # Use relative path from EXTERNAL_VIDEO_DIR as the key
            # e.g. "100fps/18_08_2008-1436-SINGLE.mp4"
            rel_path = os.path.relpath(full_path, EXTERNAL_VIDEO_DIR)
            if rel_path in annotated_names or fname in annotated_names:
                continue
            _external_video_paths[rel_path] = full_path
            external.append(rel_path)

    return external


# ---------------------------------------------------------------------------
# Sliding Window Inference — full-frame fully-convolutional inference
# ---------------------------------------------------------------------------

class SlidingWindowInference:
    """Run model inference on unannotated videos via temporal sliding window.

    The model is fully convolutional, so we feed full-resolution frames
    directly — no spatial tiling or cropping needed. The output grid
    scales proportionally with input resolution.
    """

    def __init__(self, prediction_engine):
        self.engine = prediction_engine
        self._cache = {}  # video_name → result dict
        self._lock = threading.Lock()
        self._progress = {}  # video_name → (current, total)

    def get_results(self, video_name):
        """Get cached results, or None."""
        with self._lock:
            return self._cache.get(video_name)

    def get_progress(self, video_name):
        """Get inference progress as (current, total)."""
        return self._progress.get(video_name, (0, 0))

    def run_inference(self, video_name, temporal_stride=8):
        """Run full-frame sliding-window inference on a video."""
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T
        from torchvision.transforms import ToTensor

        if not self.engine.is_loaded:
            raise RuntimeError("No model checkpoint loaded")

        video_path = resolve_video_path(video_name)
        config = self.engine.config

        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        window_size = config['data']['window_size']

        # Transform: same as training eval but WITHOUT Resize (keep native resolution)
        # The training pipeline does: ToTensor → ToPILImage → Resize(224) → Grayscale(3) → ToTensor → Normalize
        # Resize is a no-op for 224x224 crops, so we skip it for full-frame.
        to_tensor = ToTensor()
        full_frame_transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=config['augmentations']['mean'],
                        std=config['augmentations']['std']),
        ])

        # Compute temporal windows
        starts = list(range(0, max(1, total_frames - window_size + 1), temporal_stride))
        if not starts:
            starts = [0]
        total_windows = len(starts)
        self._progress[video_name] = (0, total_windows)

        all_detections = []

        print(f"  Inference: {video_name} ({vid_width}×{vid_height}, {total_frames} frames, "
              f"{vid_fps:.0f}fps) → {total_windows} windows")

        for win_idx, start_frame in enumerate(starts):
            end_frame = min(start_frame + window_size, total_frames)

            # Load frames
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)  # BGR numpy
            cap.release()

            if not frames:
                continue

            # Transform: BGR numpy → same pipeline as training
            # (ToTensor on BGR numpy keeps channel order; training does the same)
            frame_tensors = []
            for f in frames:
                t = to_tensor(f)  # HWC uint8 → CHW float [0,1]
                t = full_frame_transform(t)  # ToPIL → Grayscale(3) → ToTensor → Normalize
                frame_tensors.append(t)

            # Pad/sample to window_size
            n = len(frame_tensors)
            if n >= window_size:
                idxs = np.linspace(0, n - 1, window_size, dtype=int)
                frame_tensors = [frame_tensors[i] for i in idxs]
            else:
                reps = frame_tensors * (window_size // n + 1)
                frame_tensors = reps[:window_size]

            # Stack: (C, T, H, W) → batch: (1, C, T, H, W)
            video_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)
            batch = video_tensor.unsqueeze(0).to(self.engine.device)

            # Pad spatial dims to be divisible by 8 (backbone stride)
            _, _, _, h, w = batch.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h or pad_w:
                batch = F.pad(batch, (0, pad_w, 0, pad_h), mode='reflect')

            # Run model
            dev_type = self.engine.device.type  # 'cuda' or 'cpu'
            amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16
            with torch.no_grad():
                with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                    output = self.engine.model(batch)

            # Decode output grid → pixel-space detections
            dets = self._decode_output(
                output, start_frame, end_frame,
                frame_width=vid_width, frame_height=vid_height,
                pad_h=pad_h, pad_w=pad_w,
                window_size=window_size,
                confidence_threshold=config['eval']['confidence_threshold'],
                max_dets=config['eval']['max_dets'],
            )
            all_detections.extend(dets)
            self._progress[video_name] = (win_idx + 1, total_windows)

            if (win_idx + 1) % 50 == 0 or win_idx == total_windows - 1:
                print(f"    Window {win_idx + 1}/{total_windows} — {len(all_detections)} raw detections")

        # Post-process: cluster all detections across windows
        from src.utils.postprocess import postprocess_predictions
        post_result = postprocess_predictions(
            all_detections,
            spatial_threshold=config['post_process']['spatial_threshold'],
            temporal_threshold=config['post_process']['temporal_threshold'],
            confidence_threshold=config['post_process']['confidence_threshold'],
            strategy=config['post_process']['strategy'],
            mode=config['post_process']['mode'],
            remove_outliers=config['post_process'].get('outlier_detection', False),
            outlier_method='isolation_forest',
        )
        post_preds = post_result['filtered_predictions']

        # Ensure JSON serializable
        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_native(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        all_detections = [_to_native(d) for d in all_detections]
        post_preds = [_to_native(d) for d in post_preds]

        result = {
            'raw': all_detections,
            'postprocessed': post_preds,
            'n_windows': total_windows,
            'video_info': {
                'width': vid_width, 'height': vid_height,
                'total_frames': total_frames, 'fps': vid_fps,
            },
        }

        print(f"  {video_name}: {len(all_detections)} raw → {len(post_preds)} post-processed")

        with self._lock:
            self._cache[video_name] = result
        del self._progress[video_name]

        return result

    def _decode_output(self, output, start_frame, end_frame,
                       frame_width, frame_height, pad_h, pad_w,
                       window_size, confidence_threshold=0.001, max_dets=200):
        """Decode model output grid to pixel-space detections.

        Handles non-square grids (grid_h ≠ grid_w) for full-frame inference.
        The padded pixels are excluded by limiting decoded coordinates
        to the original frame dimensions.
        """
        import torch

        det = output[0]  # (grid_h, grid_w, max_det, 7)
        grid_h, grid_w = det.shape[0], det.shape[1]

        # Padded dimensions that were fed to the model
        padded_w = frame_width + pad_w
        padded_h = frame_height + pad_h

        cell_w = padded_w / grid_w
        cell_h = padded_h / grid_h

        detections = []
        for i in range(grid_h):
            for j in range(grid_w):
                for k in range(det.shape[2]):
                    d = det[i, j, k].detach().cpu()
                    conf = torch.sigmoid(d[0]).item()

                    if conf > confidence_threshold:
                        norm_x = d[1].item()
                        norm_y = d[2].item()
                        dir_x = d[3].item()
                        dir_y = d[4].item()
                        t_start = d[5].item()
                        t_end = d[6].item()

                        # Map grid cell to pixel coordinates
                        pos_x = (j + norm_x) * cell_w
                        pos_y = (i + norm_y) * cell_h

                        # Skip detections in the padded region
                        if pos_x >= frame_width or pos_y >= frame_height:
                            continue

                        pos_x = max(0.0, min(float(pos_x), frame_width - 1))
                        pos_y = max(0.0, min(float(pos_y), frame_height - 1))

                        # Temporal offsets → absolute frame indices
                        abs_start = int(t_start * window_size + start_frame)
                        abs_end = int(t_end * window_size + start_frame)
                        abs_start = max(start_frame, min(abs_start, end_frame))
                        abs_end = max(abs_start, min(abs_end, end_frame))

                        detections.append({
                            'confidence': float(conf),
                            'position': [pos_x, pos_y],
                            'direction': [float(dir_x), float(dir_y)],
                            'grid_cell': [i, j, k],
                            'temporal_offsets': [abs_start, abs_end],
                            'window_start': int(start_frame),
                            'window_end': int(end_frame),
                        })

        # Keep top-N by confidence per window
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[:max_dets]


sliding_inference = SlidingWindowInference(prediction_engine)


# ---------------------------------------------------------------------------
# Evaluation Engine — full STD-mAP evaluation on the validation set
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """Runs the complete evaluation pipeline (STD-mAP) on the validation set.

    Replicates exactly the evaluation loop from ckpt_eval.py / main.py:
      1. Build val DataLoader from val split
      2. get_preds_gt() → raw logits + ground truth
      3. yolo_to_img_space() / yolo_to_img_space_gt() → decoded detections
      4. batch_postprocess_predictions() → clustered predictions
      5. get_eval_metrics() → comprehensive metrics (pre + post)

    Runs in a background thread with progress tracking.
    """

    CACHE_PATH = os.path.join(os.path.dirname(__file__), '.eval_cache.pkl')

    def __init__(self):
        self._results = None          # cached final results dict
        self._results_ckpt = None     # checkpoint path of cached results
        self._lock = threading.Lock()
        self._running = False
        self._progress = {            # progress state
            'stage': 'idle',          # idle | loading | inference | postprocess | metrics | done | error
            'batch_current': 0,
            'batch_total': 0,
            'message': '',
        }
        self._error = None
        # Cached intermediate results for fast re-clustering
        self._cached_preds = None     # decoded per-window predictions (list of lists)
        self._cached_gts = None       # decoded per-window ground truths
        self._cached_video_names = None  # video name per window
        self._cached_config = None    # eval config used
        self._cached_run_ids = None   # waggle_run_id per window (from annotations)
        self._load_disk_cache()

    def _load_disk_cache(self):
        """Load cached decoded predictions from disk if available."""
        import pickle
        if os.path.exists(self.CACHE_PATH):
            try:
                with open(self.CACHE_PATH, 'rb') as f:
                    cache = pickle.load(f)
                self._cached_preds = cache['preds']
                self._cached_gts = cache['gts']
                self._cached_video_names = cache['video_names']
                self._cached_config = cache['config']
                self._results_ckpt = cache.get('ckpt_path')
                self._cached_run_ids = cache.get('run_ids')
                self._results = cache.get('results')
                if self._results:
                    self._progress = {'stage': 'done', 'batch_current': 0,
                                      'batch_total': 0, 'message': 'Loaded from cache.'}
                print(f"  ✅ Loaded eval cache from disk: {len(self._cached_preds)} windows"
                      f" (results={'yes' if self._results else 'preds only'})", flush=True)
            except Exception as e:
                print(f"  ⚠️ Failed to load eval cache: {e}", flush=True)

    def _save_disk_cache(self, ckpt_path=None):
        """Save decoded predictions + results to disk for persistence across restarts."""
        import pickle
        cache = {
            'preds': self._cached_preds,
            'gts': self._cached_gts,
            'video_names': self._cached_video_names,
            'config': self._cached_config,
            'ckpt_path': ckpt_path,
            'run_ids': self._cached_run_ids,
            'results': self._results,
        }
        with open(self.CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(self.CACHE_PATH) / (1024 * 1024)
        print(f"  💾 Saved eval cache to disk ({mb:.1f} MB)", flush=True)

    @property
    def is_running(self):
        return self._running

    def get_progress(self):
        return dict(self._progress)

    def get_results(self):
        with self._lock:
            return self._results

    def run_async(self, prediction_engine, config):
        """Start evaluation in a background thread."""
        if self._running:
            return False  # already running

        # Check if results are cached for this checkpoint
        ckpt_path = prediction_engine.checkpoint_path
        if self._results and self._results_ckpt == ckpt_path:
            return True  # results already available

        self._running = True
        self._error = None
        self._progress = {
            'stage': 'loading',
            'batch_current': 0,
            'batch_total': 0,
            'message': 'Building validation DataLoader…',
        }

        thread = threading.Thread(
            target=self._run_evaluation,
            args=(prediction_engine, config),
            daemon=True,
        )
        thread.start()
        return True

    def _run_evaluation(self, pred_engine, config):
        """Full evaluation pipeline (runs in background thread)."""
        import torch
        import torchvision.transforms as T
        from src.data.dataset import VideoYoloDataset
        from src.utils.eval_utils import (
            get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt,
            get_eval_metrics, print_evaluation_results,
        )
        from src.utils.postprocess import batch_postprocess_predictions

        try:
            # ── 1. Build val DataLoader ──
            self._progress['stage'] = 'loading'
            self._progress['message'] = 'Building validation dataset…'

            test_df = annotations_df[
                ~annotations_df['video_name'].isin(train_videos)
            ].reset_index(drop=True)

            test_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((config['augmentations']['width'],
                          config['augmentations']['height'])),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=config['augmentations']['mean'],
                            std=config['augmentations']['std']),
            ])

            test_dataset = VideoYoloDataset(
                test_df,
                config['data']['data_dir'],
                test_transform,
                width=config['data']['width'],
                height=config['data']['height'],
                window_size=config['data']['window_size'],
                grid_size=config['model']['grid_size'],
                max_detections_per_cell=config['model']['max_detections_per_cell'],
                n_classes=config['model']['n_classes'],
                augment=None,
                is_training=False,
            )

            # ── 2. Inference — manual batching (no DataLoader) ──
            # DataLoader deadlocks in daemon threads even with num_workers=0
            # due to GIL contention with Flask's request threads.
            # Manual iteration avoids this entirely.
            model = pred_engine.model
            device = pred_engine.device
            model.eval()

            eval_batch_size = min(config['eval'].get('batch_size', 64), 8)
            n_samples = len(test_dataset)
            n_batches = (n_samples + eval_batch_size - 1) // eval_batch_size

            self._progress.update({
                'stage': 'inference',
                'batch_total': n_batches,
                'message': f'Running inference on {n_samples} val windows (bs={eval_batch_size})…',
            })

            import time as _time
            import sys as _sys
            print(f"  Eval: {n_samples} val windows, {n_batches} batches (bs={eval_batch_size})", flush=True)

            # Quick sanity test: can we load any sample at all?
            print(f"  Eval: loading sample 0 from dataset...", flush=True)
            _t = _time.time()
            _s = test_dataset[0]
            print(f"  Eval: sample 0 OK ({_time.time()-_t:.2f}s, video={_s['metadata']['video_name']})", flush=True)
            del _s

            all_outputs = []
            all_targets = []
            all_starts = []
            all_ends = []
            all_video_names = []
            all_original_res = []

            t0 = _time.time()

            for batch_start in range(0, n_samples, eval_batch_size):
                t_batch = _time.time()
                batch_end = min(batch_start + eval_batch_size, n_samples)
                batch_idx = batch_start // eval_batch_size

                # Load samples one by one — yield GIL between samples
                videos = []
                targets = []
                for i in range(batch_start, batch_end):
                    if i < 3:
                        print(f"  Eval: loading sample {i}...", flush=True)
                    try:
                        sample = test_dataset[i]
                    except Exception as e:
                        print(f"  Eval: SKIP sample {i}: {e}", flush=True)
                        continue
                    videos.append(sample['video'])
                    targets.append(sample['targets'])
                    meta = sample['metadata']
                    all_starts.append(meta['start_frame'])
                    all_ends.append(meta['end_frame'])
                    all_video_names.append(meta['video_name'])
                    all_original_res.append(meta['res'])
                    _time.sleep(0)  # yield GIL to Flask threads

                if not videos:
                    continue

                inputs = torch.stack(videos).to(device)
                targets_t = torch.stack(targets)

                dev_type = device.type
                amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16
                with torch.no_grad():
                    with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                        outputs = model(inputs)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets_t.cpu())
                del inputs, outputs, videos, targets, targets_t

                elapsed = _time.time() - t0
                batch_time = _time.time() - t_batch
                eta = (elapsed / (batch_idx + 1)) * (n_batches - batch_idx - 1)

                self._progress['batch_current'] = batch_idx + 1
                self._progress['message'] = (
                    f'Inference: {batch_idx + 1}/{n_batches} '
                    f'({batch_time:.1f}s/batch, ETA {eta:.0f}s)'
                )
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"  Eval batch {batch_idx+1}/{n_batches} "
                          f"({batch_time:.1f}s, ETA {eta/60:.1f}min)", flush=True)

            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_starts = np.array(all_starts)
            all_ends = np.array(all_ends)
            all_video_names = np.array(all_video_names)

            # ── 3. Decode to image space ──
            self._progress['stage'] = 'postprocess'
            self._progress['message'] = 'Decoding predictions to image space…'

            test_gts = yolo_to_img_space_gt(
                all_targets,
                all_starts=all_starts,
                all_ends=all_ends,
                window_size=config['data']['window_size'],
                original_size=(config['data']['width'],
                               config['data']['height']),
            )

            test_preds = yolo_to_img_space(
                all_outputs,
                all_starts=all_starts,
                all_ends=all_ends,
                confidence_threshold=config['eval']['confidence_threshold'],
                window_size=config['data']['window_size'],
                original_size=(config['data']['width'],
                               config['data']['height']),
                max_dets=config['eval']['max_dets'],
            )

            # Extract waggle_run_ids from annotations (parallel to windows)
            waggle_run_ids = test_df['waggle_run_id'].values.tolist()

            # Cache decoded predictions for fast re-clustering
            self._cached_preds = test_preds
            self._cached_gts = test_gts
            self._cached_video_names = all_video_names
            self._cached_config = config
            self._cached_run_ids = waggle_run_ids
            print(f"  Eval: cached {len(test_preds)} decoded windows for re-clustering", flush=True)
            self._save_disk_cache(ckpt_path=pred_engine.checkpoint_path)

            # ── 4. Post-process ──
            self._progress['message'] = 'Running post-processing (DBSCAN clustering)…'

            post_test_preds = batch_postprocess_predictions(
                test_preds,
                spatial_threshold=config['post_process']['spatial_threshold'],
                temporal_threshold=config['post_process']['temporal_threshold'],
                confidence_threshold=config['post_process']['confidence_threshold'],
                strategy=config['post_process']['strategy'],
                mode=config['post_process']['mode'],
                remove_outliers=config['post_process'].get('outlier_detection', False),
                outlier_method='isolation_forest',
            )

            # ── 5. Compute metrics ──
            self._progress['stage'] = 'metrics'
            self._progress['message'] = 'Computing STD-mAP metrics…'

            test_metrics = get_eval_metrics(
                test_preds, test_gts,
                pos_thresholds=config['eval']['pos_thresholds'],
                iou_threshold_range=config['eval']['iou_thresholds'],
                angular_thresholds=config['eval']['angular_thresholds'],
                match_pairs=config['eval']['match_pairs'],
                waggle_run_ids=waggle_run_ids,
                video_names=all_video_names,
            )

            post_test_metrics = get_eval_metrics(
                post_test_preds, test_gts,
                pos_thresholds=config['eval']['pos_thresholds'],
                iou_threshold_range=config['eval']['iou_thresholds'],
                angular_thresholds=config['eval']['angular_thresholds'],
                match_pairs=config['eval']['match_pairs'],
                waggle_run_ids=waggle_run_ids,
                video_names=all_video_names,
            )

            # ── 6. Per-video breakdown ──
            self._progress['message'] = 'Computing per-video breakdown…'
            per_video = {}
            unique_videos = sorted(set(all_video_names))
            for vname in unique_videos:
                mask = all_video_names == vname
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                v_preds = [test_preds[i] for i in indices]
                v_post = [post_test_preds[i] for i in indices]
                v_gts = [test_gts[i] for i in indices]
                v_run_ids = [waggle_run_ids[i] for i in indices]
                v_vnames = [all_video_names[i] for i in indices]

                v_metrics = get_eval_metrics(
                    v_preds, v_gts,
                    pos_thresholds=config['eval']['pos_thresholds'],
                    iou_threshold_range=config['eval']['iou_thresholds'],
                    angular_thresholds=config['eval']['angular_thresholds'],
                    match_pairs=config['eval']['match_pairs'],
                    waggle_run_ids=v_run_ids,
                    video_names=v_vnames,
                )
                v_post_metrics = get_eval_metrics(
                    v_post, v_gts,
                    pos_thresholds=config['eval']['pos_thresholds'],
                    iou_threshold_range=config['eval']['iou_thresholds'],
                    angular_thresholds=config['eval']['angular_thresholds'],
                    match_pairs=config['eval']['match_pairs'],
                    waggle_run_ids=v_run_ids,
                    video_names=v_vnames,
                )

                per_video[vname] = {
                    'pre': _eval_metrics_to_native(v_metrics),
                    'post': _eval_metrics_to_native(v_post_metrics),
                    'n_windows': int(len(indices)),
                }

            # Print to console (same as training)
            print_evaluation_results(test_metrics, post_test_metrics)

            # ── Store results ──
            import time
            results = {
                'pre': _eval_metrics_to_native(test_metrics),
                'post': _eval_metrics_to_native(post_test_metrics),
                'per_video': per_video,
                'checkpoint': pred_engine.checkpoint_meta,
                'config': {
                    'eval': {
                        'pos_thresholds': config['eval']['pos_thresholds'],
                        'iou_thresholds': config['eval']['iou_thresholds'],
                        'angular_thresholds': config['eval']['angular_thresholds'],
                        'match_pairs': config['eval']['match_pairs'],
                        'confidence_threshold': config['eval']['confidence_threshold'],
                        'max_dets': config['eval']['max_dets'],
                    },
                    'post_process': {
                        'spatial_threshold': config['post_process']['spatial_threshold'],
                        'temporal_threshold': config['post_process']['temporal_threshold'],
                        'confidence_threshold': config['post_process']['confidence_threshold'],
                        'strategy': config['post_process']['strategy'],
                        'mode': config['post_process']['mode'],
                    },
                },
                'n_val_windows': int(len(test_df)),
                'n_val_videos': int(len(unique_videos)),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }

            with self._lock:
                self._results = results
                self._results_ckpt = pred_engine.checkpoint_path

            self._save_disk_cache(ckpt_path=pred_engine.checkpoint_path)

            self._progress = {
                'stage': 'done',
                'batch_current': n_batches,
                'batch_total': n_batches,
                'message': 'Evaluation complete.',
            }
            print(f"  ✅ Evaluation complete. Results cached.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = str(e)
            self._progress = {
                'stage': 'error',
                'batch_current': 0,
                'batch_total': 0,
                'message': f'Error: {e}',
            }
        finally:
            self._running = False

    def has_cached_preds(self):
        """Check if decoded predictions are cached for re-clustering."""
        return self._cached_preds is not None

    def recluster(self, post_params, eval_config=None):
        """Re-run post-processing + metrics with new clustering params.

        This is very fast (~seconds) since it reuses cached decoded
        predictions and skips the expensive inference step entirely.

        Args:
            post_params: dict with keys like spatial_threshold, temporal_threshold,
                        confidence_threshold, strategy, mode, min_samples, etc.
            eval_config: optional dict to override eval thresholds (pos_thresholds,
                        iou_thresholds, angular_thresholds, match_pairs).
                        If None, uses the config cached from the original eval run.
        Returns:
            Updated results dict (same shape as full eval results).
        """
        from src.utils.postprocess import batch_postprocess_predictions
        from src.utils.eval_utils import get_eval_metrics, print_evaluation_results
        import time as _time

        if self._cached_preds is None:
            raise RuntimeError("No cached predictions. Run full evaluation first.")

        config = self._cached_config
        # Allow overriding eval thresholds with live config
        if eval_config:
            config = dict(config)  # shallow copy
            config['eval'] = {**config.get('eval', {}), **eval_config}
        test_preds = self._cached_preds
        test_gts = self._cached_gts
        all_video_names = self._cached_video_names

        # Merge defaults with provided params
        pp = {
            'spatial_threshold': config['post_process']['spatial_threshold'],
            'temporal_threshold': config['post_process']['temporal_threshold'],
            'confidence_threshold': config['post_process']['confidence_threshold'],
            'strategy': config['post_process']['strategy'],
            'mode': config['post_process']['mode'],
            'min_samples': 1,
        }
        pp.update(post_params)

        t0 = _time.time()

        # Re-run post-processing
        post_test_preds = batch_postprocess_predictions(
            test_preds,
            spatial_threshold=pp['spatial_threshold'],
            temporal_threshold=pp['temporal_threshold'],
            confidence_threshold=pp['confidence_threshold'],
            strategy=pp['strategy'],
            mode=pp['mode'],
            remove_outliers=pp.get('remove_outliers', False),
            outlier_method=pp.get('outlier_method', 'isolation_forest'),
            min_samples=pp.get('min_samples', 1),
        )

        # Re-compute metrics
        run_ids = self._cached_run_ids
        test_metrics = get_eval_metrics(
            test_preds, test_gts,
            pos_thresholds=config['eval']['pos_thresholds'],
            iou_threshold_range=config['eval']['iou_thresholds'],
            angular_thresholds=config['eval']['angular_thresholds'],
            match_pairs=config['eval']['match_pairs'],
            waggle_run_ids=run_ids,
            video_names=all_video_names,
        )

        post_test_metrics = get_eval_metrics(
            post_test_preds, test_gts,
            pos_thresholds=config['eval']['pos_thresholds'],
            iou_threshold_range=config['eval']['iou_thresholds'],
            angular_thresholds=config['eval']['angular_thresholds'],
            match_pairs=config['eval']['match_pairs'],
            waggle_run_ids=run_ids,
            video_names=all_video_names,
        )

        # Per-video breakdown
        per_video = {}
        unique_videos = sorted(set(all_video_names))
        for vname in unique_videos:
            mask = all_video_names == vname
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            v_preds = [test_preds[i] for i in indices]
            v_post = [post_test_preds[i] for i in indices]
            v_gts = [test_gts[i] for i in indices]
            v_run_ids = [run_ids[i] for i in indices] if run_ids else None
            v_vnames = [all_video_names[i] for i in indices]
            v_metrics = get_eval_metrics(
                v_preds, v_gts,
                pos_thresholds=config['eval']['pos_thresholds'],
                iou_threshold_range=config['eval']['iou_thresholds'],
                angular_thresholds=config['eval']['angular_thresholds'],
                match_pairs=config['eval']['match_pairs'],
                waggle_run_ids=v_run_ids,
                video_names=v_vnames,
            )
            v_post_metrics = get_eval_metrics(
                v_post, v_gts,
                pos_thresholds=config['eval']['pos_thresholds'],
                iou_threshold_range=config['eval']['iou_thresholds'],
                angular_thresholds=config['eval']['angular_thresholds'],
                match_pairs=config['eval']['match_pairs'],
                waggle_run_ids=v_run_ids,
                video_names=v_vnames,
            )
            per_video[vname] = {
                'pre': _eval_metrics_to_native(v_metrics),
                'post': _eval_metrics_to_native(v_post_metrics),
                'n_windows': int(len(indices)),
            }

        elapsed = _time.time() - t0
        print_evaluation_results(test_metrics, post_test_metrics)
        print(f"  ✅ Re-clustered in {elapsed:.1f}s with params: {pp}", flush=True)

        # Build results dict
        results = {
            'pre': _eval_metrics_to_native(test_metrics),
            'post': _eval_metrics_to_native(post_test_metrics),
            'per_video': per_video,
            'checkpoint': self._results.get('checkpoint') if self._results else {},
            'config': {
                'eval': {
                    'pos_thresholds': config['eval']['pos_thresholds'],
                    'iou_thresholds': config['eval']['iou_thresholds'],
                    'angular_thresholds': config['eval']['angular_thresholds'],
                    'match_pairs': config['eval']['match_pairs'],
                    'confidence_threshold': config['eval']['confidence_threshold'],
                    'max_dets': config['eval']['max_dets'],
                },
                'post_process': pp,
            },
            'n_val_windows': int(len(test_preds)),
            'n_val_videos': int(len(unique_videos)),
            'timestamp': _time.strftime('%Y-%m-%d %H:%M:%S'),
            'recluster_time': round(elapsed, 1),
        }

        # Update cached results
        with self._lock:
            self._results = results

        return results


def _eval_metrics_to_native(metrics):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(metrics, dict):
        return {k: _eval_metrics_to_native(v) for k, v in metrics.items()}
    elif isinstance(metrics, (list, tuple)):
        return [_eval_metrics_to_native(v) for v in metrics]
    elif isinstance(metrics, np.integer):
        return int(metrics)
    elif isinstance(metrics, np.floating):
        return float(metrics)
    elif isinstance(metrics, np.ndarray):
        return metrics.tolist()
    elif isinstance(metrics, float) and (np.isnan(metrics) or np.isinf(metrics)):
        return None  # JSON doesn't support nan/inf
    return metrics


evaluation_engine = EvaluationEngine()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recordings")
def api_recordings():
    """List all base recordings with variants and waggle counts."""
    result = []
    for base_id in sorted(recording_index):
        rec = recording_index[base_id]
        # Count unique waggle_run_ids across all variants
        run_ids = set(r["run_id"] for r in rec["waggle_runs"].values())
        # Tag each variant with train/val split
        variants_with_split = []
        for v in rec["variants"]:
            vc = dict(v)
            vc["split"] = "train" if v["video_name"] in train_videos else "val"
            variants_with_split.append(vc)
        result.append({
            "base_id": base_id,
            "variants": variants_with_split,
            "waggle_count": len(run_ids),
        })
    return jsonify(result)


@app.route("/api/video/<path:video_name>/info")
def api_video_info(video_name):
    """Get video file metadata (resolution, fps, frame count)."""
    try:
        info = frame_server.get_video_info(video_name)
        return jsonify(info)
    except FileNotFoundError:
        return jsonify({"error": f"Video not found: {video_name}"}), 404


@app.route("/api/video/<path:video_name>/annotations")
def api_video_annotations(video_name):
    """Get deduplicated waggle runs for a specific video file.
    Returns empty list for external/unannotated videos.
    """
    base, *_ = get_base_and_variant(video_name)
    if base not in recording_index:
        # External video — no annotations, return empty list (not 404)
        return jsonify([])

    runs = [
        r for r in recording_index[base]["waggle_runs"].values()
        if r["video_name"] == video_name
    ]
    runs.sort(key=lambda r: r["waggle_start"])
    return jsonify(runs)


@app.route("/api/frame/<path:video_name>/<int:frame_idx>")
def api_frame(video_name, frame_idx):
    """Serve a single decoded video frame as JPEG."""
    try:
        jpeg = frame_server.get_frame_jpeg(video_name, frame_idx)
    except FileNotFoundError:
        return Response("Video not found", status=404)

    if jpeg is None:
        return Response("Frame not found", status=404)

    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.route("/api/random")
def api_random():
    """Return a random recording base_id (that has files on disk)."""
    import random
    candidates = [
        base_id for base_id, rec in recording_index.items()
        if any(v["exists"] for v in rec["variants"])
    ]
    if not candidates:
        return jsonify({"error": "No videos found on disk"}), 404
    return jsonify({"base_id": random.choice(candidates)})


@app.route("/api/predictions/<path:video_name>")
def api_predictions(video_name):
    """Get model predictions for a video. Runs inference if not cached."""
    if not prediction_engine.is_loaded:
        return jsonify({"error": "No checkpoint loaded. Start server with --checkpoint flag."}), 503

    result = prediction_engine.get_predictions(video_name)
    if result is None:
        return jsonify({"error": "Prediction engine not ready"}), 503

    return jsonify({
        "raw": result['raw'],
        "postprocessed": result['postprocessed'],
        "n_windows": result['n_windows'],
        "checkpoint": prediction_engine.checkpoint_meta,
    })


@app.route("/api/cluster/<path:video_name>", methods=["POST"])
def api_cluster(video_name):
    """Re-cluster raw predictions with custom parameters.

    Accepts JSON body with clustering parameters and returns consolidated
    predictions plus the raw→cluster mapping (which raw detection belongs
    to which cluster). This powers the "Clustered" view mode and the
    interactive parameter sliders in the Pipeline Inspector.
    """
    from src.utils.postprocess import cluster_and_consolidate_waggles

    # Get raw predictions from whichever cache has them
    raw_preds = None
    cached = prediction_engine._cache.get(video_name)
    if cached:
        raw_preds = cached.get('raw')
    if raw_preds is None:
        sw_cached = sliding_inference.get_results(video_name)
        if sw_cached:
            raw_preds = sw_cached.get('raw')
    if raw_preds is None:
        return jsonify({"error": "No cached predictions for this video. Run inference first."}), 404

    # Parse clustering parameters from request body
    params = request.get_json() or {}
    spatial_threshold = float(params.get('spatial_threshold', 30.0))
    temporal_threshold = int(params.get('temporal_threshold', 8))
    confidence_threshold = float(params.get('confidence_threshold', 0.0))
    remove_outliers = bool(params.get('remove_outliers', False))
    outlier_method = params.get('outlier_method', 'density')
    outlier_min_neighbors = int(params.get('outlier_min_neighbors', 2))
    clustering_method = params.get('clustering_method', 'dbscan')
    min_samples = int(params.get('min_samples', 1))
    mode = params.get('mode', 'median')

    # Run clustering with mapping
    result = cluster_and_consolidate_waggles(
        raw_preds,
        spatial_threshold=spatial_threshold,
        temporal_threshold=temporal_threshold,
        min_confidence=confidence_threshold,
        mode=mode,
        remove_outliers=remove_outliers,
        outlier_method=outlier_method,
        outlier_min_neighbors=outlier_min_neighbors,
        clustering_method=clustering_method,
        return_mapping=True,
        min_samples=min_samples,
    )

    # Convert numpy types for JSON serialization
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return jsonify(_to_native({
        'raw': raw_preds,
        'consolidated': result['consolidated'],
        'labels': result['labels'],
        'filtered_indices': result['filtered_indices'],
        'outlier_indices': result['outlier_indices'],
        'n_clusters': result['n_clusters'],
        'n_noise': result['n_noise'],
        'params': {
            'spatial_threshold': spatial_threshold,
            'temporal_threshold': temporal_threshold,
            'confidence_threshold': confidence_threshold,
            'remove_outliers': remove_outliers,
            'clustering_method': clustering_method,
            'min_samples': min_samples,
            'mode': mode,
        },
    }))


@app.route("/api/checkpoint/info")
def api_checkpoint_info():
    """Return info about the currently loaded checkpoint."""
    if not prediction_engine.is_loaded:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded": True,
        **prediction_engine.checkpoint_meta,
    })


@app.route("/api/checkpoint/load", methods=["POST"])
def api_checkpoint_load():
    """Hot-reload a different checkpoint."""
    data = request.get_json()
    path = data.get('path', '') if data else ''
    if not path or not os.path.exists(path):
        return jsonify({"error": f"Checkpoint not found: {path}"}), 404
    try:
        prediction_engine.load_checkpoint(path)
        return jsonify({"success": True, **prediction_engine.checkpoint_meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/device", methods=["GET"])
def api_device_get():
    """Return the current inference device."""
    import torch
    device_str = str(prediction_engine.device) if prediction_engine.device else "none"
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
    return jsonify({
        "device": device_str,
        "cuda_available": cuda_available,
        "cuda_name": cuda_name,
    })


@app.route("/api/device", methods=["POST"])
def api_device_set():
    """Switch inference device at runtime (e.g. 'cpu' ↔ 'cuda:0').

    Moves the model to the new device and clears prediction caches.
    """
    import torch

    if not prediction_engine.is_loaded:
        return jsonify({"error": "No model loaded"}), 400

    data = request.get_json() or {}
    new_device_str = data.get('device', 'cpu')

    # Validate
    try:
        new_device = torch.device(new_device_str)
        if 'cuda' in new_device_str and not torch.cuda.is_available():
            return jsonify({"error": "CUDA is not available on this machine"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid device: {e}"}), 400

    old_device = str(prediction_engine.device)
    if str(new_device) == old_device:
        return jsonify({"device": old_device, "changed": False})

    try:
        with prediction_engine._lock:
            # Ensure model is float32 before moving (avoids half-precision
            # residue from prior autocast causing device mismatches)
            prediction_engine.model = prediction_engine.model.float().to(new_device)
            prediction_engine.device = new_device
            prediction_engine._cache.clear()
            # Verify the move actually worked
            p = next(prediction_engine.model.parameters())
            assert str(p.device) == str(new_device), \
                f"Model param on {p.device}, expected {new_device}"
        # Also clear sliding inference cache
        with sliding_inference._lock:
            sliding_inference._cache.clear()

        print(f"  Device switched: {old_device} → {new_device}")
        return jsonify({
            "device": str(new_device),
            "changed": True,
            "previous": old_device,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to switch device: {e}"}), 500


# ---------------------------------------------------------------------------
# Evaluation API  — run full STD-mAP on the validation set
# ---------------------------------------------------------------------------

@app.route("/eval")
def eval_page():
    """Serve the evaluation dashboard page."""
    return render_template("eval.html")


@app.route("/api/eval/run", methods=["POST"])
def api_eval_run():
    """Trigger evaluation on the validation split (async)."""
    if not prediction_engine.is_loaded:
        return jsonify({"error": "No checkpoint loaded. Start with --checkpoint."}), 503

    # Check if already running
    if evaluation_engine.is_running:
        return jsonify({"status": "already_running", **evaluation_engine.get_progress()})

    # Check if results already cached for this checkpoint
    cached = evaluation_engine.get_results()
    if cached and evaluation_engine._results_ckpt == prediction_engine.checkpoint_path:
        return jsonify({"status": "cached", "message": "Results already available."})

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    ok = evaluation_engine.run_async(prediction_engine, config)
    if ok:
        return jsonify({"status": "started", "message": "Evaluation started."})
    else:
        return jsonify({"status": "error", "message": "Could not start evaluation."}), 500


@app.route("/api/eval/status")
def api_eval_status():
    """Poll evaluation progress."""
    progress = evaluation_engine.get_progress()
    progress['running'] = evaluation_engine.is_running
    progress['has_results'] = evaluation_engine.get_results() is not None
    progress['has_cached_preds'] = evaluation_engine.has_cached_preds()

    # Flag if cache is from a different model (compare epoch since best.pth gets overwritten)
    cached_results = evaluation_engine.get_results()
    if (prediction_engine.is_loaded and cached_results
            and cached_results.get('checkpoint', {}).get('epoch')
               != prediction_engine.checkpoint_meta.get('epoch')):
        progress['cache_stale'] = True
        progress['cache_epoch'] = cached_results.get('checkpoint', {}).get('epoch', '?')

    progress['model_loaded'] = prediction_engine.is_loaded
    if prediction_engine.is_loaded:
        progress['checkpoint'] = prediction_engine.checkpoint_meta
    return jsonify(progress)


@app.route("/api/eval/results")
def api_eval_results():
    """Get cached evaluation results."""
    results = evaluation_engine.get_results()
    if results is None:
        return jsonify({"error": "No evaluation results. Run evaluation first."}), 404
    return jsonify(results)


@app.route("/api/eval/recluster", methods=["POST"])
def api_eval_recluster():
    """Re-run post-processing + metrics with new clustering parameters.

    Reuses cached decoded predictions from the last eval run, so this
    is very fast (~seconds).  Accepts JSON body with clustering params:
      spatial_threshold, temporal_threshold, confidence_threshold,
      strategy, mode, min_samples, remove_outliers, outlier_method
    """
    if not evaluation_engine.has_cached_preds():
        return jsonify({"error": "No cached predictions. Run full evaluation first."}), 400

    if evaluation_engine.is_running:
        return jsonify({"error": "Evaluation is currently running."}), 409

    params = request.get_json(force=True, silent=True) or {}

    # Coerce types
    pp = {}
    if 'spatial_threshold' in params:
        pp['spatial_threshold'] = float(params['spatial_threshold'])
    if 'temporal_threshold' in params:
        pp['temporal_threshold'] = int(params['temporal_threshold'])
    if 'confidence_threshold' in params:
        pp['confidence_threshold'] = float(params['confidence_threshold'])
    if 'strategy' in params:
        pp['strategy'] = str(params['strategy'])
    if 'mode' in params:
        pp['mode'] = str(params['mode'])
    if 'min_samples' in params:
        pp['min_samples'] = int(params['min_samples'])
    if 'remove_outliers' in params:
        pp['remove_outliers'] = bool(params['remove_outliers'])
    if 'outlier_method' in params:
        pp['outlier_method'] = str(params['outlier_method'])
    if 'clustering_method' in params:
        pp['clustering_method'] = str(params['clustering_method'])

    try:
        results = evaluation_engine.recluster(pp, eval_config=_config['eval'])
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# External Video & Inference API
# ---------------------------------------------------------------------------

@app.route("/api/external_videos")
def api_external_videos():
    """List external (unannotated) videos available for inference."""
    videos = []
    for vname in sorted(_external_video_paths.keys()):
        vpath = _external_video_paths[vname]
        try:
            cap = cv2.VideoCapture(vpath)
            info = {
                'video_name': vname,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                'has_results': vname in sliding_inference._cache,
            }
            duration_s = info['total_frames'] / max(info['fps'], 1)
            info['duration'] = f"{int(duration_s // 60)}:{int(duration_s % 60):02d}"
            cap.release()
            videos.append(info)
        except Exception:
            videos.append({'video_name': vname, 'error': 'Cannot read video'})
    return jsonify({
        'videos': videos,
        'directory': EXTERNAL_VIDEO_DIR or '',
        'model_loaded': prediction_engine.is_loaded,
    })


@app.route("/api/infer/<path:video_name>", methods=["POST"])
def api_infer(video_name):
    """Run sliding-window inference on an external video.
    Returns results synchronously (may take minutes for long videos).
    """
    if not prediction_engine.is_loaded:
        return jsonify({"error": "No model checkpoint loaded. Start with --checkpoint."}), 503

    # Check video exists
    try:
        resolve_video_path(video_name)
    except FileNotFoundError:
        return jsonify({"error": f"Video not found: {video_name}"}), 404

    data = request.get_json() or {}
    temporal_stride = int(data.get('temporal_stride', 8))

    try:
        result = sliding_inference.run_inference(video_name, temporal_stride=temporal_stride)
        return jsonify({
            'raw': result['raw'],
            'postprocessed': result['postprocessed'],
            'n_windows': result['n_windows'],
            'video_info': result['video_info'],
            'checkpoint': prediction_engine.checkpoint_meta,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/infer/<path:video_name>/status")
def api_infer_status(video_name):
    """Poll inference progress."""
    # Check if results are cached
    cached = sliding_inference.get_results(video_name)
    if cached:
        return jsonify({
            'status': 'done',
            'n_raw': len(cached['raw']),
            'n_postprocessed': len(cached['postprocessed']),
        })

    current, total = sliding_inference.get_progress(video_name)
    if total > 0:
        return jsonify({
            'status': 'running',
            'current': current,
            'total': total,
            'progress': round(current / total * 100, 1),
        })

    return jsonify({'status': 'idle'})


@app.route("/api/infer/<path:video_name>/results")
def api_infer_results(video_name):
    """Get cached inference results."""
    cached = sliding_inference.get_results(video_name)
    if not cached:
        return jsonify({"error": "No results. Run inference first."}), 404
    return jsonify({
        'raw': cached['raw'],
        'postprocessed': cached['postprocessed'],
        'n_windows': cached['n_windows'],
        'video_info': cached['video_info'],
        'checkpoint': prediction_engine.checkpoint_meta,
    })


# ---------------------------------------------------------------------------
# Pipeline Inspector — step-by-step data pipeline visualization
# ---------------------------------------------------------------------------

# Build a Dataset instance for inspection (same config as main.py uses)
import torchvision.transforms as T
with open(CONFIG_PATH) as _cf:
    _insp_config = yaml.safe_load(_cf)
_insp_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((_insp_config['augmentations']['width'],
              _insp_config['augmentations']['height'])),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
])
from src.data.dataset import VideoYoloDataset as _DatasetClass
_inspector_ds = _DatasetClass(
    dataframe=annotations_df,
    video_dir=VIDEO_DIR,
    transform=_insp_transform,
    width=_insp_config['data']['width'],
    height=_insp_config['data']['height'],
    window_size=_insp_config['data']['window_size'],
    grid_size=_insp_config['model']['grid_size'],
    augment=None,
    is_training=False,
)


def _np_to_jpeg(frame_np, quality=90):
    """Encode a numpy (H,W,3) RGB frame as base64 JPEG."""
    bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('ascii')


def _tensor_to_jpeg(t, greyscale=False, quality=90):
    """Encode a (C,H,W) float tensor [0..1] as base64 JPEG."""
    import torch
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    if greyscale and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('ascii')


@app.route("/api/pipeline/<path:video_name>")
def api_pipeline(video_name):
    """Return one stage of the data pipeline for a specific annotation window.

    This endpoint delegates ALL computation to Dataset.get_item_debug(),
    ensuring the inspector shows exactly what the training pipeline produces.

    Query params:
        stage: 'raw' | 'crop' | 'transform' | 'sample' | 'target' | 'output'
        row_idx: index into this video's annotation rows (default: 0)
        crop_mode: 'inference' (centered) or 'training' (random offset)
        n_variance: number of additional random crop samples to show (crop stage only)
        augment: 'true' to apply full augmentation pipeline (transform stage only)
        aug_seed: random seed for augmentation (default: random)
    """
    import torch

    stage = request.args.get('stage', 'raw')
    row_idx = int(request.args.get('row_idx', 0))
    crop_mode = request.args.get('crop_mode', 'inference')
    n_variance = int(request.args.get('n_variance', 0))
    do_augment = request.args.get('augment', 'false').lower() == 'true'
    aug_seed = int(request.args.get('aug_seed', 42))

    # Find the global index for this video + row_idx
    video_mask = annotations_df['video_name'] == video_name
    video_indices = annotations_df.index[video_mask].tolist()
    if row_idx >= len(video_indices):
        return jsonify({"error": f"row_idx {row_idx} >= {len(video_indices)} rows"}), 404

    global_idx = video_indices[row_idx]

    # ── Run the actual pipeline ──
    try:
        dbg = _inspector_ds.get_item_debug(global_idx, crop_mode=crop_mode)
    except Exception as e:
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    meta = dict(dbg['metadata'])
    meta['row_idx'] = row_idx
    meta['total_rows'] = len(video_indices)
    meta['crop_mode'] = crop_mode

    # ── Stage: RAW ──────────────────────────────────────────────────────
    if stage == 'raw':
        frame_images = [_np_to_jpeg(f, quality=85) for f in dbg['raw_frames']]
        cp = dbg['crop_params']
        meta.update({
            'video_resolution': meta['resolution'],
            'n_frames_loaded': len(dbg['raw_frames']),
            'needs_sampling': len(dbg['raw_frames']) != meta['window_size'],
            'crop_box': {
                'x_min': cp['x_min'], 'y_min': cp['y_min'],
                'x_max': cp['x_max'], 'y_max': cp['y_max'],
            },
        })
        return jsonify({'stage': 'raw', 'frames': frame_images, 'metadata': meta})

    # ── Stage: CROP ─────────────────────────────────────────────────────
    elif stage == 'crop':
        frame_images = [_np_to_jpeg(f) for f in dbg['cropped_np']]
        cp = dbg['crop_params']
        meta.update({
            'video_resolution': meta['resolution'],
            'n_frames': len(dbg['cropped_np']),
            'crop_origin': [cp['x_min'], cp['y_min']],
            'crop_offset': [cp['offset_x'], cp['offset_y']],
        })

        # Variance crops: show multiple random crop positions
        # Uses the same get_item_debug with 'training' mode but different seeds
        variance_crops = []
        if n_variance > 0:
            raw_frames = dbg['raw_frames']
            first_frame = raw_frames[0]
            if isinstance(first_frame, np.ndarray):
                original_h, original_w = first_frame.shape[:2]
            else:
                original_h, original_w = first_frame.shape[-2:]

            gt_x = meta['gt_position'][0]
            gt_y = meta['gt_position'][1]
            crop_h, crop_w = _insp_config['data']['height'], _insp_config['data']['width']

            for vi in range(n_variance):
                # Same random crop logic as dataset.py __getitem__ L142-156
                rng = np.random.RandomState(seed=global_idx * 1000 + vi + 1)
                margin_x = int(crop_w * 0.2)
                margin_y = int(crop_h * 0.2)
                max_offset_x = (crop_w // 2) - margin_x
                max_offset_y = (crop_h // 2) - margin_y
                v_off_x = rng.randint(-max_offset_x, max_offset_x + 1)
                v_off_y = rng.randint(-max_offset_y, max_offset_y + 1)

                x_min_v = max(0, min(original_w - crop_w, int(gt_x - crop_w / 2) + v_off_x))
                y_min_v = max(0, min(original_h - crop_h, int(gt_y - crop_h / 2) + v_off_y))

                # Crop first frame only (enough to show spatial variance)
                cropped_v = raw_frames[0][y_min_v:y_min_v + crop_h, x_min_v:x_min_v + crop_w]
                bee_in_crop_x = gt_x - x_min_v
                bee_in_crop_y = gt_y - y_min_v

                variance_crops.append({
                    'image': _np_to_jpeg(cropped_v),
                    'offset_x': int(v_off_x),
                    'offset_y': int(v_off_y),
                    'bee_in_crop': [round(bee_in_crop_x, 1), round(bee_in_crop_y, 1)],
                })

        return jsonify({
            'stage': 'crop', 'frames': frame_images, 'metadata': meta,
            'variance_crops': variance_crops,
        })

    # ── Stage: TRANSFORM ────────────────────────────────────────────────
    elif stage == 'transform':
        frames_to_show = list(dbg['transformed'])

        frame_images = [_tensor_to_jpeg(t, greyscale=True) for t in frames_to_show]
        pixel_stats = [{
            'min': round(float(t.min()), 4),
            'max': round(float(t.max()), 4),
            'mean': round(float(t.mean()), 4),
        } for t in frames_to_show]
        meta.update({
            'n_frames': len(frame_images),
            'pixel_stats': pixel_stats,
        })
        return jsonify({'stage': 'transform', 'frames': frame_images, 'metadata': meta})

    # ── Stage: AUGMENT ──────────────────────────────────────────────────
    # Applies the full WaggleAugmentations pipeline (spatial + photometric
    # transforms + normalization) — exactly as in training.
    # Frames are de-normalized for display.
    elif stage == 'augment':
        config = _insp_config
        mean = config['augmentations']['mean']
        std = config['augmentations']['std']
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)

        augmenter = _load_augmenter()
        aug_info_list = []

        if augmenter and do_augment:
            import random as _random
            _random.seed(aug_seed)
            np.random.seed(aug_seed)

            cp = dbg['crop_params']
            gt_x = meta['gt_position'][0]
            gt_y = meta['gt_position'][1]
            target_dict = {
                'x': gt_x - cp['x_min'],
                'y': gt_y - cp['y_min'],
                'dir_x': meta['gt_direction'][0],
                'dir_y': meta['gt_direction'][1],
            }
            aug_frames, target_dict, aug_info_list = augmenter(
                [t.clone() for t in dbg['transformed']], target_dict
            )
            meta['augmented_target'] = {
                'x': round(float(target_dict['x']), 1),
                'y': round(float(target_dict['y']), 1),
                'dir_x': round(float(target_dict['dir_x']), 3),
                'dir_y': round(float(target_dict['dir_y']), 3),
            }
        else:
            # Fallback: just normalize without augmentation
            norm = T.Normalize(mean=mean, std=std)
            aug_frames = [norm(t.clone()) for t in dbg['transformed']]

        # De-normalize for human-readable display
        frame_images = []
        pixel_stats = []
        for t in aug_frames:
            pixel_stats.append({
                'min': round(float(t.min()), 4),
                'max': round(float(t.max()), 4),
                'mean': round(float(t.mean()), 4),
            })
            t_denorm = t * std_t + mean_t
            frame_images.append(_tensor_to_jpeg(t_denorm.clamp(0, 1), greyscale=True))

        meta.update({
            'n_frames': len(frame_images),
            'normalization': {'mean': mean, 'std': std},
            'pixel_stats': pixel_stats,
            'aug_info': aug_info_list,
        })
        return jsonify({'stage': 'augment', 'frames': frame_images, 'metadata': meta})

    # ── Stage: SAMPLE ───────────────────────────────────────────────────
    # In training: augment → sample.  We apply augmentation + normalize first,
    # then show only the sampled subset of frames.
    elif stage == 'sample':
        config = _insp_config
        mean = config['augmentations']['mean']
        std = config['augmentations']['std']
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)

        # Apply augmentation to transformed frames (same as augment stage)
        augmenter = _load_augmenter()
        if augmenter and do_augment:
            import random as _random
            _random.seed(aug_seed)
            np.random.seed(aug_seed)

            cp = dbg['crop_params']
            target_dict = {
                'x': meta['gt_position'][0] - cp['x_min'],
                'y': meta['gt_position'][1] - cp['y_min'],
                'dir_x': meta['gt_direction'][0],
                'dir_y': meta['gt_direction'][1],
            }
            aug_frames, _, _ = augmenter(
                [t.clone() for t in dbg['transformed']], target_dict
            )
        else:
            norm = T.Normalize(mean=mean, std=std)
            aug_frames = [norm(t.clone()) for t in dbg['transformed']]

        # Subsample using the same indices the dataset chose
        sampled_indices = dbg['sampled_indices']
        sampled = [aug_frames[i] for i in sampled_indices if i < len(aug_frames)]

        # De-normalize for display
        frame_images = []
        for t in sampled:
            t_denorm = t * std_t + mean_t
            frame_images.append(_tensor_to_jpeg(t_denorm.clamp(0, 1), greyscale=True))

        meta.update({
            'n_frames_loaded': meta['n_frames_loaded'],
            'n_frames_sampled': meta['window_size'],
            'sampled_indices': sampled_indices,
            'sampling_method': dbg['sampling_method'],
        })
        return jsonify({'stage': 'sample', 'frames': frame_images, 'metadata': meta})

    # ── Stage: TARGET ───────────────────────────────────────────────────
    elif stage == 'target':
        grid_size = _insp_config['model']['grid_size']
        meta.update({
            'grid_size': grid_size,
            'target': dbg['target_info'],
        })

        # Compute augmented target grid if augmentation requested
        if do_augment and dbg['target_info'].get('objectness', 0) == 1.0:
            augmenter = _load_augmenter()
            if augmenter:
                import random as _random
                _random.seed(aug_seed)
                np.random.seed(aug_seed)

                cp = dbg['crop_params']
                gt_x = meta['gt_position'][0]
                gt_y = meta['gt_position'][1]
                target_dict = {
                    'x': gt_x - cp['x_min'],
                    'y': gt_y - cp['y_min'],
                    'dir_x': meta['gt_direction'][0],
                    'dir_y': meta['gt_direction'][1],
                }
                _, aug_target, aug_info_list = augmenter(
                    [t.clone() for t in dbg['transformed']], target_dict
                )

                # Encode augmented coords into grid (same logic as dataset.py)
                W = _insp_config['data']['width']
                H = _insp_config['data']['height']
                ax = float(aug_target['x'])
                ay = float(aug_target['y'])
                x_norm = ax / W
                y_norm = ay / H
                dir_x_n = float(aug_target['dir_x'])
                dir_y_n = float(aug_target['dir_y'])
                d_norm = np.sqrt(dir_x_n**2 + dir_y_n**2)
                if d_norm > 0:
                    dir_x_n /= d_norm
                    dir_y_n /= d_norm

                grid_x = max(0, min(grid_size - 1, int(x_norm * grid_size)))
                grid_y = max(0, min(grid_size - 1, int(y_norm * grid_size)))
                cell_x = (x_norm * grid_size) - grid_x
                cell_y = (y_norm * grid_size) - grid_y

                meta['augmented_target_info'] = {
                    'objectness': 1.0,
                    'grid_cell': [grid_y, grid_x],
                    'cell_coords': [round(cell_x, 4), round(cell_y, 4)],
                    'position_norm': [round(x_norm, 4), round(y_norm, 4)],
                    'direction': [round(dir_x_n, 4), round(dir_y_n, 4)],
                    'aug_info': aug_info_list,
                }

        return jsonify({'stage': 'target', 'frames': [], 'metadata': meta})

    # ── Stage: OUTPUT ───────────────────────────────────────────────────
    elif stage == 'output':
        if not prediction_engine.is_loaded:
            return jsonify({'stage': 'output', 'frames': [],
                            'metadata': {**meta, 'error': 'No model loaded'}}), 200

        # Use the actual tensor from the pipeline (frames_tensor is model-ready)
        config = _insp_config
        mean = config['augmentations']['mean']
        std = config['augmentations']['std']
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)

        # Normalize (transform didn't include normalization, add it for inference)
        norm = T.Normalize(mean=mean, std=std)
        normalized = [norm(t) for t in dbg['sampled']]
        video_tensor = torch.stack(normalized).permute(1, 0, 2, 3)  # (C, T, H, W)
        batch = video_tensor.unsqueeze(0).to(prediction_engine.device)

        dev_type = prediction_engine.device.type  # 'cuda' or 'cpu'
        amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16
        with torch.no_grad():
            with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                output = prediction_engine.model(batch)

        from src.utils.eval_utils import yolo_to_img_space
        start_frame = meta['start_frame']
        end_frame = meta['end_frame']
        crop_size = (config['data']['width'], config['data']['height'])
        decoded = yolo_to_img_space(
            output, np.array([start_frame]), np.array([end_frame]),
            window_size=config['data']['window_size'],
            confidence_threshold=0.001,
            original_size=crop_size,
            max_dets=config['eval']['max_dets'],
        )

        cp = dbg['crop_params']
        preds = []
        for pred in decoded[0]:
            pred['position'] = [
                pred['position'][0] + cp['x_min'],
                pred['position'][1] + cp['y_min'],
            ]
            preds.append(pred)

        grid_size = config['model']['grid_size']
        obj_grid = torch.sigmoid(output[0, :, :, 0, 0]).cpu().numpy()
        heatmap_data = obj_grid.tolist()

        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_native(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        preds = [_to_native(p) for p in preds]

        meta.update({
            'crop_origin': [cp['x_min'], cp['y_min']],
            'n_predictions': len(preds),
            'predictions': preds,
            'confidence_heatmap': heatmap_data,
            'grid_size': grid_size,
            'checkpoint': prediction_engine.checkpoint_meta,
        })
        return jsonify({'stage': 'output', 'frames': [], 'metadata': meta})

    else:
        return jsonify({"error": f"Unknown stage: {stage}"}), 400


@app.route("/api/pipeline/rows/<path:video_name>")
def api_pipeline_rows(video_name):
    """Return list of annotation rows for a video, for the inspector row selector."""
    video_rows = annotations_df[annotations_df['video_name'] == video_name]
    rows = []
    for i, (_, row) in enumerate(video_rows.iterrows()):
        rows.append({
            'row_idx': i,
            'start_frame': int(row['start_frame']),
            'end_frame': int(row['end_frame']),
            'waggle': int(row.get('waggle', 0)),
            'run_id': int(row.get('waggle_run_id', -1)),
            'x': float(row['x1']),
            'y': float(row['y1']),
        })
    return jsonify({'video_name': video_name, 'rows': rows})


# ---------------------------------------------------------------------------
# Model View — exact training pipeline crop + augmentation
# ---------------------------------------------------------------------------

def _load_augmenter():
    """Lazily load the augmenter from training config."""
    if not hasattr(_load_augmenter, '_instance'):
        try:
            from src.data.augmentation import WaggleAugmentations
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)
            aug_cfg = config['augmentations']
            _load_augmenter._instance = WaggleAugmentations(
                width=aug_cfg['width'], height=aug_cfg['height'],
                prob_flip_h=aug_cfg['prob_flip_h'],
                prob_flip_v=aug_cfg['prob_flip_v'],
                prob_rotate=aug_cfg['prob_rotate'],
                rotate_range=aug_cfg['rotate_range'],
                prob_scale=aug_cfg['prob_scale'],
                scale_range=aug_cfg['scale_range'],
                prob_translate=aug_cfg['prob_translate'],
                translate_range=aug_cfg['translate_range'],
                prob_hsv=aug_cfg['prob_hsv'],
                hsv_hue=aug_cfg['hsv_hue'],
                hsv_saturation=aug_cfg['hsv_saturation'],
                hsv_value=aug_cfg['hsv_value'],
                prob_brightness=aug_cfg['prob_brightness'],
                brightness_range=aug_cfg['brightness_range'],
                prob_contrast=aug_cfg['prob_contrast'],
                contrast_range=aug_cfg['contrast_range'],
                prob_gamma=aug_cfg['prob_gamma'],
                gamma_range=aug_cfg['gamma_range'],
                prob_blur=aug_cfg['prob_blur'],
                blur_range=aug_cfg['blur_range'],
                prob_clahe=aug_cfg['prob_clahe'],
                clahe_clip_limit=aug_cfg['clahe_clip_limit'],
                clahe_tile_grid_size=aug_cfg['clahe_tile_grid_size'],
                prob_color_shuffle=aug_cfg['prob_color_shuffle'],
                prob_posterize=aug_cfg['prob_posterize'],
                posterize_bits=aug_cfg['posterize_bits'],
                prob_greyscale=aug_cfg['prob_greyscale'],
                normalize=True,  # Normalize — augment stage will de-normalize for display
                mean=aug_cfg['mean'], std=aug_cfg['std'],
            )
            print("  Loaded augmenter from config")
        except Exception as e:
            print(f"  Warning: could not load augmenter: {e}")
            _load_augmenter._instance = None
    return _load_augmenter._instance


def _frame_to_tensor(frame_bgr):
    """BGR numpy → CHW float32 tensor, matching dataset.py pipeline."""
    import torch
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    return to_tensor(frame_bgr)  # HWC uint8 → CHW float [0,1]



@app.route("/api/model_view/<path:video_name>")
def api_model_view(video_name):
    """Compute and return the model's view for a given frame + bee position.

    Query params:
        frame: frame index
        x, y: bee position in full-resolution coordinates
        dir_x, dir_y: direction vector
        seed: random seed for crop offset (default: 42)
        augment: 'true' to also return augmented version
    """
    import torch
    from torchvision import transforms as T

    frame_idx = int(request.args.get('frame', 0))
    bee_x = float(request.args.get('x', 0))
    bee_y = float(request.args.get('y', 0))
    dir_x = float(request.args.get('dir_x', 1.0))
    dir_y = float(request.args.get('dir_y', 0.0))
    seed = int(request.args.get('seed', 42))
    do_augment = request.args.get('augment', 'false').lower() == 'true'

    # --- Load raw frame ---
    cap, lock = frame_server._get_cap(video_name)
    with lock:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
    if not ret:
        return jsonify({"error": "Cannot read frame"}), 404

    original_h, original_w = frame_bgr.shape[:2]
    crop_h, crop_w = CROP_SIZE, CROP_SIZE

    # --- Replicate exact crop logic from dataset.py L140-156 ---
    rng = np.random.RandomState(seed=seed)
    margin_x = int(crop_w * 0.2)
    margin_y = int(crop_h * 0.2)
    max_offset_x = (crop_w // 2) - margin_x
    max_offset_y = (crop_h // 2) - margin_y
    offset_x = rng.randint(-max_offset_x, max_offset_x + 1)
    offset_y = rng.randint(-max_offset_y, max_offset_y + 1)

    x_min_ideal = int(bee_x - crop_w / 2) + offset_x
    y_min_ideal = int(bee_y - crop_h / 2) + offset_y
    x_min = max(0, min(original_w - crop_w, x_min_ideal))
    y_min = max(0, min(original_h - crop_h, y_min_ideal))
    x_max = x_min + crop_w
    y_max = y_min + crop_h

    # Crop
    crop_bgr = frame_bgr[y_min:y_max, x_min:x_max]

    # Bee position in crop space
    bee_in_crop_x = bee_x - x_min
    bee_in_crop_y = bee_y - y_min

    # --- Apply transforms: Grayscale (matching dataset.py pipeline) ---
    # ToTensor → ToPILImage → Resize(224) → Grayscale(3) → ToTensor
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_tensor = _frame_to_tensor(crop_rgb)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((CROP_SIZE, CROP_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])
    crop_transformed = transform(crop_tensor)

    # Encode crop as base64 JPEG
    crop_b64 = _tensor_to_jpeg(crop_transformed, greyscale=True)

    result = {
        "crop": {
            "x_min": int(x_min), "y_min": int(y_min),
            "x_max": int(x_max), "y_max": int(y_max),
            "offset_x": int(offset_x), "offset_y": int(offset_y),
            "bee_in_crop": {"x": round(bee_in_crop_x, 1), "y": round(bee_in_crop_y, 1)},
        },
        "video_resolution": {"w": original_w, "h": original_h},
        "crop_to_video_ratio": round(CROP_SIZE / max(original_w, original_h), 3),
        "crop_image": crop_b64,
    }

    # --- Augmentation (optional) ---
    if do_augment:
        augmenter = _load_augmenter()
        if augmenter:
            target_dict = {
                "x": bee_in_crop_x,
                "y": bee_in_crop_y,
                "dir_x": dir_x,
                "dir_y": dir_y,
            }
            # Augment expects a list of CHW tensors
            aug_frames, aug_target, aug_info = augmenter(
                [crop_transformed.clone()], target_dict
            )
            # De-normalize for display (augmenter now normalizes)
            aug_frame = aug_frames[0]
            mean_t = torch.tensor(config['augmentations']['mean']).view(3, 1, 1)
            std_t = torch.tensor(config['augmentations']['std']).view(3, 1, 1)
            aug_frame_display = (aug_frame * std_t + mean_t).clamp(0, 1)
            result["augmented_image"] = _tensor_to_jpeg(aug_frame_display, greyscale=True)
            result["augmented_target"] = {
                "x": round(float(aug_target["x"]), 1),
                "y": round(float(aug_target["y"]), 1),
                "dir_x": round(float(aug_target["dir_x"]), 3),
                "dir_y": round(float(aug_target["dir_y"]), 3),
            }
            result["aug_info"] = aug_info if aug_info else []
        else:
            result["augmented_image"] = None
            result["aug_info"] = ["Augmenter not available"]

    return jsonify(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waggle Dance Annotation Viewer")
    parser.add_argument("--host", default="localhost",
                        help="Bind address (use 0.0.0.0 for public)")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth) for prediction overlay")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model inference: 'cpu', 'cuda:0', etc. "
                             "Default: auto (GPU if available, else CPU)")
    parser.add_argument("--external-videos", type=str, default=None,
                        help="Path to directory with unannotated videos for inference")
    args = parser.parse_args()

    # Set up external videos directory
    if args.external_videos:
        EXTERNAL_VIDEO_DIR = os.path.abspath(args.external_videos)
        ext_videos = scan_external_videos()
        print(f"\nExternal videos: {len(ext_videos)} found in {EXTERNAL_VIDEO_DIR}")

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading model checkpoint...")
        prediction_engine.load_checkpoint(args.checkpoint, device_override=args.device)

    print(f"\n{'=' * 60}")
    print("  🐝  Waggle Dance Annotation Viewer")
    print(f"{'=' * 60}")
    print(f"  Server:  http://{args.host}:{args.port}")
    if args.host == "localhost":
        print(f"  Tunnel:  ssh -L {args.port}:localhost:{args.port} landgraf@horus")
    else:
        print(f"  Direct:  http://160.45.38.75:{args.port}")
    if prediction_engine.is_loaded:
        meta = prediction_engine.checkpoint_meta
        print(f"  Model:   {meta['filename']} (epoch {meta['epoch']})")
    else:
        print(f"  Model:   None (start with --checkpoint to enable predictions)")
    if EXTERNAL_VIDEO_DIR:
        print(f"  Videos:  {EXTERNAL_VIDEO_DIR} ({len(_external_video_paths)} external)")
    print(f"{'=' * 60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
