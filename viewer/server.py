"""
Waggle Dance Annotation Viewer — Flask Server

Serves video frames on-demand, annotation data, and model predictions
for visual inspection of the waggle dance training dataset.

Usage:
    python3 viewer/server.py                                        # localhost:5050
    python3 viewer/server.py --checkpoint ./ckpt/run/best.pth       # with model predictions
    python3 viewer/server.py --host 0.0.0.0 --checkpoint ./ckpt/run/best.pth
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

FRAME_CACHE_SIZE = 512   # max decoded JPEG frames in memory
JPEG_QUALITY = 85

app = Flask(__name__)


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
                path = os.path.join(VIDEO_DIR, video_name)
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

    def load_checkpoint(self, checkpoint_path):
        """Load (or reload) a model checkpoint."""
        import torch
        from src.utils.model_utils import load_pretrained_model, EMA
        from src.utils.data_utils import load_config

        with self._lock:
            config = load_config(CONFIG_PATH)
            self.config = config
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
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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
        result.append({
            "base_id": base_id,
            "variants": rec["variants"],
            "waggle_count": len(run_ids),
        })
    return jsonify(result)


@app.route("/api/video/<video_name>/info")
def api_video_info(video_name):
    """Get video file metadata (resolution, fps, frame count)."""
    try:
        info = frame_server.get_video_info(video_name)
        return jsonify(info)
    except FileNotFoundError:
        return jsonify({"error": f"Video not found: {video_name}"}), 404


@app.route("/api/video/<video_name>/annotations")
def api_video_annotations(video_name):
    """Get deduplicated waggle runs for a specific video file."""
    base, *_ = get_base_and_variant(video_name)
    if base not in recording_index:
        return jsonify({"error": "Recording not found"}), 404

    runs = [
        r for r in recording_index[base]["waggle_runs"].values()
        if r["video_name"] == video_name
    ]
    runs.sort(key=lambda r: r["waggle_start"])
    return jsonify(runs)


@app.route("/api/frame/<video_name>/<int:frame_idx>")
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


@app.route("/api/predictions/<video_name>")
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


# ---------------------------------------------------------------------------
# Pipeline Inspector — step-by-step data pipeline visualization
# ---------------------------------------------------------------------------

@app.route("/api/pipeline/<video_name>")
def api_pipeline(video_name):
    """Return one stage of the data pipeline for a specific annotation window.

    Query params:
        stage: 'raw' | 'crop' | 'transform' | 'normalize' | 'sample' | 'target' | 'output'
        row_idx: index into this video's annotation rows (default: 0)
    """
    import torch
    import torchvision.transforms as T
    from torchvision.transforms import ToTensor
    from src.data.video_loader import load_video_frames

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    stage = request.args.get('stage', 'raw')
    row_idx = int(request.args.get('row_idx', 0))
    crop_mode = request.args.get('crop_mode', 'inference')  # 'inference' or 'training'

    video_rows = annotations_df[annotations_df['video_name'] == video_name]
    if row_idx >= len(video_rows):
        return jsonify({"error": f"row_idx {row_idx} >= {len(video_rows)} rows"}), 404
    row = video_rows.iloc[row_idx]

    start_frame = int(row['start_frame'])
    end_frame = int(row['end_frame'])
    gt_x, gt_y = float(row['x1']), float(row['y1'])
    dir_x, dir_y = float(row.get('direction_x', 1.0)), float(row.get('direction_y', 0.0))
    is_waggle = int(row.get('waggle', 0))
    run_id = int(row.get('waggle_run_id', -1))

    crop_h, crop_w = config['data']['height'], config['data']['width']
    window_size = config['data']['window_size']

    def compute_crop_origin(orig_w, orig_h):
        """Compute crop origin. Training mode matches dataset.py with random offset."""
        if crop_mode == 'training':
            # Exact logic from dataset.py L142-154
            rng = np.random.RandomState(seed=row_idx)
            margin_x = int(crop_w * 0.2)
            margin_y = int(crop_h * 0.2)
            max_offset_x = (crop_w // 2) - margin_x
            max_offset_y = (crop_h // 2) - margin_y
            offset_x = rng.randint(-max_offset_x, max_offset_x + 1)
            offset_y = rng.randint(-max_offset_y, max_offset_y + 1)
            x_min_ideal = int(gt_x - crop_w / 2) + offset_x
            y_min_ideal = int(gt_y - crop_h / 2) + offset_y
        else:
            # Inference: centered on GT
            x_min_ideal = int(gt_x - crop_w / 2)
            y_min_ideal = int(gt_y - crop_h / 2)
            offset_x, offset_y = 0, 0
        x_min = max(0, min(orig_w - crop_w, x_min_ideal))
        y_min = max(0, min(orig_h - crop_h, y_min_ideal))
        return x_min, y_min, offset_x, offset_y

    # Common metadata
    meta = {
        'video_name': video_name,
        'row_idx': row_idx,
        'total_rows': len(video_rows),
        'start_frame': start_frame,
        'end_frame': end_frame,
        'gt_position': [gt_x, gt_y],
        'gt_direction': [dir_x, dir_y],
        'is_waggle': is_waggle,
        'run_id': run_id,
        'crop_size': [crop_w, crop_h],
        'window_size': window_size,
    }

    # ── Stage: RAW ──────────────────────────────────────────────────────
    if stage == 'raw':
        frames = load_video_frames(
            os.path.join(VIDEO_DIR, video_name),
            start_frame, end_frame, use_cache=False
        )
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        first = frames[0]
        orig_h, orig_w = first.shape[:2]

        # Compute crop box
        x_min, y_min, offset_x, offset_y = compute_crop_origin(orig_w, orig_h)

        # Encode frames as JPEG
        frame_images = []
        for f in frames:
            f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', f_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_images.append(base64.b64encode(buf).decode('ascii'))

        meta.update({
            'video_resolution': [orig_w, orig_h],
            'n_frames_loaded': len(frames),
            'crop_box': {
                'x_min': x_min, 'y_min': y_min,
                'x_max': x_min + crop_w, 'y_max': y_min + crop_h,
            },
        })
        return jsonify({'stage': 'raw', 'frames': frame_images, 'metadata': meta})

    # ── Stage: CROP ─────────────────────────────────────────────────────
    elif stage == 'crop':
        frames = load_video_frames(
            os.path.join(VIDEO_DIR, video_name),
            start_frame, end_frame, use_cache=False
        )
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        first = frames[0]
        orig_h, orig_w = first.shape[:2]

        # Exact crop logic — use compute_crop_origin for mode support
        x_min, y_min, offset_x, offset_y = compute_crop_origin(orig_w, orig_h)

        # Bee position in crop space
        bee_crop_x = gt_x - x_min
        bee_crop_y = gt_y - y_min

        cropped = [f[y_min:y_min+crop_h, x_min:x_min+crop_w] for f in frames]

        frame_images = []
        for f in cropped:
            f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', f_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_images.append(base64.b64encode(buf).decode('ascii'))

        meta.update({
            'video_resolution': [orig_w, orig_h],
            'n_frames': len(cropped),
            'crop_origin': [x_min, y_min],
            'crop_offset': [offset_x, offset_y],
            'crop_mode': crop_mode,
            'bee_in_crop': [round(bee_crop_x, 1), round(bee_crop_y, 1)],
        })
        return jsonify({'stage': 'crop', 'frames': frame_images, 'metadata': meta})

    # ── Stage: TRANSFORM ────────────────────────────────────────────────
    elif stage == 'transform':
        frames = load_video_frames(
            os.path.join(VIDEO_DIR, video_name),
            start_frame, end_frame, use_cache=False
        )
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        first = frames[0]
        orig_h, orig_w = first.shape[:2]
        x_min, y_min, _, _ = compute_crop_origin(orig_w, orig_h)

        to_tensor = ToTensor()

        # Exact pipeline from dataset.py + main.py:
        # 1. Crop  2. ToTensor  3. transform(ToPILImage→Resize→Grayscale→ToTensor)
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config['augmentations']['width'],
                      config['augmentations']['height'])),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ])

        frame_images = []
        pixel_stats = []
        for f in frames:
            cropped = f[y_min:y_min+crop_h, x_min:x_min+crop_w]
            t = to_tensor(cropped)
            t = transform(t)
            # Stats before normalization
            pixel_stats.append({
                'min': round(float(t.min()), 4),
                'max': round(float(t.max()), 4),
                'mean': round(float(t.mean()), 4),
            })
            frame_images.append(_tensor_to_jpeg(t, greyscale=True))

        meta.update({
            'n_frames': len(frame_images),
            'pixel_stats': pixel_stats,
            'transform_pipeline': [
                'ToTensor()',
                'ToPILImage()',
                f'Resize(({crop_w}, {crop_h}))',
                'Grayscale(num_output_channels=3)',
                'ToTensor()',
            ],
        })
        return jsonify({'stage': 'transform', 'frames': frame_images, 'metadata': meta})

    # ── Stage: NORMALIZE ────────────────────────────────────────────────
    elif stage == 'normalize':
        frames = load_video_frames(
            os.path.join(VIDEO_DIR, video_name),
            start_frame, end_frame, use_cache=False
        )
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        first = frames[0]
        orig_h, orig_w = first.shape[:2]
        x_min, y_min, _, _ = compute_crop_origin(orig_w, orig_h)

        to_tensor = ToTensor()
        mean = config['augmentations']['mean']
        std = config['augmentations']['std']

        # Full eval transform (same as server.py inference)
        test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config['augmentations']['width'],
                      config['augmentations']['height'])),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        frame_images = []  # de-normalized for display
        pixel_stats_pre = []
        pixel_stats_post = []
        for f in frames:
            cropped = f[y_min:y_min+crop_h, x_min:x_min+crop_w]
            t_raw = to_tensor(cropped)

            # Pre-norm via transform without normalize
            transform_no_norm = T.Compose([
                T.ToPILImage(),
                T.Resize((crop_w, crop_h)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
            ])
            t_pre = transform_no_norm(t_raw)
            pixel_stats_pre.append({
                'min': round(float(t_pre.min()), 4),
                'max': round(float(t_pre.max()), 4),
                'mean': round(float(t_pre.mean()), 4),
            })

            t_norm = test_transform(t_raw)
            pixel_stats_post.append({
                'min': round(float(t_norm.min()), 4),
                'max': round(float(t_norm.max()), 4),
                'mean': round(float(t_norm.mean()), 4),
            })

            # De-normalize for display
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            t_denorm = t_norm * std_t + mean_t
            frame_images.append(_tensor_to_jpeg(t_denorm, greyscale=True))

        meta.update({
            'n_frames': len(frame_images),
            'normalization': {'mean': mean, 'std': std},
            'pixel_stats_pre_norm': pixel_stats_pre,
            'pixel_stats_post_norm': pixel_stats_post,
        })
        return jsonify({'stage': 'normalize', 'frames': frame_images, 'metadata': meta})

    # ── Stage: SAMPLE ───────────────────────────────────────────────────
    elif stage == 'sample':
        frames = load_video_frames(
            os.path.join(VIDEO_DIR, video_name),
            start_frame, end_frame, use_cache=False
        )
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        n_loaded = len(frames)

        # Exact sampling logic from dataset.py._sample_frames
        if n_loaded >= window_size:
            sampled_idxs = np.linspace(0, n_loaded - 1, window_size, dtype=int).tolist()
        else:
            # Repeat frames
            reps = list(range(n_loaded)) * (window_size // n_loaded + 1)
            sampled_idxs = reps[:window_size]

        # Return the cropped+transformed sampled frames
        first = frames[0]
        orig_h, orig_w = first.shape[:2]
        x_min, y_min, _, _ = compute_crop_origin(orig_w, orig_h)

        to_tensor = ToTensor()
        test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((crop_w, crop_h)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ])

        frame_images = []
        for idx in sampled_idxs:
            f = frames[idx]
            cropped = f[y_min:y_min+crop_h, x_min:x_min+crop_w]
            t = to_tensor(cropped)
            t = test_transform(t)
            frame_images.append(_tensor_to_jpeg(t, greyscale=True))

        meta.update({
            'n_frames_loaded': n_loaded,
            'n_frames_sampled': window_size,
            'sampled_indices': sampled_idxs,
            'sampling_method': 'linspace' if n_loaded >= window_size else 'repeat_pad',
        })
        return jsonify({'stage': 'sample', 'frames': frame_images, 'metadata': meta})

    # ── Stage: TARGET ───────────────────────────────────────────────────
    elif stage == 'target':
        grid_size = config['model']['grid_size']

        # Compute crop and bee-in-crop position
        video_path = os.path.join(VIDEO_DIR, video_name)
        frames = load_video_frames(video_path, start_frame, end_frame, use_cache=False)
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        first = frames[0]
        orig_h, orig_w = first.shape[:2]
        x_min, y_min, _, _ = compute_crop_origin(orig_w, orig_h)

        # Bee in crop space (same as dataset.py L166-167)
        x = gt_x - x_min
        y = gt_y - y_min

        H, W = crop_h, crop_w
        target_info = {}

        if is_waggle == 1:
            x_norm = x / W
            y_norm = y / H
            ws_in = int(row.get('waggle_start_in_window', -1))
            we_in = int(row.get('waggle_end_in_window', -1))

            duration = end_frame - start_frame
            start_norm = (ws_in - start_frame) / duration if duration > 0 and ws_in != -1 else -1
            end_norm = (we_in - start_frame) / duration if duration > 0 and we_in != -1 else -1

            # Normalize direction
            d_norm = np.sqrt(dir_x**2 + dir_y**2)
            if d_norm > 0:
                dir_x_n, dir_y_n = dir_x / d_norm, dir_y / d_norm
            else:
                dir_x_n, dir_y_n = 1.0, 0.0

            grid_x = max(0, min(grid_size - 1, int(x_norm * grid_size)))
            grid_y = max(0, min(grid_size - 1, int(y_norm * grid_size)))
            cell_x = (x_norm * grid_size) - grid_x
            cell_y = (y_norm * grid_size) - grid_y

            target_info = {
                'objectness': 1.0,
                'grid_cell': [grid_y, grid_x],
                'cell_coords': [round(cell_x, 4), round(cell_y, 4)],
                'position_norm': [round(x_norm, 4), round(y_norm, 4)],
                'direction': [round(dir_x_n, 4), round(dir_y_n, 4)],
                'temporal_norm': [round(start_norm, 4), round(end_norm, 4)],
                'waggle_start_in_window': ws_in,
                'waggle_end_in_window': we_in,
            }
        else:
            target_info = {'objectness': 0.0, 'note': 'Non-waggle window'}

        meta.update({
            'bee_in_crop': [round(x, 1), round(y, 1)],
            'grid_size': grid_size,
            'target': target_info,
        })
        return jsonify({'stage': 'target', 'frames': [], 'metadata': meta})

    # ── Stage: OUTPUT ───────────────────────────────────────────────────
    elif stage == 'output':
        if not prediction_engine.is_loaded:
            return jsonify({'stage': 'output', 'frames': [],
                            'metadata': {**meta, 'error': 'No model loaded'}}), 200

        # Replicate full inference for this single window
        frames = load_video_frames(
            os.path.join(VIDEO_DIR, video_name),
            start_frame, end_frame, use_cache=False
        )
        if not frames:
            return jsonify({"error": "No frames loaded"}), 500

        first = frames[0]
        orig_h, orig_w = first.shape[:2]
        x_min, y_min, _, _ = compute_crop_origin(orig_w, orig_h)

        to_tensor = ToTensor()
        test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config['augmentations']['width'],
                      config['augmentations']['height'])),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=config['augmentations']['mean'],
                        std=config['augmentations']['std']),
        ])

        tensors = [test_transform(to_tensor(f[y_min:y_min+crop_h, x_min:x_min+crop_w]))
                    for f in frames]

        # Sample frames
        n = len(tensors)
        if n >= window_size:
            idxs = np.linspace(0, n - 1, window_size, dtype=int)
            tensors = [tensors[i] for i in idxs]
        else:
            reps = tensors * (window_size // n + 1)
            tensors = reps[:window_size]

        video_tensor = torch.stack(tensors).permute(1, 0, 2, 3)  # (C, T, H, W)
        batch = video_tensor.unsqueeze(0).to(prediction_engine.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = prediction_engine.model(batch)

        # Decode predictions
        from src.utils.eval_utils import yolo_to_img_space
        crop_size = (config['data']['width'], config['data']['height'])
        decoded = yolo_to_img_space(
            output, np.array([start_frame]), np.array([end_frame]),
            window_size=window_size,
            confidence_threshold=0.001,
            original_size=crop_size,
            max_dets=config['eval']['max_dets'],
        )

        # Offset to full-frame coords
        preds = []
        for pred in decoded[0]:
            pred['position'] = [
                pred['position'][0] + x_min,
                pred['position'][1] + y_min,
            ]
            preds.append(pred)

        # Confidence heatmap from output grid
        grid_size = config['model']['grid_size']
        obj_grid = torch.sigmoid(output[0, :, :, 0, 0]).cpu().numpy()
        heatmap_data = obj_grid.tolist()

        # Convert numpy types
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
            'crop_origin': [x_min, y_min],
            'n_predictions': len(preds),
            'predictions': preds,
            'confidence_heatmap': heatmap_data,
            'grid_size': grid_size,
            'checkpoint': prediction_engine.checkpoint_meta,
        })
        return jsonify({'stage': 'output', 'frames': [], 'metadata': meta})

    else:
        return jsonify({"error": f"Unknown stage: {stage}"}), 400


@app.route("/api/pipeline/rows/<video_name>")
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
                normalize=False,  # Don't normalize — we want visible pixels
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


def _tensor_to_jpeg(tensor, greyscale=True):
    """CHW float tensor → base64 JPEG string."""
    import torch
    # Clamp to [0,1]
    t = tensor.clamp(0, 1)
    # CHW → HWC uint8
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if greyscale and arr.shape[2] == 3:
        arr_gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = cv2.cvtColor(arr_gray, cv2.COLOR_GRAY2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('ascii')


@app.route("/api/model_view/<video_name>")
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
            # Undo normalization for display if it was applied
            aug_frame = aug_frames[0]
            result["augmented_image"] = _tensor_to_jpeg(aug_frame, greyscale=True)
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
    args = parser.parse_args()

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading model checkpoint...")
        prediction_engine.load_checkpoint(args.checkpoint)

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
    print(f"{'=' * 60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

