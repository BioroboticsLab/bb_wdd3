"""
Waggle Dance Annotation Viewer — Flask Server

Serves video frames on-demand and annotation data for visual inspection
of the waggle dance training dataset.

Usage:
    python3 viewer/server.py                          # localhost:5050
    python3 viewer/server.py --host 0.0.0.0           # public binding
    python3 viewer/server.py --port 8080              # custom port
    python3 viewer/server.py --host 0.0.0.0 --debug   # dev mode
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
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("  🐝  Waggle Dance Annotation Viewer")
    print(f"{'=' * 60}")
    print(f"  Server:  http://{args.host}:{args.port}")
    if args.host == "localhost":
        print(f"  Tunnel:  ssh -L {args.port}:localhost:{args.port} landgraf@horus")
    else:
        print(f"  Direct:  http://160.45.38.75:{args.port}")
    print(f"{'=' * 60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
