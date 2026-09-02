# WaggleNet

End-to-end video detection and decoding of honey bee waggle dances.

A waggle dance encodes the direction and distance of a food source in the orientation and
duration of a bee's waggle run. WaggleNet takes raw video of a comb and returns the individual
waggle runs it contains — where each one happened, when it started and ended, and which way the
bee was pointing — without hand-tuned motion filters or a separate tracking stage.

The dataset spans three recording setups (Berlin, Nieh, Sharoni) at several resolution and frame
rate tiers, so generalising across camera geometries is a first-class concern rather than an
afterthought.

## How it works

```
video ──► 16-frame windows, 224×224 crops
            │
            ▼
     R(2+1)D-18 backbone (Kinetics-pretrained)
            │
            ├── spatial head  ──► confidence, x, y
            ├── direction head ─► dir_x, dir_y
            └── temporal head ──► t_start, t_end
            │
            ▼
     28×28 grid, one detection per cell
     [conf, x, y, dir_x, dir_y, t_start, t_end]
            │
            ▼
     overlap averaging across sliding windows
            │
            ▼
     cross-window DBSCAN clustering  ──► waggle runs
```

Windows overlap, so the same waggle run is usually seen several times. Two post-processing stages
turn per-window detections into dance-level predictions: `average_overlapping_predictions` merges
detections of the same run seen in different windows, then `cross_window_cluster_predictions`
clusters what remains — in normalised coordinates and seconds, so the result is invariant to
resolution and frame rate. Both live in [`src/utils/dance_eval.py`](src/utils/dance_eval.py).

The three prediction heads can run as plain 3D convolutions, as per-head temporal transformers, or
with the temporal head cross-attending to the spatial head's features — the last being the current
default. The intuition for cross-attention: given *where* the spatial head sees a waggle, *when*
does it start and end? See the docstring in [`src/models/model.py`](src/models/model.py).

## Two levels of evaluation

This distinction runs through the whole codebase and is worth understanding before reading any
number the repo produces.

| | **Window level** | **Dance level** |
|---|---|---|
| Unit scored | one 16-frame window | one complete waggle run |
| Position threshold | pixels (10–30) | fraction of frame width (0.02–0.10) |
| Answers | "does the model fire in the right cell?" | "did we find the dance?" |
| Config | `eval.window_level` | `eval.dance_level` |
| Default | disabled | enabled |

Window-level metrics measure the network in isolation. Dance-level metrics measure the network
*plus* its post-processing, which is what an actual user of the system cares about — and what the
paper reports. Both are computed by [`src/eval/ckpt_eval.py`](src/eval/ckpt_eval.py); either can be
switched off in the config.

## Repository layout

| Path | What lives there |
|---|---|
| `main.py` | Training loop — data loading, augmentation, EMA, AMP, W&B logging |
| `train.sh` | Wrapper around `main.py`: names the run, copies the config for reproducibility, tees the log |
| `src/models/` | Model definitions. **`model.py` is the live one**; the other files are earlier variants kept for reference |
| `src/loss/` | Multi-task loss (objectness, coordinates, direction, temporal) |
| `src/data/` | Dataset, collator, and the comb-cell annotator |
| `src/eval/ckpt_eval.py` | Full evaluation of a checkpoint at both levels |
| `src/utils/dance_eval.py` | Cross-window clustering and dance-level metrics |
| `src/utils/eval_utils.py`, `eval_utils_fast.py` | Window-level metrics; `_fast` is the vectorised equivalent |
| `src/utils/postprocess.py` | Per-window clustering strategies |
| `viewer/` | Browser dashboard — see [viewer/README.md](viewer/README.md) |
| `scripts/` | Diagnostic and one-off analysis scripts — see [scripts/README.md](scripts/README.md) |
| `configs/` | Configuration — see [configs/README.md](configs/README.md) |
| `prepare_training_data.py` | Builds the cleaned annotation CSV from the raw one |

## Setup

Python 3.11 or newer. Install torch from the CUDA index that matches your driver — the wheels on
PyPI are CPU-only, and the failure is silent:

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
```

Verify the GPU is actually visible before starting a run:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Data

Neither videos nor annotations are in the repository. Expected layout:

```
data/
├── videos/                        # one .mp4 per recording × resolution × fps tier
└── annotations/
    └── fps_multires_clean.csv     # the training/eval annotation set
```

Filename prefixes identify the recording setup, and several parts of the codebase rely on this:

| Prefix | Lab | Example |
|---|---|---|
| digit | berlin | `001_2304_1760.mp4` |
| `T` | nieh | `T3-D1-B15-V5-C_960_540.mp4` |
| `C` | sharoni | `C1_..._1224_1024.mp4` |

Files sharing a base name but differing in resolution or carrying a `_dsNNfps` tag are the *same
recording* re-encoded, not separate videos. The train/val split is computed per base recording, so
no recording can appear on both sides of the split at different resolutions.

### Annotation schema

One row per waggle run per window, 18 columns. The ones that matter:

| Column | Meaning |
|---|---|
| `video_name` | file in `data/videos/` |
| `start_frame`, `end_frame` | the 16-frame window this row describes |
| `waggle_start`, `waggle_end` | frame range of the waggle run itself |
| `waggle_start_in_window`, `waggle_end_in_window` | the same, clipped to the window |
| `origin_x`, `origin_y` | run position in frame coordinates |
| `direction_x`, `direction_y` | unit vector of the waggle direction |
| `waggle_run_id` | run identifier, **numbered per video** — ids are not unique across videos |
| `x1`, `y1`, `x2`, `y2` | crop box used for this window |

`prepare_training_data.py` produces `fps_multires_clean.csv` from the raw annotations: it corrects
the +30 frame offset in the Jerusalem (sharoni) annotations, drops upsampled fps variants and
spatially downsampled Jerusalem videos, and generates temporally downsampled versions of the 60 fps
recordings.

## Training

```bash
./train.sh                          # fresh run, timestamped name
./train.sh clean_split_v1           # fresh run, explicit name
./train.sh clean_split_v1 --resume  # resume from ckpt/clean_split_v1/latest.pth
./train.sh clean_split_v1 --resume-best
```

Each run writes to `ckpt/<run_name>/`: `best.pth`, `latest.pth`, `training.log`, and a copy of the
config used. Metrics go to Weights & Biases under the `wagglenet` project.

`main.py` can also be called directly (`--config_path`, `--run_name`, `--resume`,
`--reset_scheduler`), but `train.sh` is preferred because it snapshots the config next to the
checkpoint — without that, a checkpoint cannot be reproduced.

Training uses mixed precision, an EMA copy of the weights (evaluation always uses the EMA weights),
and a linear warmup. `data.data_fraction_divisor` subsamples the *training* set only — set it to 1
for a full run, higher for a quick iteration.

## Evaluation

```bash
.venv/bin/python -m src.eval.ckpt_eval --ckpt_path ckpt/clean_split_v1/best.pth
```

Reports dance mAP, precision/recall/F1, coverage (how many ground-truth dances were found at all),
and mean spatial, temporal and directional error. Which levels run is controlled by
`eval.window_level.enabled` and `eval.dance_level.enabled`.

The post-processing parameters that dance-level results depend on — clustering radius, temporal
threshold, confidence cut, direction threshold, bee-size multiplier — all live under `post_process`
in the config. They are tuned, not arbitrary; see [configs/README.md](configs/README.md) before
changing them.

## The viewer

A browser dashboard for stepping through recordings frame by frame, overlaying ground truth and
model predictions, editing annotations, and running the full evaluation with adjustable
post-processing. This is the fastest way to see what a checkpoint is actually doing.

```bash
./restart_server.sh
```

Full documentation, including remote access and keyboard shortcuts, in
[viewer/README.md](viewer/README.md).

## Bee-size calibration

Dance-level clustering can express its radius in bee lengths rather than pixels, which makes one
tuned value transfer across recording setups with different camera distances. The per-lab bee
length comes from measuring comb cells — a physical constant of roughly known size — in
[`src/data/combcell_annotator/`](src/data/combcell_annotator/). Its output feeds
`BEE_LENGTH_FRACTION` in `src/utils/dance_eval.py`.

## Tests

```bash
.venv/bin/python -m pytest tests/ src/tests/
```

`src/tests/test_eval_utils_fast.py` is the important one: it asserts that the vectorised
window-level metrics agree with the original implementation across edge cases, which is what allows
the fast path to be trusted.
