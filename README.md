# WaggleNet

End-to-end video-based action detection of honey bee waggle dances.

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
| `main_temporal_cls.py` | Training loop for the per-frame temporal-classification variant (`src/models/model_temporal_cls.py`, with its own dataset and loss) |
| `train.sh` | Wrapper around `main.py`: names the run, copies the config for reproducibility, tees the log |
| `src/models/` | Model definitions. **`model.py` is the live one**; the other files are earlier variants kept for reference |
| `src/loss/` | Multi-task loss (objectness, coordinates, direction, temporal) |
| `src/data/` | Dataset, collator, and the comb-cell annotator |
| `src/eval/ckpt_eval.py` | Full evaluation of a checkpoint at both levels |
| `src/utils/dance_eval.py` | Cross-window clustering and dance-level metrics |
| `src/utils/eval_utils.py`, `eval_utils_fast.py` | Window-level metrics; `_fast` is the vectorised equivalent |
| `src/utils/postprocess.py` | Per-window clustering strategies |
| `viewer/` | Browser dashboard — see [viewer/README.md](viewer/README.md) |
| `scripts/` | Data ingestion chain (see **Get data**), plus diagnostics and one-off analyses — see [scripts/README.md](scripts/README.md) |
| `configs/` | Configuration — see [configs/README.md](configs/README.md) |
| `prepare_training_data.py` | Builds the cleaned annotation CSV from the raw one |

## Getting started

Requires a conda environment named `wagglenet` (Python 3.11) with CUDA
support.

```bash
conda create -n wagglenet python=3.11
conda activate wagglenet
pip install -r requirements.txt
```

GPU with CUDA 12.1+ is required for training; evaluation/inference also
run on CPU but much more slowly.

## Get data

<details>
<summary><strong>Using the existing dataset</strong></summary>

The training set lives in `data/videos/` (video files) and
`data/annotations/fps_multires_clean.csv` (annotations), already fully
processed and ready to train on. See the Dataset section of the paper
draft for composition details (source, resolution, fps, and annotated
waggle run counts per recording site).

</details>

<details>
<summary><strong>Adding new data</strong></summary>

New waggle dance annotations are collected with **bb_waggle_annotator**, a
local tool that produces one CSV per annotated video segment with columns:

```
video_name, thorax_positions, thorax_frames,
waggle_start_positions, waggle_start_frames, waggle_directions
```

Each `*_positions`/`*_frames`/`*_directions` column holds a Python-literal
list -- one list element per individual waggle run in that segment. A
run's start is marked by a position + frame (`waggle_start_*`), its end by
a separate position + frame using the bee's thorax as the marker
(`thorax_*`), and direction is recorded once per run
(`waggle_directions`). Position/direction are captured once per run, not
separately for start and end -- there is no averaging step anywhere in
this pipeline.

**Staging.** Drop new, not-yet-processed material into `data/raw/`,
mirroring the structure of the final dataset:

```
data/raw/
├── videos/        ← new video segments, native resolution/fps
└── annotations/   ← matching bb_waggle_annotator CSV exports, one per video
```

Don't put raw material directly into `data/videos/`/`data/annotations/` --
those are the final, already-validated dataset the training pipeline reads
from, and every file in them follows the naming convention below.

**Lab prefix.** The pipeline (and training code -- `lab_source()` /
`get_video_category()` in `src/utils/dance_eval.py` /
`src/utils/video_utils.py`) identifies which lab a video came from, and
therefore which bee-size calibration (`BEE_LENGTH_FRACTION`) to use, purely
from its filename prefix: `0` = Berlin, `T` = Nieh/San Diego, `C` =
Sharoni/Jerusalem. `scripts/convert_raw_annotations.py`'s `DEST_PREFIX`
constant prepends this to every video/annotation name it produces --
set it to the prefix matching whichever lab the new data is from before
running the pipeline (it's currently `"C_"`, used for a Sharoni/Jerusalem
batch collected via bb_waggle_annotator). Avoid prefixes already used
with a special meaning elsewhere in the code, e.g. `"C1_"`/`"Capture_"`
specifically, which `prepare_training_data.py` treats as *old* Jerusalem
recordings and restricts to a single resolution tier -- a plain `"C_"`
still reads as Sharoni/Jerusalem for calibration purposes, but isn't
caught by that older, narrower filter, so all 4 resolution tiers are kept.

**The six-step chain, run in order:**

1. **Validate + convert annotations**: `python scripts/convert_raw_annotations.py`
   Checks every CSV in `data/raw/annotations/` independently. A file is
   rejected entirely (not partially) if any row has mismatched list
   lengths, unparseable values, or a run where the end (thorax) frame
   comes before the start frame -- a dance can't end before it begins.
   Rejected files are reported with the exact row/run at fault, for manual
   examination and fixing (re-run once corrected). Clean files are
   expanded into one row per individual waggle run (prefixed per
   `DEST_PREFIX`) and written to `data/raw/annotations_expanded.csv`.

2. **Integrate videos**: `python scripts/integrate_raw_videos.py`
   For every video whose matching CSV passed step 1 (a video is *never*
   integrated if its annotations were rejected, and vice versa -- the two
   scripts share the same check), generates the same resolution/fps tier
   structure already used for the existing Berlin/Nieh/Sharoni footage:
   native resolution plus /2, /4, /8 downsamples, each additionally
   downsampled to 30fps and 15fps where the source fps supports it. Output
   goes straight into `data/videos/`, following the existing naming
   convention: `<name>_<W>_<H>.mp4` per resolution tier,
   `<name>_<W>_<H>_ds<FPS>fps.mp4` for frame-rate variants. Already-generated
   files are skipped (not re-encoded), so re-running after fixing one bad
   CSV only processes what's actually new.

3. **Expand into training windows**: `python scripts/expand_annotation_windows.py`
   Takes `annotations_expanded.csv` and, per video, assigns sequential
   `waggle_run_id`s, linearly interpolates each run's position across its
   frames (start marker to thorax/end marker), and splits each run into
   overlapping fixed 16-frame windows (stride 4) -- reproducing
   `notebooks/AnnotationsPreprocessing.ipynb`'s cells 7/10/12/15/16.
   Writes native-resolution, native-fps windows to
   `data/raw/annotations_windowed.csv`.

4. **Expand across resolution tiers**: `python scripts/expand_annotation_resolutions.py`
   Duplicates each window's rows once per resolution tier generated in
   step 2, scaling position columns (`x1,y1,x2,y2,origin_x,origin_y`) by
   the same width/height ratio as the video resize. Frame numbers and
   direction are untouched (resolution doesn't affect either). Writes to
   `data/raw/annotations_resolution_expanded.csv`.

5. **Expand across fps variants**: `python scripts/expand_annotation_fps.py`
   For each resolution-tier video, adds rows for its `ds30fps`/`ds15fps`
   variants by reusing `prepare_training_data.py`'s
   `create_downsampled_annotations()` -- the same function (and formula,
   verified against the existing dataset) already used for every other
   fps-downsampled video in `fps_multires_clean.csv`. Requires the fps-variant
   video files from step 2 to already exist (reads their real frame counts
   to clamp windows to valid bounds). Writes to
   `data/raw/annotations_fps_expanded.csv`.

6. **Merge into the training set**: `python scripts/merge_new_annotations.py`
   Backs up `data/annotations/fps_multires_clean.csv` to `.csv.bak` (once --
   won't overwrite an existing backup), refuses to merge if any
   `video_name` already exists in it, then appends the new rows. This is
   the only step that touches the real training file; everything before
   it only writes to `data/raw/`.

</details>

> **The annotation CSV is stored in Git LFS.** A plain `git clone` without
> `git lfs install` leaves you with a 134-byte pointer file instead of the
> annotations, and everything downstream fails confusingly. Run:
>
> ```bash
> git lfs install
> git lfs pull
> ```

<details>
<summary><strong>Annotation schema</strong></summary>

`fps_multires_clean.csv` has one row per waggle run per training window, 18
columns. Current size: 918 video files, 513,213 rows. The columns that matter:

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

Files sharing a base name but differing in resolution or carrying a `_dsNNfps`
tag are the *same recording* re-encoded, not separate videos. The train/val
split is computed per base recording, so no recording can appear on both sides
of the split at different resolutions.

`prepare_training_data.py` produced the original cleaned set from the raw
annotations: it corrects the +30 frame offset in the older Jerusalem
annotations, drops upsampled fps variants and spatially downsampled Jerusalem
videos, and generates temporally downsampled versions of the 60 fps recordings.
New material goes through the six-step chain above instead.

</details>

## Training

```bash
python main.py --config_path ./configs/config.yaml --run_name <name>
```

Key flags: `--resume <path/to/checkpoint.pth>` to resume training,
`--reset_scheduler` to reset the LR scheduler on resume (use when epoch
count or dataset size changed). All hyperparameters (model architecture,
loss weights, augmentation, post-processing) are set in
`configs/config.yaml`.

For the per-frame temporal classification variant (alternative temporal
head, see architecture notes) use `main_temporal_cls.py` with the same
arguments instead.

Training logs to Weights & Biases (project set in `config.yaml`'s
`logging.wandb_project`); checkpoints save to `./ckpt/<run_name>/`.

## Evaluation

```bash
python -m src.eval.ckpt_eval --ckpt_path ckpt/clean_split_v1/best.pth
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
python -m pytest tests/ src/tests/
```

`src/tests/test_eval_utils_fast.py` is the important one: it asserts that the vectorised
window-level metrics agree with the original implementation across edge cases, which is what allows
the fast path to be trusted.
