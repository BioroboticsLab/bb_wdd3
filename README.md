## WaggleNet: End-to-End Video Based Action Detection of Honey Bee Waggle Dances

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

**Staging.** Each new data drop gets its own batch subfolder (named by
source/date, e.g. `jerusalem_2026`, `berlin_2024`) under `data/raw/`,
mirroring the structure of the final dataset:

```
data/raw/
├── videos/
│   └── <batch>/   ← new video segments, native resolution/fps
└── annotations/
    └── <batch>/   ← matching bb_waggle_annotator CSV exports, one per video
```

Keeping each drop in its own named subfolder (instead of dumping
everything into one flat `data/raw/videos/`/`data/raw/annotations/`)
means you can always tell where a given raw file came from, and running
the pipeline for one batch can't accidentally pull in another batch's
files that happen to need a different lab prefix. Don't put raw material
directly into `data/videos/`/`data/annotations/` -- those are the final,
already-validated dataset the training pipeline reads from, and every
file in them follows the naming convention below.

Note: data merged before this batch-subfolder convention existed
(everything except the batches listed above) doesn't have its raw-source
staging preserved -- only the already-integrated result in
`data/annotations/fps_multires_clean.csv`. Not urgent to reconstruct,
since that data is already fully integrated and working; worth
documenting if its original source/provenance ever turns up.

**Batch config.** `scripts/convert_raw_annotations.py` is the single
source of truth for which batch is currently being processed -- every
other script in the chain imports `BATCH` (and `RAW_ANNOT_DIR`/
`RAW_VIDEO_DIR`/`DEST_PREFIX`) from it, so a new batch only ever needs
changing in one place:

```python
# scripts/convert_raw_annotations.py
BATCH = "jerusalem_2026"   # <- change this per batch, before running the chain
DEST_PREFIX = "C_"         # <- and this (see "Lab prefix" below)
```

**Lab prefix.** The pipeline (and training code -- `lab_source()` /
`get_video_category()` in `src/utils/dance_eval.py` /
`src/utils/video_utils.py`) identifies which lab a video came from, and
therefore which bee-size calibration (`BEE_LENGTH_FRACTION`) to use, purely
from its filename prefix: `0` = Berlin, `T` = Nieh/San Diego, `C` =
Sharoni/Jerusalem. `DEST_PREFIX` prepends this to every video/annotation
name the pipeline produces -- set it to the prefix matching whichever lab
the new batch is from before running the pipeline. Avoid prefixes already
used with a special meaning elsewhere in the code, e.g. `"C1_"`/`"Capture_"`
specifically, which `prepare_training_data.py` treats as *old* Jerusalem
recordings and restricts to a single resolution tier -- a plain `"C_"`
still reads as Sharoni/Jerusalem for calibration purposes, but isn't
caught by that older, narrower filter, so all 4 resolution tiers are kept.

**The six-step chain, run in order (after setting `BATCH`/`DEST_PREFIX` above):**

1. **Validate + convert annotations**: `python scripts/convert_raw_annotations.py`
   Checks every CSV in `data/raw/annotations/<BATCH>/` independently. A file
   is rejected entirely (not partially) if any row has mismatched list
   lengths, unparseable values, or a run where the end (thorax) frame
   comes before the start frame -- a dance can't end before it begins.
   Rejected files are reported with the exact row/run at fault, for manual
   examination and fixing (re-run once corrected). Clean files are
   expanded into one row per individual waggle run (prefixed per
   `DEST_PREFIX`) and written to `data/raw/<BATCH>_annotations_expanded.csv`.

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
   Takes `<BATCH>_annotations_expanded.csv` and, per video, assigns
   sequential `waggle_run_id`s, linearly interpolates each run's position
   across its frames (start marker to thorax/end marker), and splits each
   run into overlapping fixed 16-frame windows (stride 4) -- reproducing
   `notebooks/AnnotationsPreprocessing.ipynb`'s cells 7/10/12/15/16.
   Writes native-resolution, native-fps windows to
   `data/raw/<BATCH>_annotations_windowed.csv`.

4. **Expand across resolution tiers**: `python scripts/expand_annotation_resolutions.py`
   Duplicates each window's rows once per resolution tier generated in
   step 2, scaling position columns (`x1,y1,x2,y2,origin_x,origin_y`) by
   the same width/height ratio as the video resize. Frame numbers and
   direction are untouched (resolution doesn't affect either). Writes to
   `data/raw/<BATCH>_annotations_resolution_expanded.csv`.

5. **Expand across fps variants**: `python scripts/expand_annotation_fps.py`
   For each resolution-tier video, adds rows for its `ds30fps`/`ds15fps`
   variants by reusing `prepare_training_data.py`'s
   `create_downsampled_annotations()` -- the same function (and formula,
   verified against the existing dataset) already used for every other
   fps-downsampled video in `fps_multires_clean.csv`. Requires the fps-variant
   video files from step 2 to already exist (reads their real frame counts
   to clamp windows to valid bounds). De-duplicates rows afterward --
   downsampling can make several nearby native windows round to the same
   fps-adjusted window, especially for long runs with many overlapping
   windows. Writes to `data/raw/<BATCH>_annotations_fps_expanded.csv`.

6. **Merge into the training set**: `python scripts/merge_new_annotations.py`
   Backs up `data/annotations/fps_multires_clean.csv` to `.csv.bak` (once --
   won't overwrite an existing backup). Rows for `video_name`s already
   present in the clean CSV are skipped (not re-merged, not treated as an
   error) -- only genuinely new rows get appended. This makes the whole
   chain safe to re-run after fixing one more bad CSV in a batch that's
   partially merged already: steps 1-5 reprocess the full batch from
   scratch (not just the fixed file), so this step's skip-if-already-merged
   check is what prevents that from duplicating rows for videos merged in
   an earlier run. This is the only step that touches the real training
   file; everything before it only writes to `data/raw/`.

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