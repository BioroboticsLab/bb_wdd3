# Configuration

Everything is driven by [`config.yaml`](config.yaml). Training, evaluation and the viewer all read
the same file, which is why changing a post-processing value in the dashboard changes what
`ckpt_eval.py` will report afterwards.

`train.sh` copies the config into `ckpt/<run_name>/config.yaml` when a run starts. That snapshot is
the only record of how a checkpoint was produced — the live file will have moved on.

## `post_process` — the parameters that decide dance-level results

These control how per-window detections are clustered into waggle runs. Dance-level metrics are
sensitive to them, and they are **tuned values, not defaults** — a search produced them. Treat a
change here as an experiment, not a tweak.

| Key | Current | Meaning |
|---|---|---|
| `spatial_threshold` | 49.4 | DBSCAN radius in pixels at a reference width of 1000 px. **Ignored when `cluster_radius_mm` is set**, except as the fallback for a lab with no mm calibration |
| `temporal_threshold_ms` | 130.3 | How far apart in time two detections can be and still belong to the same run. In milliseconds, so frame rate does not matter |
| `confidence_threshold` | 0.2175 | Detections below this are dropped before clustering |
| `min_samples` | 2 | DBSCAN `min_samples` |
| `direction_threshold_deg` | 45 | Angular separation beyond which detections will not cluster together. **`0` disables the direction dimension entirely** |
| `cluster_radius_mm` | 5.0 | Clustering radius as a physical distance — see below. `null` restores the plain `spatial_threshold` behaviour |
| `mode` | `mean` | How a cluster is summarised into one prediction: `mean` or `median` |
| `strategy` | `cluster_consolidate` | Window-level strategy. Also accepts `nms`, `max`, `weighted`, `cluster_max`, `line_nms`; only `cluster_consolidate` is used in practice |
| `temporal_threshold` | 8 | Legacy frame-based threshold, used only by the window-level path. `temporal_threshold_ms` is the one that matters |

### `cluster_radius_mm`

"Detections within this many millimetres are the same dance." Converted per video through
`FRAME_WIDTH_MM` in `src/utils/dance_eval.py`, so one configured number means the same physical
distance whatever the camera distance — which is the point: the model is meant to work across
setups without per-lab hyperparameter fitting.

`FRAME_WIDTH_MM` is measured, not tuned. The comb-cell annotator in
`src/data/combcell_annotator/` gives a mean cell diameter in pixels; a worker comb cell is ~5 mm,
so `frame_width_mm = frame_width_px / (cell_px / 5)`. Currently berlin 446.5, nieh 129.0,
sharoni 303.3 — field-of-view widths you can sanity-check against the real hives.

The default 5.0 is exactly one comb cell, and reproduces the previous `bee_size_multiplier: 0.5`
operating point bit for bit (that parameterisation was `radius_mm = 10 x N`, so the rename is a
pure reparameterisation).

### Restoring the baseline

[`post_process_baseline_tim.json`](post_process_baseline_tim.json) holds a known-good tuned
parameter set, captured before later searches overwrote the live config. The dashboard's optimiser
writes its best result straight into `config.yaml` (backing the old one up to
`config.yaml.bak.<timestamp>`), so this file is what you compare against when a search makes things
worse.

## `eval` — two levels

| Key | Meaning |
|---|---|
| `batch_size` | Evaluation batch size |
| `confidence_threshold` | Decode threshold, deliberately near zero so metrics see the full precision/recall curve |
| `max_dets` | Top-N detections kept per window, ranked by confidence — the equivalent of COCO's `maxDets`. **Feeds the dance-level pipeline too**, so changing it moves dance-level numbers |
| `metric` | Which validation metric selects `best.pth` |
| `map_freq` | Compute mAP every N epochs during training. The full sweep is expensive, so this is not 1 |

`window_level` scores individual 16-frame windows; its `pos_thresholds` are in pixels.
`dance_level` scores complete waggle runs after clustering; its `pos_thresholds_mm` are millimetres.
Each block has an `enabled` flag — window-level is currently off, dance-level on.

Both take `iou_thresholds` as a `[min, max]` range (temporal IoU, stepped internally) and
`angular_thresholds` as a list of degrees. A prediction matches a ground-truth dance only if it
passes position, temporal IoU **and** angle simultaneously.

> Dance-level `pos_thresholds_mm` are millimetres, converted per video the same way as the
> clustering radius. Previously they were fractions of frame width, which meant one configured
> threshold was ~20 mm for berlin but ~6 mm for nieh — a 3.4x difference in strictness coming
> purely from camera distance. Per-lab comparisons made before that change are reading that
> artefact as well as model quality.

## `model`

| Key | Meaning |
|---|---|
| `pretrained` | Load Kinetics-400 weights into the R(2+1)D-18 backbone |
| `self_attention` / `cross_attention` | Head architecture. Both `false` gives 3D-conv heads; `self_attention` gives per-head temporal transformers; `cross_attention` has the temporal head attend to spatial features. **Setting both raises `ValueError`** |
| `transformer_heads`, `transformer_layers` | Only used when an attention mode is on |
| `dropout` | Dropout in the heads |
| `grid_size` | Detection grid, 28×28 over the 224×224 input |
| `max_detections_per_cell` | Detections per grid cell |
| `n_classes` | 1 — waggle or not |

## `loss`

Multi-task weights: `lambda_obj`, `lambda_coord`, `lambda_direction`, `lambda_temporal` and
`lambda_noobj`. The last is far smaller than the rest because the grid is overwhelmingly empty and
an unweighted no-object term drowns out everything else.

`use_varifocal` switches the objectness term to varifocal loss (with `varifocal_gamma`), intended to
address overconfident predictions. `quality_decay` controls how target quality falls off with
distance from the true position.

## `data` and `augmentations`

`window_size` (16) frames per sample at `width`×`height` (224). `train_ratio` splits per base
recording, never per file, so the same footage at two resolutions cannot straddle the split.

`data_fraction_divisor` subsamples the **training** set only — 1 uses everything, 4 uses a quarter.
Handy for fast iteration, easy to leave set by accident.

Augmentation probabilities are all `prob_*` keys with an accompanying range. `mean` and `std` are
computed from the training set; the three channels are identical because the pipeline converts to
greyscale and repeats the channel.

## `train`, `system`, `logging`

Standard: `batch_size`, `epochs`, `lr`, `weight_decay`, `warmup_ratio`, `betas`, `use_amp`,
`ema_decay`, `val_freq`, `num_workers`, `seed`, `device`. Evaluation always uses the EMA weights.
`wandb_project` names the Weights & Biases project.

## Unused keys

Present in the file but read by nothing. Left in place rather than removed, but do not expect
changing them to do anything:

- `logging.checkpoint_dir` — checkpoints go to `ckpt/<run_name>/`, set by `train.sh`
- `data.subset_size`
