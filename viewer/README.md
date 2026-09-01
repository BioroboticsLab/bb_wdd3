# Waggle Dance Viewer

Browser dashboard for the WaggleNet pipeline. It started as a ground-truth inspector and has grown
into the main tool for working with the model: it overlays predictions on video, lets you edit
annotations, exposes each stage of the data pipeline, and runs the full evaluation with
post-processing parameters you can change and immediately re-score.

## Starting it

```bash
./restart_server.sh
```

The wrapper kills any dangling instance, then starts the server with a checkpoint loaded. Note that
it binds `0.0.0.0`, not localhost — see [Remote access](#remote-access). To run the server directly:

```bash
.venv/bin/python viewer/server.py \
    --checkpoint ckpt/clean_split_v1/best.pth \
    --device cuda:0
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Bind address. `0.0.0.0` exposes it on the network |
| `--port` | `5050` | |
| `--checkpoint` | none | Model `.pth`. **Without it the prediction and evaluation features are disabled** and the server is a ground-truth viewer only |
| `--device` | auto | `cuda:0`, `cpu`, … Auto picks a GPU if one is available |
| `--external-videos` | none | Directory of unannotated video to run inference over |
| `--debug` | off | Flask debug mode |

Configuration is read from `configs/config.yaml` at the repository root — in particular the
`post_process` block, which drives every clustering control in the UI.

## Remote access

The server binds `localhost` by default, so tunnel to it:

```bash
ssh -N -L 5050:localhost:5050 <host>
```

Then open <http://localhost:5050>. Binding `--host 0.0.0.0` and connecting directly also works where
the port is reachable, but exposes the dashboard — including its annotation-editing endpoints — to
anyone who can route to the machine.

> The startup banner prints a tunnel command naming `horus` regardless of where it is running. It is
> hardcoded; ignore it if you are on another machine.

## Main view

Pick a recording from the sidebar. Badges mark whether it is in the **T**rain or **V**alidation
split, and the number is its ground-truth run count. Recordings that exist at several resolution or
frame rate tiers show one tab per tier — the same footage re-encoded, useful for checking that
predictions are resolution-invariant.

The right panel has three tabs:

- **Annotations** — the ground-truth runs, with frame ranges. Click one to jump to it.
- **Inspector** — the data pipeline, stage by stage (see below).
- **Model** — what the network sees for a given bee.

## Predictions

Toggle with the **Predictions** button or `P`. The first request for a video runs inference over it,
which takes a while on long or high-resolution footage; results are cached per video afterwards.

The dropdown selects what is drawn:

| Mode | Shows |
|---|---|
| **Raw (decoded)** | every per-window detection above threshold, straight out of the grid |
| **Averaged** | detections merged across overlapping windows |
| **Clustered** | full post-processing — DBSCAN in normalised space, producing one marker per waggle run |

**Clustered** is the mode that corresponds to what evaluation scores, so it is the honest one to
judge quality by. Raw and Averaged are diagnostic: they show how much of the model's output the
clustering is absorbing.

The confidence slider filters live without re-running inference. The device chip toggles inference
between GPU and CPU.

## Editing ground truth

`N` starts a new run, `Delete` removes the selected one, `Escape` cancels. Edits are **not** written
back to the annotation CSV — they go to a sidecar `data/annotations/gt_overrides.json` recording
additions, modifications and deletions separately, so the original stays intact and any edit can be
reverted. See [`gt_overrides.py`](gt_overrides.py).

`scripts/propagate_gt.py` copies overrides made on a 60 fps recording down to its 30 and 15 fps
variants, so the same correction does not have to be made three times.

## Inspector

Press `I`. Walks one annotation window through the pipeline exactly as training sees it:

`raw` → `crop` → `transform` → `sample` → `target` → `output`

It delegates to the dataset's own debug path rather than reimplementing anything, so what it shows
is what training actually produces. Useful when a model is training badly and you suspect the
inputs. Crop mode can be switched between centred (inference) and random-offset (training), and the
augmentation pipeline can be applied or bypassed.

## Model view

Press `M`. Given a frame and a bee position, shows the crop the network receives, optionally
augmented — the answer to "is the bee even inside the crop?"

## Evaluation dashboard

The **Evaluate** button, or <http://localhost:5050/eval>.

**Run Evaluation** runs the checkpoint over the whole validation split and reports dance-level mAP,
precision, recall, F1, coverage, and mean spatial, temporal and directional error. Results are
cached to disk, so reopening the page does not recompute them; the cache is invalidated when the
checkpoint's epoch changes.

**Re-clustering** is the reason the dashboard exists. Post-processing parameters — spatial
threshold, temporal threshold, confidence, `min_samples`, direction threshold, bee-size multiplier —
have sliders, and changing one re-scores the cached predictions without re-running inference. This
turns a parameter sweep from an overnight job into a few seconds.

**Optimize** runs an Optuna search over those parameters against a metric you choose (F1, mAP,
recall, precision). The best parameters found can be written back to `configs/config.yaml`; the
previous values are backed up to `configs/config.yaml.bak.<timestamp>` first.

> A tuned baseline is kept in `configs/post_process_baseline_tim.json`. Restore it if a search
> produces something worse — the search persists its result over the live config.

Clicking a video in the results table opens a per-video breakdown of matched and missed dances.

## External videos

With `--external-videos <dir>`, the sidebar gains an **External Videos** section for footage that has
no annotations. **Run Inference** processes a whole video and shows the predictions; there is
nothing to score against, but it is the way to see how the model behaves on new material.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Previous / next frame |
| `Shift+←` / `Shift+→` | Jump 10 frames |
| `1`–`9` | Jump to waggle run N |
| `A` | Toggle ground-truth annotations |
| `P` | Toggle predictions |
| `Z` | Cycle zoom: 1× → 2× → 4× → 8× |
| `M` | Toggle model view |
| `I` | Toggle pipeline inspector |
| `N` | New ground-truth annotation |
| `Delete` | Delete selected annotation |
| `Escape` | Cancel annotation editing |
| `R` | Random recording |

Shortcuts are ignored while typing in the search box.

## Annotation states

- **Solid** — the frame is within `[waggle_start, waggle_end]`
- **Ghost** (semi-transparent) — the frame is within the one-second padding around the run, showing
  where the bee will appear or last appeared
- **Hidden** — outside the padding window
