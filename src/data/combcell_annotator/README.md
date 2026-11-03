# Comb Cell Annotator

Measures how large a comb cell is in pixels, for each recording setup, by having a human click a few
cells. A comb cell is close to a physical constant, so once its pixel size is known for a video, the
size of anything else in that video — a bee — can be estimated from it.

This is what lets clustering express its radius in bee lengths instead of pixels, and therefore
transfer across cameras at different distances from the comb.

## Where the output goes

The measurement ends up in **`BEE_LENGTH_FRACTION`** in `src/utils/dance_eval.py`, as bee length
divided by frame width, per lab. `post_process.bee_size_multiplier` in the main config multiplies
it. See [configs/README.md](../../../configs/README.md).

Copying the numbers across is manual — nothing reads `output/` at runtime.

## What it does

1. Groups the videos in the data directory by source recording. Files sharing a base name at
   different resolutions are the same footage re-encoded, so they count as one recording.
2. Samples recordings per lab and extracts frames from the **highest-resolution copy** of each.
3. You draw shapes over comb cells.
4. Averages the diameters and **projects the measurement to every sibling resolution** by exact
   ratio, rather than annotating each tier separately.
5. Converts diameter to bee dimensions using `BEE_WIDTH_RATIO = 1.0` and `BEE_LENGTH_RATIO = 2.0` —
   a bee is roughly one cell wide and two cells long.

## Running it

```bash
python annotator.py
```

A window opens per frame. Draw the cells, press `D` for the next frame, `Q` to stop early. Results
print to the terminal and are written to `output/`.

## Controls

| Key / Mouse | Action |
|---|---|
| Left click + drag | Draw a circle over a cell |
| Click (hexagon mode) | Place a vertex — closes automatically after six |
| Right click + drag | Pan when zoomed in |
| Arrow keys | Pan when zoomed in |
| `+` / `-` | Zoom in / out |
| `B` | Reset zoom |
| `U` | Undo last circle, or last vertex |
| `D` | Done with this frame, continue |
| `S` | Skip this frame — discards it and asks for another from the same video. For frames with too few clear cells |
| `Q` | Quit early, keeping what was annotated so far |

## Configuration

```yaml
data_path: ../../../data   # relative to this directory
subfolders:
  - name: videos           # subdirectory of data_path holding the videos
n_videos: 1                # source recordings sampled per lab
n_frames: 1                # frames extracted per recording
shape: circle              # circle or hexagon
save_csv: true             # write CSVs to output/ when finished
```

`n_videos: 1` and `n_frames: 1` are deliberate, not placeholders. Camera position and distance are
fixed within a lab, so a second recording or a second frame re-measures the same constant. Annotate
several cells within the one frame instead — that is where the useful averaging happens.

> Earlier versions took a `rotation:` key per subfolder. It no longer has any effect: the videos are
> preprocessed training clips assumed to be correctly oriented, and rotation is fixed at 0 in the
> code.

## Output

| File | Contents |
|---|---|
| `output/annotations_<shape>.csv` | One row per annotated cell: lab, video, resolution, frame index, centre, radius, diameter |
| `output/projected_summary.csv` | Per lab **and resolution tier**: average diameter, derived bee width and length, number of source recordings, and a `measured` flag |

The `measured` column matters when reading the summary: `True` marks the tier that was actually
annotated, `False` marks tiers whose values were computed by scaling. Only one row per lab is a real
measurement.

Both files are overwritten on each run — there is no timestamp in the filename.

## Verifying the projection

```bash
python verify_projection.py [output/annotations_circle.csv]
```

Takes each click made on the high-resolution copy, scales it by the resolution ratio, draws it on
the matching frame of a downsampled sibling, and saves a side-by-side image. Lets you confirm the
projected annotation still lands on the same cell at the right size. With no argument it uses the
most recent `annotations_*.csv` in `output/`.

## Which lab is which

See [`docs.txt`](docs.txt) for how videos are mapped to labs by filename prefix, and how confident
that mapping is.
