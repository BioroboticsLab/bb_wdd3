"""
Convert raw local-tool annotation exports
(data/raw/annotations/<BATCH>/*.csv) into the per-run "expanded" format
the rest of the pipeline expects, mirroring
notebooks/AnnotationsPreprocessing.ipynb's cell 2 conversion.

This is the single source of truth for BATCH/DEST_PREFIX and the raw
staging paths -- every other script in the pipeline imports these from
here, so a batch only ever needs to be set in one place.

Each new data drop gets its own batch subfolder under data/raw/videos/
and data/raw/annotations/ (e.g. "jerusalem_2026", "berlin_2024"), so raw
material from different sources/labs never gets mixed together in one
conversion run, and each batch's provenance stays traceable by folder
name instead of being lost once everything lands in data/raw/ flat.

Input schema (one row per annotation session, list-valued columns -- one
list element per individual waggle run within that session):
    video_name, thorax_positions, thorax_frames,
    waggle_start_positions, waggle_start_frames, waggle_directions

Output schema (one row per individual waggle run, video_name stripped to a
bare filename):
    video_name, start_frame, end_frame, x1, y1, x2, y2, angle, waggle,
    origin_x, origin_y, direction_x, direction_y

Per-file validation, not a global gate: each CSV (and its corresponding
video) is checked independently. A clean file gets converted and integrated
normally. A file with any misalignment -- anywhere in any row -- is
skipped entirely, along with its matching video, and reported clearly so
it can be fixed by hand and picked up on a later run. Good files are never
held back by problems in unrelated files.

A row/run is invalid if:
  - one or more of the 5 list columns fails to parse
  - the 5 lists for a row don't all have the same length
  - for a paired run (same list index across lists), end_frame (thorax)
    comes before start_frame (waggle start) -- a dance can't end before
    it begins
"""
import ast
import math
import os
import subprocess
import pandas as pd

# The only things that should change between batches -- all five settings
# below, together, in this one place. (integrate_raw_videos.py imports
# VIDEO_EXT/ROTATE_DEGREES/RESOLUTION_DIVISORS from here rather than
# defining its own copies, so a batch switch only ever means editing this
# one block instead of hunting through two files.)
#
# BATCH: subfolder name under data/raw/videos/ and data/raw/annotations/
#   holding the new drop.
# DEST_PREFIX: pick a prefix per batch that (a) maps to the right lab in
#   lab_source()/get_video_category() (src/utils/dance_eval.py,
#   src/utils/video_utils.py: '0'->berlin, 'T'->nieh, 'C'->sharoni, else
#   "other") for correct BEE_LENGTH_FRACTION calibration, and (b) doesn't
#   collide with an existing naming convention. Unlike the existing 001-084
#   videos (bare sequential numbers, no traceability to source), new
#   batches keep the lab + batch tag + full original filename --
#   self-describing from the name alone, and traceable back to the exact
#   source file on trove. Existing video names are NOT touched -- this
#   convention only applies to new data going forward.
# VIDEO_EXT: the raw source video container extension for this batch
#   (varies -- Berlin/Jerusalem drops were .mp4, Bristol is .mov).
# ROTATE_DEGREES / RESOLUTION_DIVISORS: see integrate_raw_videos.py's
#   module docstring for how these are derived per batch.
BATCH = "ucsd_2026"
DEST_PREFIX = ""
VIDEO_EXT = ".MP4"
ROTATE_DEGREES = 0
RESOLUTION_DIVISORS = [1, 2]

RAW_ANNOT_DIR = f"./data/raw/annotations/{BATCH}"
RAW_VIDEO_DIR = f"./data/raw/videos/{BATCH}"
OUTPUT_PATH = f"./data/raw/{BATCH}_annotations_expanded.csv"

REQUIRED_COLS = [
    "video_name", "thorax_positions", "thorax_frames",
    "waggle_start_positions", "waggle_start_frames", "waggle_directions",
]


def safe_literal_eval(val):
    if pd.isna(val):
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None


def check_file(fname):
    """Check one CSV in isolation. Returns (is_clean, issues, rows) where
    rows is the list of (video_name, parsed_lists) for every non-empty row
    -- only meaningful/used when is_clean is True."""
    path = os.path.join(RAW_ANNOT_DIR, fname)
    df = pd.read_csv(path)
    issues = []
    rows = []

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return False, [f"missing columns {missing_cols}"], []

    for idx, row in df.iterrows():
        video_name_raw = row["video_name"]
        if pd.isna(video_name_raw) or str(video_name_raw).strip() == "":
            continue

        keys = ["thorax_positions", "thorax_frames", "waggle_start_positions",
                "waggle_start_frames", "waggle_directions"]
        parsed = {k: safe_literal_eval(row[k]) for k in keys}
        values = list(parsed.values())

        if any(v is None for v in values) or any(not isinstance(v, list) for v in values):
            issues.append(f"row {idx}: one or more columns failed to parse")
            continue

        lengths = {len(v) for v in values}
        if len(lengths) > 1:
            issues.append(f"row {idx}: length mismatch {[len(v) for v in values]}")
            continue

        wf, tf = parsed["waggle_start_frames"], parsed["thorax_frames"]
        for i, (w, t) in enumerate(zip(wf, tf)):
            if t < w:
                issues.append(f"row {idx} index {i}: end({t}) < start({w})")

        rows.append((str(video_name_raw).strip(), parsed))

    # Per-file gate: the whole file is used only if zero issues were found
    # anywhere in it. A single bad row disqualifies every row in the file,
    # by design -- not a per-row filter.
    is_clean = len(issues) == 0
    return is_clean, issues, (rows if is_clean else [])


def ffprobe_wh(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", path]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h = out.split(",")
    return int(w), int(h)


def rotate_position(x, y, w_orig, h_orig, degrees):
    """Map a position from the raw (pre-rotation) frame into the
    post-rotation frame -- must match the ffmpeg transpose
    integrate_raw_videos.py applies to the actual video pixels, or the
    annotation ends up pointing at the wrong place in the rotated video."""
    if degrees == 90:      # clockwise (ffmpeg transpose=1)
        return h_orig - y, x
    if degrees == 270:     # counter-clockwise (ffmpeg transpose=2)
        return y, w_orig - x
    return x, y


def rotate_direction(dx, dy, degrees):
    """Same rotation applied to a direction vector -- no translation term,
    just the linear/rotational part."""
    if degrees == 90:
        return -dy, dx
    if degrees == 270:
        return dy, -dx
    return dx, dy


def convert_rows(rows, raw_video_filename=None):
    """Expand a clean file's rows into individual run dicts.

    raw_video_filename: the source video's filename under RAW_VIDEO_DIR
    (as returned by guess_video_filename(), NOT the CSV's own possibly-messy
    video_name column) -- only needed when ROTATE_DEGREES requires mapping
    positions/directions from the raw (pre-rotation) frame into the
    post-rotation frame this batch's videos were actually integrated at.
    """
    out = []

    w_orig = h_orig = None
    if ROTATE_DEGREES and raw_video_filename:
        src_path = os.path.join(RAW_VIDEO_DIR, raw_video_filename)
        w_orig, h_orig = ffprobe_wh(src_path)

    for video_name_raw, parsed in rows:
        video_name = DEST_PREFIX + os.path.basename(video_name_raw)
        for t_pos, t_frame, w_pos, w_frame, w_dir in zip(
            parsed["thorax_positions"], parsed["thorax_frames"],
            parsed["waggle_start_positions"], parsed["waggle_start_frames"],
            parsed["waggle_directions"],
        ):
            wx, wy = w_pos[0], w_pos[1]
            tx, ty = t_pos[0], t_pos[1]
            dx, dy = w_dir[0], w_dir[1]
            if ROTATE_DEGREES:
                wx, wy = rotate_position(wx, wy, w_orig, h_orig, ROTATE_DEGREES)
                tx, ty = rotate_position(tx, ty, w_orig, h_orig, ROTATE_DEGREES)
                dx, dy = rotate_direction(dx, dy, ROTATE_DEGREES)

            out.append({
                "video_name": video_name,
                "start_frame": int(w_frame),
                "end_frame": int(t_frame),
                "x1": float(wx), "y1": float(wy),
                "x2": float(tx), "y2": float(ty),
                "angle": math.atan2(dx, dy),
                "waggle": 1,
                "origin_x": float(wx), "origin_y": float(wy),
                "direction_x": float(dx), "direction_y": float(dy),
            })
    return out


def guess_video_filename(csv_fname):
    """CSV is '<base>_waggle_annotations.csv' -> video is '<base>VIDEO_EXT'."""
    base = csv_fname.replace("_waggle_annotations.csv", "")
    return base + VIDEO_EXT


def main():
    csv_files = sorted(f for f in os.listdir(RAW_ANNOT_DIR) if f.endswith(".csv"))
    print(f"Checking {len(csv_files)} annotation CSVs (per-file, independent)...")
    print()

    expanded_rows = []
    integrated_files = []
    skipped_files = []

    for fname in csv_files:
        is_clean, issues, rows = check_file(fname)
        video_name = guess_video_filename(fname)

        # ROTATE_DEGREES != 0 means convert_rows() needs to ffprobe the raw
        # video's dimensions -- a CSV with no matching video (orphan) can't
        # be rotation-corrected, so treat it the same as any other skip
        # rather than letting ffprobe crash the whole run on it.
        if is_clean and ROTATE_DEGREES and not os.path.exists(os.path.join(RAW_VIDEO_DIR, video_name)):
            is_clean = False
            issues = [f"no matching video at {RAW_VIDEO_DIR}/{video_name} -- "
                      f"can't rotation-correct without probing its dimensions"]

        if is_clean:
            run_dicts = convert_rows(rows, raw_video_filename=video_name)
            expanded_rows.extend(run_dicts)
            integrated_files.append((fname, video_name, len(run_dicts)))
        else:
            skipped_files.append((fname, video_name, issues))

    out_df = pd.DataFrame(expanded_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print(f"Integrated: {len(integrated_files)} of {len(csv_files)} files "
          f"({len(expanded_rows)} total runs) -> {OUTPUT_PATH}")
    print("=" * 70)
    for fname, video_name, n_runs in integrated_files:
        print(f"  OK   {fname}  ({n_runs} runs, video: {video_name})")

    print()
    print("=" * 70)
    print(f"NOT converted/integrated: {len(skipped_files)} file(s) -- "
          f"needs manual examination and fix")
    print("=" * 70)
    for fname, video_name, issues in skipped_files:
        print(f"  SKIPPED  {fname}")
        print(f"           (and its video, {video_name}, was NOT integrated)")
        for issue in issues:
            print(f"           - {issue}")
    if not skipped_files:
        print("  (none)")


if __name__ == "__main__":
    main()
