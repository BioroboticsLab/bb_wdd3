"""
Convert raw local-tool annotation exports (data/raw/annotations/*.csv) into
the per-run "expanded" format the rest of the pipeline expects, mirroring
notebooks/AnnotationsPreprocessing.ipynb's cell 2 conversion.

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
import pandas as pd

RAW_ANNOT_DIR = "./data/raw/annotations"
RAW_VIDEO_DIR = "./data/raw/videos"
OUTPUT_PATH = "./data/raw/annotations_expanded.csv"

# This data is from the Sharoni/Jerusalem lab. Prefixing with "C_" (not the
# existing "C1_"/"Capture_" used by earlier Jerusalem recordings) makes it
# picked up as "sharoni" by lab_source()/get_video_category() (anything
# starting with 'C'), while deliberately NOT matching prepare_training_data
# .py's is_jerusalem_downsampled() check (which only matches "C1_"/"Capture_"
# exactly) -- so, unlike that older data, all 4 resolution tiers of this
# batch are kept rather than filtered down to just one.
DEST_PREFIX = "C_"

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


def convert_rows(rows):
    """Expand a clean file's rows into individual run dicts."""
    out = []
    for video_name_raw, parsed in rows:
        video_name = DEST_PREFIX + os.path.basename(video_name_raw)
        for t_pos, t_frame, w_pos, w_frame, w_dir in zip(
            parsed["thorax_positions"], parsed["thorax_frames"],
            parsed["waggle_start_positions"], parsed["waggle_start_frames"],
            parsed["waggle_directions"],
        ):
            out.append({
                "video_name": video_name,
                "start_frame": int(w_frame),
                "end_frame": int(t_frame),
                "x1": float(w_pos[0]), "y1": float(w_pos[1]),
                "x2": float(t_pos[0]), "y2": float(t_pos[1]),
                "angle": math.atan2(w_dir[0], w_dir[1]),
                "waggle": 1,
                "origin_x": float(w_pos[0]), "origin_y": float(w_pos[1]),
                "direction_x": float(w_dir[0]), "direction_y": float(w_dir[1]),
            })
    return out


def guess_video_filename(csv_fname):
    """CSV is '<base>_waggle_annotations.csv' -> video is '<base>.mp4'."""
    base = csv_fname.replace("_waggle_annotations.csv", "")
    return base + ".mp4"


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

        if is_clean:
            run_dicts = convert_rows(rows)
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
