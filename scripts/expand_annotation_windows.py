"""
Expand per-run annotations (data/raw/annotations_expanded.csv) into fixed
16-frame training windows with interpolated positions and per-video
waggle_run_id, reproducing notebooks/AnnotationsPreprocessing.ipynb's
cells 7 (run id), 10 (position interpolation), 12 (sliding windows) and
15/16 (apply interpolated position per window).

This operates at native resolution/fps only. Resolution tiers and fps
variants of the resulting rows are a separate, later step (see
integrate_raw_videos.py for the video side, and prepare_training_data.py's
create_downsampled_annotations() for how fps-variant annotation rows are
derived from native-fps ones).

Usage:
    python scripts/expand_annotation_windows.py
"""
import numpy as np
import pandas as pd

INPUT_PATH = "./data/raw/annotations_expanded.csv"
OUTPUT_PATH = "./data/raw/annotations_windowed.csv"

WINDOW = 16
STRIDE = 4


def assign_run_ids(df):
    df = df.copy()
    df["waggle_run_id"] = df.groupby("video_name").cumcount() + 1
    return df


def interpolate_positions(df):
    """One row per (video_name, frame, waggle_run_id) with interpolated x,y,
    linearly from the run's start position (x1,y1) to its end position
    (x2,y2) across its frames."""
    rows = []
    for _, r in df.iterrows():
        start_f, end_f = int(r["start_frame"]), int(r["end_frame"])
        n_frames = end_f - start_f + 1
        xs = np.linspace(r["x1"], r["x2"], n_frames)
        ys = np.linspace(r["y1"], r["y2"], n_frames)
        frames = np.arange(start_f, end_f + 1)
        for f, x, y in zip(frames, xs, ys):
            rows.append((r["video_name"], f, int(r["waggle_run_id"]), x, y))
    return pd.DataFrame(rows, columns=["video_name", "frame", "waggle_run_id", "x", "y"])


def expand_windows(df, stride=STRIDE, clamp_start_at_zero=True):
    """16-frame windows overlapping each run. Short runs get one centered
    window; long runs get overlapping sliding windows. Mirrors
    expand_waggle_windows_with_offsets_fixed16 from the notebook."""
    rows = []
    for _, row in df.iterrows():
        waggle_start = int(row["start_frame"])
        waggle_end = int(row["end_frame"])
        dif = waggle_end - waggle_start
        if dif <= 0:
            continue

        mid = dif // 2
        mid_frame = waggle_start + mid
        starts = set()

        if dif < WINDOW:
            pad_left = (WINDOW - dif) // 2
            s = waggle_start - pad_left
            if clamp_start_at_zero:
                s = max(0, s)
            starts.add(s)

        if dif >= WINDOW:
            last_inside_start = waggle_end - WINDOW
            s = waggle_start
            while s <= last_inside_start:
                starts.add(max(0, s) if clamp_start_at_zero else s)
                s += stride

        i = mid_frame - dif
        while i < waggle_start:
            s = max(0, i) if clamp_start_at_zero else i
            if s + WINDOW > waggle_start and s < waggle_end:
                starts.add(s)
            i += stride

        j = waggle_end - mid
        while j < waggle_end:
            s = max(0, j) if clamp_start_at_zero else j
            if s + WINDOW > waggle_start and s < waggle_end:
                starts.add(s)
            j += stride

        for s in sorted(starts):
            e = s + WINDOW
            overlap_start = max(s, waggle_start)
            overlap_end = min(e, waggle_end)
            if overlap_start >= overlap_end:
                continue

            new_row = row.copy()
            new_row["start_frame"] = int(s)
            new_row["end_frame"] = int(e)
            new_row["waggle_start"] = waggle_start
            new_row["waggle_end"] = waggle_end
            new_row["waggle_start_in_window"] = int(overlap_start)
            new_row["waggle_end_in_window"] = int(overlap_end)
            rows.append(new_row)

    return pd.DataFrame(rows).reset_index(drop=True)


def update_xy_with_offsets(expanded_df, offsets_df):
    """Replace each window's x1,y1 with the interpolated position at the
    frame where the run actually becomes visible inside that window
    (waggle_start_in_window), instead of reusing the run's overall start
    position for every window."""
    lookup = {
        (r.video_name, r.frame, r.waggle_run_id): (r.x, r.y)
        for r in offsets_df.itertuples(index=False)
    }
    expanded_df = expanded_df.copy()
    for idx, row in expanded_df.iterrows():
        key = (row["video_name"], row["waggle_start_in_window"], row["waggle_run_id"])
        if key in lookup:
            expanded_df.at[idx, "x1"], expanded_df.at[idx, "y1"] = lookup[key]
    return expanded_df


def main():
    df = pd.read_csv(INPUT_PATH)
    df = assign_run_ids(df)

    offsets_df = interpolate_positions(df)
    windowed_df = expand_windows(df)
    windowed_df = update_xy_with_offsets(windowed_df, offsets_df)

    windowed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"{len(df)} runs -> {len(windowed_df)} windows "
          f"across {df['video_name'].nunique()} videos")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
