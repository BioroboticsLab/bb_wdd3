"""
Expand native-resolution annotation windows (data/raw/annotations_windowed.csv)
into one set of rows per resolution tier, matching the tiers
integrate_raw_videos.py actually generates for each video.

Only position columns (x1, y1, x2, y2, origin_x, origin_y) are rescaled, by
the same width/height ratio as the video resize. Frame numbers are
unaffected by resolution (only fps variants change those -- see
prepare_training_data.py's create_downsampled_annotations() for that
step). direction_x/direction_y are a unit vector and angle is already
scale-invariant, so neither is touched.

Usage:
    python scripts/expand_annotation_resolutions.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from integrate_raw_videos import RAW_VIDEO_DIR, RESOLUTION_DIVISORS, ffprobe_wh_fps
from convert_raw_annotations import DEST_PREFIX

INPUT_PATH = "./data/raw/annotations_windowed.csv"
OUTPUT_PATH = "./data/raw/annotations_resolution_expanded.csv"

POSITION_COLS = ["x1", "y1", "x2", "y2", "origin_x", "origin_y"]


def expand_resolutions(df):
    out_rows = []
    missing_videos = set()

    for video_name, group in df.groupby("video_name"):
        # video_name carries DEST_PREFIX (e.g. "C_"), but the raw source file
        # on disk doesn't -- strip it to find the actual file.
        raw_video_name = video_name
        if raw_video_name.startswith(DEST_PREFIX):
            raw_video_name = raw_video_name[len(DEST_PREFIX):]

        src_path = os.path.join(RAW_VIDEO_DIR, raw_video_name)
        if not os.path.exists(src_path):
            missing_videos.add(video_name)
            continue

        w, h, _ = ffprobe_wh_fps(src_path)
        base_name = os.path.splitext(video_name)[0]

        for d in RESOLUTION_DIVISORS:
            tw, th = w // d, h // d
            ratio = tw / w

            tier_df = group.copy()
            tier_df["video_name"] = f"{base_name}_{tw}_{th}.mp4"
            for col in POSITION_COLS:
                tier_df[col] = tier_df[col] * ratio

            out_rows.append(tier_df)

    if missing_videos:
        print(f"WARNING: {len(missing_videos)} video(s) referenced in "
              f"{INPUT_PATH} not found in {RAW_VIDEO_DIR}, skipped:")
        for v in sorted(missing_videos):
            print(f"  - {v}")

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def main():
    df = pd.read_csv(INPUT_PATH)
    expanded = expand_resolutions(df)
    expanded.to_csv(OUTPUT_PATH, index=False)
    print(f"{len(df)} native-resolution rows -> {len(expanded)} rows "
          f"across {len(RESOLUTION_DIVISORS)} resolution tiers")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
