"""
Data preprocessing for WDD3 training run.

1. Fix Jerusalem annotation offset (+30 frames)
2. Filter out upsampled FPS variants
3. Filter out spatially downsampled Jerusalem videos
4. Create temporally downsampled versions of 60fps videos
5. Save cleaned CSV

Usage:
    python prepare_training_data.py
"""

import os
import re
import shutil
import cv2
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
ANNOTATIONS_PATH = os.path.join(DATA_DIR, "annotations", "fps_multires_full_data.csv")
BACKUP_PATH = os.path.join(DATA_DIR, "annotations", "fps_multires_full_data.csv.bak")
CLEAN_PATH = os.path.join(DATA_DIR, "annotations", "fps_multires_clean.csv")

JERUSALEM_OFFSET = 30  # frames to shift forward


def is_upsampled_fps(video_name: str) -> bool:
    """Check if a video is an upsampled FPS variant (frame-duplicated)."""
    # Match any _Nfps suffix (including doubled like _30fps_30fps)
    name = video_name.replace(".mp4", "")
    match = re.match(r"^(.+?)_(\d+)_(\d+)((?:_\d+fps)+)$", name)
    return match is not None and match.group(4) != ""


def is_jerusalem_downsampled(video_name: str) -> bool:
    """Check if a video is a spatially downsampled Jerusalem variant."""
    # Jerusalem videos: C1_* and Capture_*
    # Native resolution is 1224x1024, downsampled are 612x512 and 306x256
    if not (video_name.startswith("C1_") or video_name.startswith("Capture_")):
        return False
    # Check if it's NOT the native resolution
    return "_1224_1024" not in video_name


def get_native_fps(video_path: str) -> float:
    """Get the FPS of a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def create_downsampled_video(src_path: str, dst_path: str, factor: int) -> int:
    """Create a temporally downsampled video by keeping every Nth frame.
    
    Returns the number of frames in the output video.
    """
    cap = cv2.VideoCapture(src_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    dst_fps = src_fps / factor
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, dst_fps, (w, h))
    
    frame_idx = 0
    out_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % factor == 0:
            out.write(frame)
            out_count += 1
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"  Created {dst_path} ({out_count} frames at {dst_fps:.1f}fps from {total} frames at {src_fps:.1f}fps)")
    return out_count


def create_downsampled_annotations(df: pd.DataFrame, src_video: str, dst_video: str,
                                    factor: int, target_window: int = 16,
                                    video_frame_count: int = None) -> pd.DataFrame:
    """Create annotation rows for a downsampled video.
    
    Instead of simply dividing all frame columns by `factor` (which shrinks
    the window from 16 to 16/factor), we:
      1. Convert waggle_start / waggle_end to the new timebase
      2. Recompute start_frame / end_frame to keep a full `target_window`-frame window
      3. Recompute waggle_start_in_window / waggle_end_in_window accordingly
    """
    src_rows = df[df["video_name"] == src_video].copy()
    if len(src_rows) == 0:
        return pd.DataFrame()
    
    dst_rows = src_rows.copy()
    dst_rows["video_name"] = dst_video
    
    # Max frame index in downsampled video (for clamping)
    max_frame = (video_frame_count - 1) if video_frame_count else None
    
    # Convert absolute waggle positions to new timebase
    for col in ["waggle_start", "waggle_end"]:
        if col in dst_rows.columns:
            dst_rows[col] = (dst_rows[col] / factor).astype(int)
    
    # For each row, recompute the window to be target_window frames
    new_starts = []
    new_ends = []
    new_ws_in_win = []
    new_we_in_win = []
    
    for _, row in dst_rows.iterrows():
        # Original window midpoint in source timebase, converted
        orig_start = int(row["start_frame"] / factor)
        orig_end = int(row["end_frame"] / factor)
        mid = (orig_start + orig_end) // 2
        
        # Create target_window-sized window centered on midpoint
        new_start = mid - target_window // 2
        new_end = new_start + target_window
        
        # Clamp to video bounds
        if max_frame is not None:
            if new_end > max_frame + 1:
                new_end = max_frame + 1
                new_start = new_end - target_window
            if new_start < 0:
                new_start = 0
                new_end = new_start + target_window
        if new_start < 0:
            new_start = 0
        
        new_starts.append(new_start)
        new_ends.append(new_end)
        
        # Recompute waggle position within the new window
        ws = row["waggle_start"]
        we = row["waggle_end"]
        new_ws_in_win.append(max(ws, new_start))  # clamp to window
        new_we_in_win.append(min(we, new_end))     # clamp to window
    
    dst_rows["start_frame"] = new_starts
    dst_rows["end_frame"] = new_ends
    dst_rows["waggle_start_in_window"] = new_ws_in_win
    dst_rows["waggle_end_in_window"] = new_we_in_win
    
    # Coordinates stay the same (same resolution, just fewer frames)
    return dst_rows


def main():
    print("=" * 60)
    print("  WDD3 Data Preprocessing")
    print("=" * 60)
    
    # --- Step 0: Backup ---
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(ANNOTATIONS_PATH, BACKUP_PATH)
        print(f"\n✓ Backed up original annotations to {BACKUP_PATH}")
    else:
        print(f"\n✓ Backup already exists at {BACKUP_PATH}")
    
    df = pd.read_csv(ANNOTATIONS_PATH)
    print(f"  Original: {len(df):,} rows, {df['video_name'].nunique()} videos")
    
    # --- Step 1: Fix Jerusalem annotation offset ---
    print(f"\n--- Step 1: Fix Jerusalem annotation offset (+{JERUSALEM_OFFSET} frames) ---")
    jerusalem_mask = df["video_name"].str.startswith("C1_") | df["video_name"].str.startswith("Capture_")
    n_jerusalem = jerusalem_mask.sum()
    
    frame_cols = ["start_frame", "end_frame", "waggle_start", "waggle_end",
                  "waggle_start_in_window", "waggle_end_in_window"]
    for col in frame_cols:
        if col in df.columns:
            df.loc[jerusalem_mask, col] = df.loc[jerusalem_mask, col] + JERUSALEM_OFFSET
    
    print(f"  Shifted {n_jerusalem} Jerusalem rows by +{JERUSALEM_OFFSET} frames")
    
    # --- Step 2: Filter upsampled FPS variants ---
    print("\n--- Step 2: Remove upsampled FPS variants ---")
    upsampled_mask = df["video_name"].apply(is_upsampled_fps)
    n_upsampled = upsampled_mask.sum()
    upsampled_videos = df[upsampled_mask]["video_name"].nunique()
    df = df[~upsampled_mask].reset_index(drop=True)
    print(f"  Removed {n_upsampled:,} rows ({upsampled_videos} upsampled videos)")
    
    # --- Step 3: Filter spatially downsampled Jerusalem ---
    print("\n--- Step 3: Remove spatially downsampled Jerusalem variants ---")
    jerusalem_ds_mask = df["video_name"].apply(is_jerusalem_downsampled)
    n_jerusalem_ds = jerusalem_ds_mask.sum()
    df = df[~jerusalem_ds_mask].reset_index(drop=True)
    print(f"  Removed {n_jerusalem_ds:,} rows (downsampled Jerusalem videos)")
    
    # --- Step 4: Temporal downsampling of 60fps videos ---
    print("\n--- Step 4: Temporal downsampling of native 60fps videos ---")
    
    # Find native 60fps videos in the remaining dataset
    remaining_videos = df["video_name"].unique()
    high_fps_videos = []
    
    for vname in remaining_videos:
        vpath = os.path.join(VIDEO_DIR, vname)
        if not os.path.exists(vpath):
            continue
        fps = get_native_fps(vpath)
        if fps > 50:  # ~60fps
            high_fps_videos.append((vname, fps))
    
    print(f"  Found {len(high_fps_videos)} native 60fps videos")
    
    new_annotation_rows = []
    for vname, fps in high_fps_videos:
        vpath = os.path.join(VIDEO_DIR, vname)
        base_name = vname.replace(".mp4", "")
        
        # Create 30fps version (every 2nd frame)
        dst_30 = f"{base_name}_ds30fps.mp4"
        dst_30_path = os.path.join(VIDEO_DIR, dst_30)
        if not os.path.exists(dst_30_path):
            n_frames_30 = create_downsampled_video(vpath, dst_30_path, factor=2)
        else:
            print(f"  Skipping {dst_30} (already exists)")
            cap = cv2.VideoCapture(dst_30_path)
            n_frames_30 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        new_rows_30 = create_downsampled_annotations(df, vname, dst_30, factor=2,
                                                      video_frame_count=n_frames_30)
        new_annotation_rows.append(new_rows_30)
        
        # Create 15fps version (every 4th frame)
        dst_15 = f"{base_name}_ds15fps.mp4"
        dst_15_path = os.path.join(VIDEO_DIR, dst_15)
        if not os.path.exists(dst_15_path):
            n_frames_15 = create_downsampled_video(vpath, dst_15_path, factor=4)
        else:
            print(f"  Skipping {dst_15} (already exists)")
            cap = cv2.VideoCapture(dst_15_path)
            n_frames_15 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        new_rows_15 = create_downsampled_annotations(df, vname, dst_15, factor=4,
                                                      video_frame_count=n_frames_15)
        new_annotation_rows.append(new_rows_15)
    
    if new_annotation_rows:
        new_df = pd.concat(new_annotation_rows, ignore_index=True)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"  Added {len(new_df):,} annotation rows for downsampled videos")
    
    # --- Step 5: Save ---
    print(f"\n--- Step 5: Save cleaned CSV ---")
    df.to_csv(CLEAN_PATH, index=False)
    print(f"  Saved to {CLEAN_PATH}")
    print(f"  Final: {len(df):,} rows, {df['video_name'].nunique()} videos")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  Original rows:     {pd.read_csv(BACKUP_PATH).shape[0]:,}")
    print(f"  After cleanup:     {len(df):,}")
    print(f"  Videos remaining:  {df['video_name'].nunique()}")
    
    # Show per-source breakdown
    print(f"\n  Per-source breakdown:")
    for vname in sorted(df["video_name"].unique()):
        pass  # just count
    
    berlin = df[df["video_name"].str.match(r"^0\d+")]["video_name"].nunique()
    san_diego = df[df["video_name"].str.startswith("T")]["video_name"].nunique()
    jerusalem = df[df["video_name"].str.match(r"^(C1_|Capture_)")]["video_name"].nunique()
    print(f"    Berlin:     {berlin} videos")
    print(f"    San Diego:  {san_diego} videos")
    print(f"    Jerusalem:  {jerusalem} videos")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
