"""
Create OOD (out-of-distribution) framerate test dataset from Tim's 100fps bee videos.

The source videos in /mnt/horus/bee_videos_tim/ were recorded at 100fps but
the AVI container metadata says 25fps (4× slow-motion playback). This script:

1. Reads every .avi file from the source directory
2. For each target framerate, creates a downsampled .mp4 by frame-skipping
3. Writes correct FPS metadata to outputs (fixing the 25fps playback lie)
4. Generates a manifest CSV with metadata for all produced variants

Output structure:
    /mnt/horus/bee_videos_tim_ood/
    ├── manifest.csv
    ├── 100fps/   ← re-muxed originals with corrected FPS metadata
    ├── 50fps/    ← every 2nd frame
    ├── 25fps/    ← every 4th frame
    ├── 20fps/    ← every 5th frame
    └── 10fps/    ← every 10th frame

Usage:
    python create_ood_test_set.py                          # full run
    python create_ood_test_set.py --dry-run                # preview only
    python create_ood_test_set.py --limit 5                # process first 5 videos
    python create_ood_test_set.py --targets 50 25 10       # custom target framerates
"""

import argparse
import csv
import os
import sys
import time

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = "/mnt/horus/bee_videos_tim/"
OUTPUT_DIR = "/mnt/horus/bee_videos_tim_ood/"
TRUE_SOURCE_FPS = 100  # actual capture rate (container lies and says 25fps)

# Default target framerates — all clean integer divisors of 100
DEFAULT_TARGETS = [100, 50, 25, 20, 10]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_source_videos(source_dir: str) -> list[str]:
    """Return sorted list of .avi filenames (excluding .avi.avi artifacts)."""
    files = []
    for f in sorted(os.listdir(source_dir)):
        if f.endswith(".avi") and not f.endswith(".avi.avi"):
            files.append(f)
    return files


def downsample_video(src_path: str, dst_path: str,
                     factor: int, target_fps: float) -> dict:
    """Create a temporally downsampled video by keeping every Nth frame.

    Args:
        src_path: path to source video
        dst_path: path to write output video
        factor:   keep every factor-th frame (1 = keep all)
        target_fps: FPS to write into the output container metadata

    Returns:
        dict with keys: src_frames, dst_frames, duration_s
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, target_fps, (w, h))

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

    duration_s = out_count / target_fps if target_fps > 0 else 0

    return {
        "src_frames": src_total,
        "dst_frames": out_count,
        "duration_s": round(duration_s, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create OOD framerate test dataset from 100fps bee videos.")
    parser.add_argument("--source-dir", default=SOURCE_DIR,
                        help=f"Source video directory (default: {SOURCE_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--targets", nargs="+", type=int, default=DEFAULT_TARGETS,
                        help=f"Target framerates (default: {DEFAULT_TARGETS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N videos (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--source-fps", type=int, default=TRUE_SOURCE_FPS,
                        help=f"True capture FPS of source videos (default: {TRUE_SOURCE_FPS})")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    true_fps = args.source_fps
    targets = sorted(args.targets, reverse=True)

    # Validate targets
    for t in targets:
        if true_fps % t != 0:
            print(f"WARNING: {true_fps}fps / {t}fps = {true_fps/t:.2f} "
                  f"(not integer — will use floor, some temporal drift)")
        if t > true_fps:
            print(f"ERROR: target {t}fps > source {true_fps}fps — skipping")
            targets.remove(t)

    # Discover source videos
    videos = get_source_videos(source_dir)
    if args.limit:
        videos = videos[:args.limit]

    print("=" * 65)
    print("  OOD Framerate Test Dataset Generator")
    print("=" * 65)
    print(f"  Source:       {source_dir} ({len(get_source_videos(source_dir))} videos)")
    print(f"  True FPS:     {true_fps}")
    print(f"  Output:       {output_dir}")
    print(f"  Targets:      {targets}")
    print(f"  Processing:   {len(videos)} videos")
    print(f"  Dry run:      {args.dry_run}")
    print("=" * 65)

    if args.dry_run:
        for t in targets:
            factor = true_fps // t
            print(f"\n  {t}fps (factor={factor}):")
            for v in videos[:5]:
                base = os.path.splitext(v)[0]
                print(f"    {v} → {base}.mp4")
            if len(videos) > 5:
                print(f"    ... and {len(videos) - 5} more")
        print("\nDry run complete. Re-run without --dry-run to create files.")
        return

    # Create output directories
    for t in targets:
        subdir = os.path.join(output_dir, f"{t}fps")
        os.makedirs(subdir, exist_ok=True)

    # Manifest data
    manifest_rows = []
    total_ops = len(videos) * len(targets)
    completed = 0
    skipped = 0
    t_start = time.time()

    for vid_idx, video_file in enumerate(videos):
        src_path = os.path.join(source_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        # Strip ".raw" suffix if present for cleaner output names
        if base_name.endswith(".raw"):
            base_name = base_name[:-4]
        out_name = f"{base_name}.mp4"

        print(f"\n[{vid_idx + 1}/{len(videos)}] {video_file}")

        for target_fps in targets:
            factor = true_fps // target_fps
            subdir = os.path.join(output_dir, f"{target_fps}fps")
            dst_path = os.path.join(subdir, out_name)

            # Resume-safe: skip if output already exists
            if os.path.exists(dst_path):
                # Read existing file metadata for manifest
                cap = cv2.VideoCapture(dst_path)
                dst_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                cap = cv2.VideoCapture(src_path)
                src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                manifest_rows.append({
                    "source_file": video_file,
                    "output_file": out_name,
                    "target_fps": target_fps,
                    "factor": factor,
                    "src_frames": src_frames,
                    "dst_frames": dst_frames,
                    "duration_s": round(dst_frames / target_fps, 2) if target_fps > 0 else 0,
                    "width": 640,
                    "height": 480,
                    "output_path": dst_path,
                })

                skipped += 1
                completed += 1
                print(f"  {target_fps:>4}fps  SKIP (exists: {dst_frames} frames)")
                continue

            # Do the downsampling
            try:
                info = downsample_video(src_path, dst_path, factor, target_fps)
                print(f"  {target_fps:>4}fps  OK   {info['src_frames']} → "
                      f"{info['dst_frames']} frames ({info['duration_s']}s)")

                manifest_rows.append({
                    "source_file": video_file,
                    "output_file": out_name,
                    "target_fps": target_fps,
                    "factor": factor,
                    "src_frames": info["src_frames"],
                    "dst_frames": info["dst_frames"],
                    "duration_s": info["duration_s"],
                    "width": 640,
                    "height": 480,
                    "output_path": dst_path,
                })
            except Exception as e:
                print(f"  {target_fps:>4}fps  FAIL {e}")

            completed += 1

        # Progress estimate
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (total_ops - completed) / rate if rate > 0 else 0
        print(f"  Progress: {completed}/{total_ops} "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Write manifest CSV
    manifest_path = os.path.join(output_dir, "manifest.csv")
    if manifest_rows:
        fieldnames = ["source_file", "output_file", "target_fps", "factor",
                      "src_frames", "dst_frames", "duration_s",
                      "width", "height", "output_path"]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print("  Summary")
    print(f"{'=' * 65}")
    print(f"  Videos processed: {len(videos)}")
    print(f"  Variants created: {completed - skipped}")
    print(f"  Variants skipped: {skipped} (already existed)")
    print(f"  Manifest:         {manifest_path}")
    print(f"  Time:             {elapsed:.1f}s")

    for t in targets:
        subdir = os.path.join(output_dir, f"{t}fps")
        n_files = len([f for f in os.listdir(subdir) if f.endswith(".mp4")])
        print(f"  {t:>4}fps: {n_files} files in {subdir}")

    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
