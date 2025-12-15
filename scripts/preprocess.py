import cv2
import os
import pandas as pd
import argparse
import ast
import math
from pathlib import Path

# ======================================================
# ARGUMENTS
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand waggle annotations and/or downsample videos + CSVs"
    )

    # -------- Modes --------
    parser.add_argument("--expand_only", action="store_true")
    parser.add_argument("--downsample_only", action="store_true")
    parser.add_argument("--expand_and_downsample", action="store_true")

    # -------- Inputs --------
    parser.add_argument("--video_folder", type=str)
    parser.add_argument("--annotation_folder", type=str)
    parser.add_argument("--annotation_csv", type=str)

    # -------- Expansion --------
    parser.add_argument("--expanded_annotation_folder", type=str)

    # -------- Downsampling outputs --------
    parser.add_argument("--output_video_folder", type=str)
    parser.add_argument("--output_csv_folder", type=str)

    # -------- Downsampling params --------
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--target_width", type=int, default=960)
    parser.add_argument("--target_height", type=int, default=540)

    return parser.parse_args()


# ======================================================
# UTILS
# ======================================================
def get_annotation_csvs(annotation_folder=None, annotation_csv=None):
    if annotation_folder:
        return list(Path(annotation_folder).glob("*.csv"))
    elif annotation_csv:
        return [Path(annotation_csv)]
    else:
        raise ValueError("Provide --annotation_folder or --annotation_csv")


# ======================================================
# EXPAND ANNOTATIONS
# ======================================================
def expand_annotation_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required_cols = [
        "thorax_positions", "thorax_frames",
        "waggle_start_positions", "waggle_start_frames",
        "waggle_directions", "video_name"
    ]

    if not all(col in df.columns for col in required_cols):
        print(f"Skipped (missing columns): {input_csv.name}")
        return None

    expanded_rows = []

    for _, row in df.iterrows():
        thorax_positions = ast.literal_eval(row["thorax_positions"])
        thorax_frames = ast.literal_eval(row["thorax_frames"])
        waggle_positions = ast.literal_eval(row["waggle_start_positions"])
        waggle_frames = ast.literal_eval(row["waggle_start_frames"])
        waggle_dirs = ast.literal_eval(row["waggle_directions"])

        for t_pos, t_frame, w_pos, w_frame, w_dir in zip(
                thorax_positions, thorax_frames,
                waggle_positions, waggle_frames,
                waggle_dirs):

            expanded_rows.append({
                "video_name": row["video_name"],
                "start_frame": w_frame,
                "end_frame": t_frame,
                "x1": w_pos[0],
                "y1": w_pos[1],
                "x2": t_pos[0],
                "y2": t_pos[1],
                "angle": math.atan2(w_dir[0], w_dir[1]),
                "waggle": 1,
                "origin_x": w_pos[0],
                "origin_y": w_pos[1],
                "direction_x": w_dir[0],
                "direction_y": w_dir[1]
            })

    out_df = pd.DataFrame(expanded_rows)
    out_df.to_csv(output_csv, index=False)

    print(f"✓ Expanded: {output_csv.name}")
    return output_csv


# ======================================================
# VIDEO DOWNSAMPLING
# ======================================================
def downsample_video(video_path, output_video_path, frame_stride, target_w, target_h):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    og_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    og_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    sx = target_w / og_w
    sy = target_h / og_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_video_path), fourcc, fps / frame_stride, (target_w, target_h)
    )

    frame_map = {}
    frame_idx = 0
    new_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            resized = cv2.resize(frame, (target_w, target_h))
            out.write(resized)
            frame_map[frame_idx] = new_idx
            new_idx += 1

        frame_idx += 1

    cap.release()
    out.release()
    return frame_map, sx, sy


# ======================================================
# CSV DOWNSAMPLING
# ======================================================
def downsample_csv(df, output_csv, frame_map, sx, sy, new_video_name):
    df = df[df["start_frame"].isin(frame_map.keys())]
    if df.empty:
        return None

    df = df.copy()

    df["start_frame"] = df["start_frame"].map(frame_map)
    df["end_frame"] = df["end_frame"].apply(
        lambda f: frame_map.get(f, max(frame_map.values()))
    )

    for col in ["x1", "x2"]:
        df[col] *= sx
    for col in ["y1", "y2"]:
        df[col] *= sy

    df["video_name"] = new_video_name
    df.to_csv(output_csv, index=False)
    return df


# ======================================================
# MAIN
# ======================================================
def main():
    args = parse_args()

    modes = [
        args.expand_only,
        args.downsample_only,
        args.expand_and_downsample
    ]

    if sum(modes) != 1:
        raise ValueError(
            "Choose exactly ONE mode:\n"
            "--expand_only | --downsample_only | --expand_and_downsample"
        )

    # =========================
    # EXPAND ONLY
    # =========================
    if args.expand_only:
        if not (args.annotation_csv or args.annotation_folder):
            raise ValueError(
                "For --expand_only, provide --annotation_csv OR --annotation_folder"
            )
        assert args.expanded_annotation_folder, "--expanded_annotation_folder required"
        os.makedirs(args.expanded_annotation_folder, exist_ok=True)

        csvs = get_annotation_csvs(args.annotation_folder, args.annotation_csv)

        for csv_path in csvs:
            out_csv = Path(args.expanded_annotation_folder) / f"{csv_path.stem}_expanded.csv"
            expand_annotation_csv(csv_path, out_csv)

        print("\n✓ Expansion complete (no downsampling)")
        return

    # =========================
    # EXPAND → DOWNSAMPLE
    # =========================
    if args.expand_and_downsample:
        assert args.expanded_annotation_folder, "--expanded_annotation_folder required"
        os.makedirs(args.expanded_annotation_folder, exist_ok=True)

        csvs = get_annotation_csvs(args.annotation_folder, args.annotation_csv)

        for csv_path in csvs:
            out_csv = Path(args.expanded_annotation_folder) / f"{csv_path.stem}_expanded.csv"
            expand_annotation_csv(csv_path, out_csv)

        args.annotation_folder = args.expanded_annotation_folder
        args.annotation_csv = None
        print("\n✓ Expansion done. Starting downsampling...\n")

    # =========================
    # DOWNSAMPLE ONLY (or after expansion)
    # =========================
    os.makedirs(args.output_video_folder, exist_ok=True)
    os.makedirs(args.output_csv_folder, exist_ok=True)

    for video_path in Path(args.video_folder).glob("*.mp4"):
        base = video_path.stem
        print(f"Processing video: {base}")

        csv_path = Path(args.annotation_folder) / f"{base}_expanded.csv"
        if not csv_path.exists():
            print("  Skipped (no matching CSV)")
            continue

        df = pd.read_csv(csv_path)

        out_video = Path(args.output_video_folder) / \
            f"{base}_{args.target_width}_{args.target_height}.mp4"

        out_csv = Path(args.output_csv_folder) / \
            f"{base}_{args.target_width}_{args.target_height}_expanded.csv"

        frame_map, sx, sy = downsample_video(
            video_path,
            out_video,
            args.frame_stride,
            args.target_width,
            args.target_height
        )

        downsample_csv(df, out_csv, frame_map, sx, sy, out_video.name)
        print(f"✓ Finished: {out_video.name}\n")


# ======================================================
if __name__ == "__main__":
    main()
