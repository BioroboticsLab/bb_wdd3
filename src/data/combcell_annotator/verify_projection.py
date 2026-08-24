#!/usr/bin/env python3
"""
Verify Projection

Sanity-checks the resolution projection annotator.py relies on: for each
annotated comb cell, take the click made on the highest-resolution copy,
scale it by the exact resolution ratio, and draw it onto the matching frame
of a downsampled sibling video. Saves a side-by-side image (native frame with
the original annotation | downsampled frame with the projected annotation)
so you can eyeball whether it still lands on the same cell at the right size.

Usage:
    python verify_projection.py [path/to/annotations_circle_TIMESTAMP.csv]

If no path is given, uses the most recently written annotations_*.csv in
output/.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from annotator import extract_frame, get_video_files, parse_resolution, base_stem


def find_latest_csv() -> Path:
    output_dir = Path(__file__).parent / "output"
    candidates = sorted(output_dir.glob("annotations_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No annotations_*.csv found in {output_dir}")
    return candidates[-1]


def find_sibling_files(video_path: Path) -> dict:
    """Return {resolution: file_path} for every resolution tier available for
    this video's source recording (one file per unique resolution, even if
    multiple fps variants exist at that resolution)."""
    stem = base_stem(video_path.name)
    siblings = {}
    for v in get_video_files(video_path.parent):
        if base_stem(v.name) != stem:
            continue
        res = parse_resolution(v.name)
        if res and res not in siblings:
            siblings[res] = v
    return siblings


def draw_circle(frame, cx, cy, r, color):
    frame = frame.copy()
    cv2.circle(frame, (int(cx), int(cy)), int(r), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
    return frame


def draw_hexagon(frame, pts, color):
    frame = frame.copy()
    arr = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2)
    for p in arr:
        cv2.circle(frame, tuple(p), 4, color, -1)
    return frame


def side_by_side(native_img, sib_img):
    # Scale the sibling (smaller) image up to the native frame's height so
    # both sit at a comparable size for a visual check.
    nh, nw = native_img.shape[:2]
    sh, sw = sib_img.shape[:2]
    sib_disp = cv2.resize(sib_img, (int(sw * nh / sh), nh))
    return np.hstack([native_img, sib_disp])


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_csv()
    print(f"Reading: {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No annotations in that CSV.")
        return

    is_hexagon = "hex_idx" in rows[0]

    out_dir = Path(__file__).parent / "output" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for row in rows:
        video_path = Path(row["video"])
        if not video_path.exists():
            print(f"  [Skip] video not found: {video_path}")
            continue
        if not row["res_w"] or not row["res_h"]:
            continue
        native_res = (int(row["res_w"]), int(row["res_h"]))
        frame_idx = int(row["frame_idx"])

        try:
            native_frame = extract_frame(video_path, frame_idx)
        except ValueError:
            print(f"  [Skip] could not read frame {frame_idx} from {video_path.name}")
            continue

        for sib_res, sib_path in find_sibling_files(video_path).items():
            if sib_res == native_res:
                continue
            scale = sib_res[0] / native_res[0]

            try:
                sib_frame = extract_frame(sib_path, frame_idx)
            except ValueError:
                print(f"  [Skip] could not read frame {frame_idx} from {sib_path.name}")
                continue

            if is_hexagon:
                pts = [(float(row[f"x{i}"]), float(row[f"y{i}"])) for i in range(1, 7)]
                native_img = draw_hexagon(native_frame, pts, (0, 220, 0))
                sib_img = draw_hexagon(sib_frame, [(x * scale, y * scale) for x, y in pts], (0, 165, 255))
                idx_field = row["hex_idx"]
            else:
                cx, cy, r = float(row["center_x"]), float(row["center_y"]), float(row["radius_px"])
                native_img = draw_circle(native_frame, cx, cy, r, (0, 220, 0))
                sib_img = draw_circle(sib_frame, cx * scale, cy * scale, r * scale, (0, 165, 255))
                idx_field = row["circle_idx"]

            combined = side_by_side(native_img, sib_img)
            fname = (f"{video_path.stem}_frame{frame_idx}_ann{idx_field}"
                      f"_{native_res[0]}x{native_res[1]}_to_{sib_res[0]}x{sib_res[1]}.png")
            cv2.imwrite(str(out_dir / fname), combined)
            n_written += 1

    print(f"\nWrote {n_written} comparison image(s) to {out_dir}")
    print("Left = native annotation (green). Right = projected onto downsampled sibling (orange).")


if __name__ == "__main__":
    main()
