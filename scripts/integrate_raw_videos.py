"""
Integrate new raw videos (data/raw/videos/<BATCH>/) into the main dataset
(data/videos/), generating the same resolution/fps tier structure already
used for the existing Berlin/Nieh/Sharoni footage.

BATCH, DEST_PREFIX, VIDEO_EXT, ROTATE_DEGREES and RESOLUTION_DIVISORS are
all set once, together, in convert_raw_annotations.py's per-batch settings
block and imported from there -- a batch switch only ever means editing
that one block in that one file, never this one.

Only videos whose matching annotation CSV passed check_file() (see
convert_raw_annotations.py) are integrated -- a video is skipped whenever
its CSV was flagged, so the two never drift out of sync with each other.

No batch before bristol_2026 ever had more than 3 resolution tiers
(Berlin/Nieh/old Sharoni all top out at 3) -- [1, 2, 4, 8] was introduced
specifically for the Sharoni/Jerusalem batch's higher native resolution,
not an established convention to default to.

FPS tiers: ds30fps / ds15fps are generated for each resolution tier, but
only for source videos whose native fps is actually above that target
(skipped otherwise -- matches how the existing dataset only has fps
variants for ~60fps-native sources).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from convert_raw_annotations import (
    check_file, guess_video_filename, RAW_ANNOT_DIR, RAW_VIDEO_DIR, DEST_PREFIX,
    ROTATE_DEGREES, RESOLUTION_DIVISORS,
)

OUT_VIDEO_DIR = "./data/videos"

# RESOLUTION_DIVISORS / ROTATE_DEGREES are imported from
# convert_raw_annotations.py (the single per-batch settings block) rather
# than defined here, so switching batches only ever means editing one
# file. Derivation notes, kept here since they're specific to each batch:
#
#   berlin_2024: native (post-rotation) is 4608x3520 -- almost exactly 2x
#     the existing Berlin camera's 2304x1760, so skipping this batch's own
#     native tier and using /2,/4,/8 lands on exactly the same three
#     resolutions (2304x1760 / 1152x880 / 576x440) the rest of the Berlin
#     dataset already uses, instead of introducing a new resolution.
#     Source videos were portrait (3520x4608) with no rotation metadata in
#     the container -- confirmed by inspection the recording was never
#     configured for landscape (not a flag being ignored by some tool).
#     Rotating 90deg clockwise (ffmpeg transpose=1, verified against an
#     extracted frame) produced the correct landscape orientation.
#
#   bristol_2026: native 1920x1080, already landscape (no rotation
#     needed). No existing Bristol footage to match, so this is a fresh
#     native/half/quarter tier set (1920x1080 / 960x540 / 480x270),
#     following the same "max 3 tiers" convention every other lab uses.
#
#   ucsd_2026: source is UCSD's own "Downsampled Annotated Videos" export
#     (ucsd_datashare_globus on trove) -- already landscape at a uniform
#     960x540 native (confirmed: 60/60 randomly sampled videos matched),
#     not the higher-res "_original.MP4" UCSD also has but we don't. This
#     already matches the dominant convention among the 74 pre-existing
#     Nieh videos (93% of those use 960x540 as their largest tier, not
#     1920x1080 -- checked directly against data/videos/ rather than
#     assumed), so no new resolution is being introduced: native + /2
#     (960x540 / 480x270), matching that existing majority pattern.

FPS_TARGETS = [30, 15]


def ffprobe_wh_fps(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0", path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h, fr = out.split(",")
    if "/" in fr:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) != 0 else float(num)
    else:
        fps = float(fr)
    return int(w), int(h), fps


def run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "1"] + args, check=True)


def get_good_videos():
    """Mirror convert_raw_annotations.py's per-file check to get the set of
    videos that are cleared for integration. A CSV that passes check_file()
    but has zero non-empty rows (e.g. reviewed-but-no-dances, or just not
    yet annotated -- check_file() can't tell which) is kept out of `good`
    too: encoding its video would produce a file with no annotation rows
    anywhere in the pipeline, so it would never be used as either a
    positive or negative training example -- just disk space with no
    purpose until the CSV's actual status is confirmed."""
    csv_files = sorted(f for f in os.listdir(RAW_ANNOT_DIR) if f.endswith(".csv"))
    good, bad = [], []
    for fname in csv_files:
        is_clean, issues, rows = check_file(fname)
        video_name = guess_video_filename(fname)
        if is_clean and len(rows) == 0:
            is_clean = False
            issues = ["0 annotation rows -- reviewed-empty or not-yet-annotated? confirm before integrating"]
        (good if is_clean else bad).append((fname, video_name, issues))
    return good, bad


TRANSPOSE_FILTER = {90: "transpose=1", 270: "transpose=2"}  # clockwise / counter-clockwise


def integrate_one(base_name, src_path, w, h, fps, rotate_degrees=0):
    """Generate all resolution/fps tiers for one source video. w, h are the
    POST-rotation (final, intended) dimensions -- caller swaps them before
    calling this if rotate_degrees is 90/270."""
    produced = []
    tiers = [(w // d, h // d) for d in RESOLUTION_DIVISORS]

    for (tw, th) in tiers:
        native_out = os.path.join(OUT_VIDEO_DIR, f"{base_name}_{tw}_{th}.mp4")
        if os.path.exists(native_out):
            print(f"    SKIP (already exists): {native_out}")
        else:
            vf_parts = []
            if rotate_degrees in TRANSPOSE_FILTER:
                vf_parts.append(TRANSPOSE_FILTER[rotate_degrees])
            if (tw, th) != (w, h):
                vf_parts.append(f"scale={tw}:{th}")
            if vf_parts:
                run_ffmpeg(["-i", src_path, "-vf", ",".join(vf_parts),
                            "-c:v", "libx264", "-crf", "18", native_out])
            else:
                run_ffmpeg(["-i", src_path, "-c:v", "libx264", "-crf", "18", native_out])
            produced.append(native_out)

        for target in FPS_TARGETS:
            if fps > target:
                fps_out = os.path.join(OUT_VIDEO_DIR, f"{base_name}_{tw}_{th}_ds{target}fps.mp4")
                if os.path.exists(fps_out):
                    print(f"    SKIP (already exists): {fps_out}")
                    continue
                run_ffmpeg(["-i", native_out, "-vf", f"fps={target}",
                            "-c:v", "libx264", "-crf", "18", fps_out])
                produced.append(fps_out)

    return produced


def main():
    os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
    good, bad = get_good_videos()

    print(f"Videos cleared for integration: {len(good)}")
    print(f"Videos skipped (annotation issues): {len(bad)}")
    print()

    for fname, video_name, issues in bad:
        print(f"  SKIP video (bad annotation): {video_name}")

    for fname, video_name, _ in good:
        src_path = os.path.join(RAW_VIDEO_DIR, video_name)
        if not os.path.exists(src_path):
            print(f"  MISSING source video for {video_name} -- expected at {src_path}, skipping")
            continue

        base_name = DEST_PREFIX + os.path.splitext(video_name)[0]
        w_raw, h_raw, fps = ffprobe_wh_fps(src_path)
        if ROTATE_DEGREES in TRANSPOSE_FILTER:
            w, h = h_raw, w_raw
            print(f"Integrating {video_name} ({w_raw}x{h_raw} @ {fps:.1f}fps, "
                  f"rotating {ROTATE_DEGREES}deg -> {w}x{h})...")
        else:
            w, h = w_raw, h_raw
            print(f"Integrating {video_name} ({w}x{h} @ {fps:.1f}fps)...")
        produced = integrate_one(base_name, src_path, w, h, fps, rotate_degrees=ROTATE_DEGREES)
        for p in produced:
            print(f"    -> {p}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
