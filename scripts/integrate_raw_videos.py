"""
Integrate new raw videos (data/raw/videos/) into the main dataset
(data/videos/), generating the same resolution/fps tier structure already
used for the existing Berlin/Nieh/Sharoni footage.

Only videos whose matching annotation CSV passed check_file() (see
convert_raw_annotations.py) are integrated -- a video is skipped whenever
its CSV was flagged, so the two never drift out of sync with each other.

Resolution tiers: native, /2, /4, /8 (matches the existing Sharoni
convention: 2448x2048 native -> 1224x1024 -> 612x512 -> 306x256).
FPS tiers: ds30fps / ds15fps are generated for each resolution tier, but
only for source videos whose native fps is actually above that target
(skipped otherwise -- matches how the existing dataset only has fps
variants for ~60fps-native sources).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from convert_raw_annotations import check_file, guess_video_filename, RAW_ANNOT_DIR, DEST_PREFIX

RAW_VIDEO_DIR = "./data/raw/videos"
OUT_VIDEO_DIR = "./data/videos"
RESOLUTION_DIVISORS = [1, 2, 4, 8]
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
    videos that are cleared for integration."""
    csv_files = sorted(f for f in os.listdir(RAW_ANNOT_DIR) if f.endswith(".csv"))
    good, bad = [], []
    for fname in csv_files:
        is_clean, issues, _ = check_file(fname)
        video_name = guess_video_filename(fname)
        (good if is_clean else bad).append((fname, video_name, issues))
    return good, bad


def integrate_one(base_name, src_path, w, h, fps):
    """Generate all resolution/fps tiers for one source video."""
    produced = []
    tiers = []
    for d in RESOLUTION_DIVISORS:
        tw, th = w // d, h // d
        tiers.append((tw, th))

    for (tw, th) in tiers:
        native_out = os.path.join(OUT_VIDEO_DIR, f"{base_name}_{tw}_{th}.mp4")
        if os.path.exists(native_out):
            print(f"    SKIP (already exists): {native_out}")
        else:
            if tw == w and th == h:
                run_ffmpeg(["-i", src_path, "-c:v", "libx264", "-crf", "18", native_out])
            else:
                run_ffmpeg(["-i", src_path, "-vf", f"scale={tw}:{th}",
                            "-c:v", "libx264", "-crf", "18", native_out])
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
        w, h, fps = ffprobe_wh_fps(src_path)
        print(f"Integrating {video_name} ({w}x{h} @ {fps:.1f}fps)...")
        produced = integrate_one(base_name, src_path, w, h, fps)
        for p in produced:
            print(f"    -> {p}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
