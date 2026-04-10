import sys
import csv
import ast
import json
import subprocess
from pathlib import Path
from collections import defaultdict

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pandas as pd
except ImportError:
    pd = None

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
SUFFIXES_TO_STRIP = ("_downsample", "_original", "_reencoded", "_reencoded2", "_reencoded_", "_trimmed", "_resized")


def normalize_name(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("./"):
        text = text[2:]
    text = text.replace('\\', '/').strip()
    text = Path(text).stem
    text = text.lower().strip()
    for suffix in SUFFIXES_TO_STRIP:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip()


def get_video_info(video_path):
    """Returns width, height and fps for the video file.

    Uses OpenCV when available, and falls back to ffprobe if needed.
    """
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return width, height, round(fps, 2)

    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"unable to probe video file {video_path}")

    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width")
            height = stream.get("height")
            fps_str = stream.get("r_frame_rate") or stream.get("avg_frame_rate", "?")
            num, den = fps_str.split("/")
            fps = round(int(num) / int(den), 2)
            return width, height, fps

    raise RuntimeError(f"no video stream found in {video_path}")


def count_dances(csv_path):
    """Counts waggle dances in a CSV annotation file."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "dance_number" in fieldnames:
            unique_numbers = set()
            for row in reader:
                raw = row.get("dance_number")
                if raw is None:
                    continue
                raw = str(raw).strip()
                if raw == "" or raw.lower() == "nan":
                    continue
                unique_numbers.add(raw)
            return len(unique_numbers)

        if "waggle_start_frames" in fieldnames:
            total = 0
            for row in reader:
                raw = row.get("waggle_start_frames", "[]")
                if raw is None:
                    continue
                raw = str(raw).strip()
                if raw == "" or raw.lower() == "nan":
                    continue
                try:
                    frames = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    frames = []
                total += len(frames)
            return total

        if "start_frame" in fieldnames and "end_frame" in fieldnames:
            count = 0
            for row in reader:
                if row.get("start_frame") or row.get("end_frame"):
                    count += 1
            return count

        count = 0
        for row in reader:
            if any((str(row.get(col)).strip() not in ("", "nan", "None") for col in fieldnames)):
                count += 1
        return count


def load_xlsx_metadata(base):
    """Loads optional XLSX metadata summarizing videos and annotation files."""
    if pd is None:
        return []

    metadata_files = sorted(base.glob("*.xlsx"))
    info = []

    for xlsx_file in metadata_files:
        try:
            xls = pd.ExcelFile(xlsx_file)
        except Exception:
            continue

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
            except Exception:
                continue

            columns = [c.strip() for c in df.columns.astype(str)]
            df.columns = columns
            video_col = next((c for c in columns if "video" in c.lower()), None)
            ann_col = next((c for c in columns if "annotation" in c.lower()), None)
            if video_col is None or ann_col is None:
                continue

            mappings = []
            current_video = None
            current_ann = None
            for _, row in df.iterrows():
                video_val = row.get(video_col)
                ann_val = row.get(ann_col)
                if isinstance(video_val, str) and video_val.strip():
                    current_video = video_val.strip()
                if isinstance(ann_val, str) and ann_val.strip():
                    current_ann = ann_val.strip()
                if current_video or current_ann:
                    mappings.append((current_video, current_ann))

            info.append({
                "xlsx_file": xlsx_file.name,
                "sheet": sheet_name,
                "row_count": len(df),
                "unique_videos": len({v for v, _ in mappings if v}),
                "unique_annotations": len({a for _, a in mappings if a}),
            })
            break

    return info


def build_annotation_index(base):
    csv_paths = sorted(base.glob("*.csv"))
    prefix_map = defaultdict(list)
    video_name_map = defaultdict(list)
    all_csvs = []

    for csv_path in csv_paths:
        stem = normalize_name(csv_path.stem)
        kind = "unknown"
        prefix = stem
        if stem.endswith("_waggle_annotations_simple"):
            kind = "simple"
            prefix = stem[: -len("_waggle_annotations_simple")]
        elif stem.endswith("_waggle_annotations"):
            kind = "raw"
            prefix = stem[: -len("_waggle_annotations")]

        annotation = {"path": csv_path, "kind": kind, "prefix": prefix, "stem": stem}
        all_csvs.append(annotation)
        prefix_map[prefix].append(annotation)

    for annotation in all_csvs:
        try:
            with open(annotation["path"], newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = [h.strip().lower() for h in (reader.fieldnames or [])]
                if "video_name" not in headers:
                    continue
                for row in reader:
                    raw = row.get("video_name")
                    name = normalize_name(raw)
                    if name:
                        video_name_map[name].append(annotation)
        except Exception:
            continue

    return prefix_map, video_name_map, csv_paths


def find_annotation_candidate(video, prefix_map, video_name_map):
    normalized = normalize_name(video.name)
    candidates = []
    if normalized in prefix_map:
        candidates.extend(prefix_map[normalized])
    if normalized in video_name_map:
        candidates.extend(video_name_map[normalized])

    if candidates:
        return candidates

    for prefix, annotations in prefix_map.items():
        if normalized.startswith(prefix) or prefix.startswith(normalized):
            candidates.extend(annotations)
    return candidates


def choose_best_annotation_csv(candidates):
    if not candidates:
        return None
    simple = [c for c in candidates if c["kind"] == "simple"]
    if simple:
        return simple[0]["path"]
    raw = [c for c in candidates if c["kind"] == "raw"]
    if raw:
        return raw[0]["path"]
    return candidates[0]["path"]


def main():
    if len(sys.argv) < 2:
        print("usage: python src/data/fetch_wdd_data_info.py <folder>")
        sys.exit(1)

    folder = sys.argv[1]
    base = Path(folder)
    if not base.exists():
        raise FileNotFoundError(f"folder not found: {base}")

    videos = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    csv_paths = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    print(f"found {len(videos)} videos in {base}")
    print(f"found {len(csv_paths)} CSV annotation files in {base}\n")

    metadata = load_xlsx_metadata(base)
    if metadata:
        print(f"found {len(metadata)} xlsx metadata file(s)")
        for item in metadata:
            print(f"  {item['xlsx_file']} (sheet: {item['sheet']}, rows: {item['row_count']}, unique videos: {item['unique_videos']}, unique annotations: {item['unique_annotations']})")
        print()

    prefix_map, video_name_map, all_csvs = build_annotation_index(base)
    resolutions = defaultdict(list)
    fps_map = defaultdict(list)
    total_dances = 0
    missing_csv = []
    matched_videos = []
    video_csv_map = {}

    for video in videos:
        try:
            width, height, fps = get_video_info(video)
        except Exception as exc:
            print(f"warning: could not read video info for {video.name}: {exc}")
            continue
        resolutions[f"{width}x{height}"].append(video.name)
        fps_map[fps].append(video.name)

        candidates = find_annotation_candidate(video, prefix_map, video_name_map)
        csv_path = choose_best_annotation_csv(candidates)
        if csv_path is None:
            missing_csv.append(video.name)
            continue

        video_csv_map[video.name] = csv_path.name
        matched_videos.append(video.name)
        total_dances += count_dances(csv_path)

    print("resolution summary")
    if len(resolutions) == 1:
        print(f"resolution: {list(resolutions.keys())[0]} (all videos)")
    else:
        print("resolution: multiple found")
        for res, names in sorted(resolutions.items()):
            print(f"  {res}: {len(names)} videos")
            for n in names:
                print(f"    {n}")

    print("\nfps summary")
    if len(fps_map) == 1:
        print(f"fps: {list(fps_map.keys())[0]} (all videos)")
    else:
        print("fps: multiple found")
        for fps, names in sorted(fps_map.items()):
            print(f"  {fps} fps: {len(names)} videos")
            for n in names:
                print(f"    {n}")

    print(f"\ntotal waggle dances annotated: {total_dances}")
    print(f"videos with annotations: {len(matched_videos)}/{len(videos)}")

    if missing_csv:
        print("\nno csv found for:")
        for n in missing_csv:
            print(f"  {n}")

    matched_csv_names = {n for n in video_csv_map.values()}
    unmatched_csvs = [p.name for p in csv_paths if p.name not in matched_csv_names]
    if unmatched_csvs:
        print("\nunmatched CSV annotation files:")
        for n in unmatched_csvs:
            print(f"  {n}")

    if video_csv_map:
        print("\nvideo -> annotation mapping")
        for video_name, csv_name in sorted(video_csv_map.items()):
            print(f"  {video_name} -> {csv_name}")


if __name__ == "__main__":
    main()