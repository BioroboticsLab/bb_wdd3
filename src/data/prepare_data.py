import pandas as pd
import os
import cv2
cv2.setNumThreads(1)


def prepare_dataset(csv_path, clip_len=32, pos=False):
    df = pd.read_csv(csv_path)

    all_clips_data = []

    for id, row in df.iterrows():
        if pos and row['waggle']==0:
            continue

        video_duration_frames = row["end_frame"] - row["start_frame"]
        
        if video_duration_frames < clip_len:
            num_segments_to_create = 1            
            num_segments_to_create = max(1, video_duration_frames - clip_len + 1)
            
        else:
            num_segments_to_create = video_duration_frames - clip_len + 1

        for i in range(num_segments_to_create):
            clip_info = {
                "video_name": row["video_name"],
                "start_frame": row["start_frame"] + i,
                "end_frame": row["start_frame"] + i + clip_len -1 if video_duration_frames >= clip_len else row["end_frame"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "angle": row["angle"],
                "waggle": row["waggle"]
            }
            all_clips_data.append(clip_info)
            
    data = pd.DataFrame(all_clips_data)
    return data

def extract_frames_from_video(video_name, width=224, height=224):
    video_path = os.path.join(os.curdir, "/home/prajna/complete_data/videos/", video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()
    return frames

def create_video_frames_df(video_names, width=224, height=224):
    dict = {}
    for video_name in video_names:
        frames = extract_frames_from_video(video_name, width=width,height=height)

        dict[video_name] = frames

    return dict

        