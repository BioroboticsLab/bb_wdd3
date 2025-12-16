import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor
from .video_loader import load_video_frames
from src.utils import get_video_category
import os

class VideoYoloDataset(Dataset):    
    def __init__(self, dataframe, video_dir, transform=None, width=224, height=224, 
                 clip_len=16, grid_size=25, max_detections_per_cell=1, num_classes=1,
                 augment=None, is_training=True):

        self.data = dataframe
        self.video_dir = video_dir
        self.transform = transform
        self.height = height
        self.width = width
        self.clip_len = clip_len
        self.grid_size = grid_size
        self.max_detections_per_cell = max_detections_per_cell
        self.num_classes = num_classes
        self.augment = augment if is_training else None
        self.to_tensor = ToTensor()
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def _sample_frames(self, frames, target_length):
        num_frames = len(frames)
        if num_frames >= target_length:
            idxs = np.linspace(0, num_frames - 1, target_length, dtype=int)
            return [frames[i] for i in idxs]
        else:
            reps = frames * (target_length // num_frames + 1)
            return reps[:target_length]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_name = row['video_name']
        label = row['waggle']

        video_path = os.path.join(self.video_dir, video_name)
        start_frame, end_frame = row["start_frame"], row["end_frame"]

        frames = load_video_frames(video_path, start_frame, end_frame)
        if not frames:
            raise ValueError(f"No frames found for video {video_name}")

        # ---------------------------
        # Coordinates before resize
        # ---------------------------
        gt_x, gt_y = row["x1"], row["y1"]

        # original resolution
        first_frame = frames[0]
        if isinstance(first_frame, torch.Tensor):
            original_h, original_w = first_frame.shape[-2:]
        else:
            original_h, original_w = first_frame.shape[:2]

        # ---------------------------
        # NO CROP → direct transform
        # ---------------------------
        frames = [self.to_tensor(f) if not torch.is_tensor(f) else f for f in frames]
        if self.transform:
            frames = [self.transform(f) for f in frames]   # includes resize 224×224

        # ---------------------------
        # RESCALE GT to resized frame
        # ---------------------------
        scale_x = self.width / original_w
        scale_y = self.height / original_h
        x = gt_x * scale_x
        y = gt_y * scale_y

        # ---------------------------
        # Apply augmentations
        # ---------------------------
        aug_info = None
        if self.augment:
            target_dict = {
                "x": x,
                "y": y,
                "dir_x": row.get('direction_x', 1.0),
                "dir_y": row.get('direction_y', 0.0)
            }
            frames, target_dict, aug_info = self.augment(frames, target_dict)
            x, y = target_dict["x"], target_dict["y"]
            dir_x, dir_y = target_dict["dir_x"], target_dict["dir_y"]
        else:
            dir_x, dir_y = row.get('direction_x', 1.0), row.get('direction_y', 0.0)

        # Frame sampling
        sampled_frames = self._sample_frames(frames, self.clip_len)
        frames_tensor = torch.stack(sampled_frames).permute(1, 0, 2, 3)

        # ---------------------------
        # Build YOLO target tensor
        # ---------------------------
        target_tensor = torch.zeros(
            self.grid_size,
            self.grid_size,
            self.max_detections_per_cell,
            7,
            dtype=torch.float32
        )

        if label == 1:
            H, W = self.height, self.width
            x_norm, y_norm = x / W, y / H

            ws_in = row['waggle_start_in_window']
            we_in = row['waggle_end_in_window']
            start_norm, end_norm = -1, -1

            if ws_in != -1 and we_in != -1:
                duration = end_frame - start_frame
                if duration > 0:
                    start_norm = (ws_in - start_frame) / duration
                    end_norm = (we_in - start_frame) / duration

            dir_norm = np.sqrt(dir_x**2 + dir_y**2)
            if dir_norm > 0:
                dir_x, dir_y = dir_x / dir_norm, dir_y / dir_norm
            else:
                dir_x, dir_y = 1.0, 0.0

            grid_x = int(x_norm * self.grid_size)
            grid_y = int(y_norm * self.grid_size)
            grid_x = max(0, min(self.grid_size - 1, grid_x))
            grid_y = max(0, min(self.grid_size - 1, grid_y))

            cell_x = (x_norm * self.grid_size) - grid_x
            cell_y = (y_norm * self.grid_size) - grid_y

            target_tensor[grid_y, grid_x, 0, 0] = 1.0
            target_tensor[grid_y, grid_x, 0, 1] = cell_x
            target_tensor[grid_y, grid_x, 0, 2] = cell_y
            target_tensor[grid_y, grid_x, 0, 3] = dir_x
            target_tensor[grid_y, grid_x, 0, 4] = dir_y
            target_tensor[grid_y, grid_x, 0, 5] = start_norm
            target_tensor[grid_y, grid_x, 0, 6] = end_norm

        if "fps" in video_name:
            idx = video_name.find("fps")
            fps = video_name[idx-2:idx]
        else:
            category = get_video_category(video_name)
            if category=='0':
                fps=15
            else:
                fps=60

        return {
            "video": frames_tensor,
            "targets": target_tensor,
            "label": label,
            "metadata": {
                "video_name": video_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "res": (original_h, original_w),
                "agmented": 1 if self.augment else 0,
                "augmentation": aug_info if self.augment else None,
                "original_coords": (row.get('x1', 0.0), row.get('y1', 0.0)),
                "fps": fps
            }
        }
