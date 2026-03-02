import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor
from src.data.video_loader import load_video_frames
import os
from src.utils.video_utils import get_video_category

class TemporalWaggleCollator:
    """
        Collate function to load the dataset into data loaders
    """
    def __call__(self, batch):
        videos = [sample['video'] for sample in batch]
        targets = [sample['targets'] for sample in batch]
        labels = [sample['label'] for sample in batch]
        metadata = [sample['metadata'] for sample in batch]

        return {
            "video": torch.stack(videos),
            "targets": torch.stack(targets),
            "label": torch.tensor(labels),
            "metadata": metadata
        }

class VideoYoloDataset(Dataset):    
    """
        A PyTorch Dataset for loading variable-resolution bee waggle dance videos and
        producing YOLO-style supervision for a temporal video model.

        This dataset performs:
            1. Video loading for a variable frame-range [start_frame, end_frame]
            2. Adaptive crop selection based on the expected bee size in pixels
            3. Random placement of the bee within the crop area (to add positional diversity)
            4. Resizing of the crop to a fixed resolution (width x height)
            5. Optional augmentations (flip, rotate, scale, etc.) that correctly update:
                    - (x, y) bee coordinates
                    - (dir_x, dir_y) body orientation vector
            6. Frame sampling/padding to produce a fixed-length temporal clip
            7. YOLO-style target encoding on a (grid_size x grid_size) grid, including:
                    - objectness
                    - relative cell-coordinates (cell_x, cell_y)
                    - normalized direction vector (dir_x, dir_y)
                    - normalized temporal start/end of waggle (start_norm, end_norm)

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Table describing training samples, containing:
                'video_name', 'start_frame', 'end_frame',
                'x1', 'y1' (bee position),
                'waggle', 'waggle_start_in_window', 'waggle_end_in_window',
                and optionally 'direction_x', 'direction_y'.
        video_dir : str
            Directory containing raw video files.
        transform : callable, optional
            Optional transformation applied to each frame (e.g., normalization).
        width, height : int
            Output spatial resolution of every crop (default = 224 x 224).
        clip_len : int
            Number of frames returned per sample (frames are sampled or repeated).
        grid_size : int
            Size of YOLO detection grid (g x g).
        max_detections_per_cell : int
            Limit of objects per grid cell (usually 1 for this task).
        num_classes : int
            Number of detection classes (default = 1 for bee).
        augment : callable, optional
            Augmentation module that modifies the clip and target dict.

    """
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
        self.augment = augment if is_training else None  # disable augment for val
        self.to_tensor = ToTensor()
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def _sample_frames(self, frames, target_length):
        """Uniformly sample frames"""
        num_frames = len(frames)
        if num_frames >= target_length:
            idxs = np.linspace(0, num_frames - 1, target_length, dtype=int)
            return [frames[i] for i in idxs]
        else:
            reps = frames * (target_length // num_frames + 1)
            return reps[:target_length]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]    
        #(f"\nSample {idx}:")
        #print(f"  Window: {row['start_frame']} to {row['end_frame']}")
        #print(f"  Waggle start in window: {row['waggle_start_in_window']}")
        #print(f"  Waggle end in window: {row['waggle_end_in_window']}")
        video_name = row['video_name']
        label = row['waggle'] 

        video_path = os.path.join(self.video_dir, video_name)         
        start_frame, end_frame = row["start_frame"], row["end_frame"]
        
        # Load frames
        #print('video_path:', video_path)
        frames = load_video_frames(video_path, start_frame, end_frame, use_cache=False)
        #print('Passes')
        if not frames:
            raise ValueError(f"No frames found for video {video_name}")

        gt_x, gt_y = row["x1"], row["y1"]
        first_frame = frames[0]
        
        if isinstance(first_frame, torch.Tensor):
            original_h, original_w = first_frame.shape[-2:]
        else:
            original_h, original_w = first_frame.shape[:2]

        crop_h, crop_w = self.height, self.width

        # Deterministic random crop
        rng = np.random.RandomState(seed=idx)
        margin_x = int(crop_w * 0.2)
        margin_y = int(crop_h * 0.2)
        max_offset_x = (crop_w // 2) - margin_x
        max_offset_y = (crop_h // 2) - margin_y
        offset_x = rng.randint(-max_offset_x, max_offset_x + 1)
        offset_y = rng.randint(-max_offset_y, max_offset_y + 1)

        x_min_ideal = int(gt_x - crop_w / 2) + offset_x
        y_min_ideal = int(gt_y - crop_h / 2) + offset_y
        x_min = max(0, min(original_w - crop_w, x_min_ideal))
        y_min = max(0, min(original_h - crop_h, y_min_ideal))
        x_max = x_min + crop_w
        y_max = y_min + crop_h

        # Crop and convert to tensors in one step avoids keeping np arrays
        frames = [self.to_tensor(f[y_min:y_max, x_min:x_max]) for f in frames]
        
        # Apply transform if needed
        if self.transform:
            frames = [self.transform(f) for f in frames]

        # GT in cropped region
        x = gt_x - x_min
        y = gt_y - y_min

        # Apply augmentations
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

        H, W = self.height, self.width

        # Frame sampling
        sampled_frames = self._sample_frames(frames, self.clip_len)
        frames_tensor = torch.stack(sampled_frames).permute(1, 0, 2, 3)
        
        # Clear intermediate variables
        del frames, sampled_frames

        # Target tensor
        target_tensor = torch.zeros(
            self.grid_size, 
            self.grid_size, 
            self.max_detections_per_cell, 
            7,
            dtype=torch.float32
        )

        if label == 1:
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
            idx_fps = video_name.find("fps")
            fps = video_name[idx_fps-2:idx_fps]
        else:
            category = get_video_category(video_name)
            if category == '0':
                fps = 15
            else:
                fps = 60
        
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