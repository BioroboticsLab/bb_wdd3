import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor
from src.data.video_loader import load_video_frames
import os
from src.utils.video_utils import get_video_category
from collections import Counter

# Same dataset as dataset.py, except the temporal target is no longer two
# regression scalars (start_norm, end_norm). Instead each of the `window_size`
# sampled frames gets its own 0/1 "is this frame part of the waggle" label,
# so the temporal head (in model_temporal_cls.py) can be trained as a
# per-frame classifier instead of a single pooled regression. See the
# "per-frame classification" plan we scoped out for fixing the dance-level
# temporal IoU weakness.
#
# target_tensor layout is now (grid, grid, max_det, 5 + window_size):
#   [0] confidence, [1:3] cell_x/cell_y, [3:5] dir_x/dir_y,
#   [5:5+window_size] per-frame waggle labels (0.0 or 1.0)


class TemporalWaggleCollator:
    """Collate function -- identical to the base one, just stacks whatever
    shape the target tensor happens to be, so it doesn't care about the
    5+window_size vs 7 difference."""
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


class VideoYoloDatasetTemporalCls(Dataset):
    """
        Same as VideoYoloDataset, but the temporal target is per-frame
        classification labels instead of two normalized start/end scalars.
        See module docstring above for the target layout.

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
        window_size : int
            Number of frames returned per sample (frames are sampled or repeated).
        grid_size : int
            Size of YOLO detection grid (g x g).
        max_detections_per_cell : int
            Limit of objects per grid cell (usually 1 for this task).
        n_classes : int
            Number of detection classes (default = 1 for bee).
        augment : callable, optional
            Augmentation module that modifies the clip and target dict.
    """
    def __init__(self, dataframe, video_dir, transform=None, width=224, height=224,
                 window_size=16, grid_size=25, max_detections_per_cell=1, n_classes=1,
                 augment=None, is_training=True):

        self.data = dataframe
        self.video_dir = video_dir
        self.transform = transform
        self.height = height
        self.width = width
        self.window_size = window_size
        self.grid_size = grid_size
        self.max_detections_per_cell = max_detections_per_cell
        self.n_classes = n_classes
        self.augment = augment if is_training else None  # disable augment for val
        self.to_tensor = ToTensor()
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def count_categories(self):
        """Count samples per video category."""
        counts = Counter(
            get_video_category(name) for name in self.data['video_name']
        )
        print("Category counts:")
        for cat, count in sorted(counts.items()):
            print(f"  {cat:10s}: {count}")
        return counts

    def _sample_frame_indices(self, num_frames, target_length):
        """Which raw-frame index each of the `target_length` model-input
        slots comes from -- same linspace/repeat-pad logic as the old
        _sample_frames, just returning indices instead of the frames
        themselves, since we need those indices to build per-frame labels.
        Keep this in sync with the copy in eval_utils.py's decode functions,
        so a predicted "frame t" maps back to the same real frame number at
        both train and eval time."""
        if num_frames >= target_length:
            return np.linspace(0, num_frames - 1, target_length, dtype=int).tolist()
        else:
            reps = list(range(num_frames)) * (target_length // num_frames + 1)
            return reps[:target_length]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_name = row['video_name']
        label = row['waggle']

        video_path = os.path.join(self.video_dir, video_name)
        start_frame, end_frame = row["start_frame"], row["end_frame"]

        # Load frames
        frames = load_video_frames(video_path, start_frame, end_frame, use_cache=False)
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

        # Frame sampling -- grab the indices too this time, so we know which
        # real frame each of the window_size slots corresponds to
        sampled_idxs = self._sample_frame_indices(len(frames), self.window_size)
        sampled_frames = [frames[i] for i in sampled_idxs]
        frames_tensor = torch.stack(sampled_frames).permute(1, 0, 2, 3)

        del frames, sampled_frames

        # Target tensor -- last dim is now 5 + window_size (per-frame labels
        # instead of 2 start/end scalars)
        target_tensor = torch.zeros(
            self.grid_size,
            self.grid_size,
            self.max_detections_per_cell,
            5 + self.window_size,
            dtype=torch.float32
        )

        if label == 1:
            x_norm, y_norm = x / W, y / H
            ws_in = row['waggle_start_in_window']
            we_in = row['waggle_end_in_window']

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

            # per-frame labels: 1.0 if that sampled frame's real frame number
            # falls inside [ws_in, we_in]. If we don't actually know the
            # waggle boundary (-1 sentinel), just leave all frames at 0 --
            # unlike the old -1/-1 sentinel, BCE needs real 0/1 targets.
            if ws_in != -1 and we_in != -1:
                for t, fidx in enumerate(sampled_idxs):
                    real_frame = start_frame + fidx
                    if ws_in <= real_frame <= we_in:
                        target_tensor[grid_y, grid_x, 0, 5 + t] = 1.0

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

    def get_item_debug(self, idx, crop_mode='inference'):
        """Same pipeline-inspector helper as the base dataset -- mirrors
        __getitem__ line-for-line, just with the per-frame target layout."""
        row = self.data.iloc[idx]
        video_name = row['video_name']
        label = row['waggle']
        video_path = os.path.join(self.video_dir, video_name)
        start_frame, end_frame = row["start_frame"], row["end_frame"]

        raw_frames = load_video_frames(video_path, start_frame, end_frame, use_cache=False)
        if not raw_frames:
            raise ValueError(f"No frames found for video {video_name}")

        gt_x, gt_y = row["x1"], row["y1"]
        first_frame = raw_frames[0]
        if isinstance(first_frame, torch.Tensor):
            original_h, original_w = first_frame.shape[-2:]
        else:
            original_h, original_w = first_frame.shape[:2]

        crop_h, crop_w = self.height, self.width

        if crop_mode == 'training':
            rng = np.random.RandomState(seed=idx)
            margin_x = int(crop_w * 0.2)
            margin_y = int(crop_h * 0.2)
            max_offset_x = (crop_w // 2) - margin_x
            max_offset_y = (crop_h // 2) - margin_y
            offset_x = rng.randint(-max_offset_x, max_offset_x + 1)
            offset_y = rng.randint(-max_offset_y, max_offset_y + 1)
        else:
            offset_x, offset_y = 0, 0

        x_min_ideal = int(gt_x - crop_w / 2) + offset_x
        y_min_ideal = int(gt_y - crop_h / 2) + offset_y
        x_min = max(0, min(original_w - crop_w, x_min_ideal))
        y_min = max(0, min(original_h - crop_h, y_min_ideal))
        x_max = x_min + crop_w
        y_max = y_min + crop_h

        cropped_np = [f[y_min:y_max, x_min:x_max] for f in raw_frames]
        cropped_tensors = [self.to_tensor(f) for f in cropped_np]

        if self.transform:
            transformed = [self.transform(f) for f in cropped_tensors]
        else:
            transformed = list(cropped_tensors)

        x = gt_x - x_min
        y = gt_y - y_min
        dir_x = row.get('direction_x', 1.0)
        dir_y = row.get('direction_y', 0.0)

        n_loaded = len(transformed)
        sampled_idxs = self._sample_frame_indices(n_loaded, self.window_size)
        sampled = [transformed[i] for i in sampled_idxs]
        sampling_method = 'linspace' if n_loaded >= self.window_size else 'repeat_pad'

        frames_tensor = torch.stack(sampled).permute(1, 0, 2, 3)

        H, W = self.height, self.width
        target_tensor = torch.zeros(
            self.grid_size, self.grid_size,
            self.max_detections_per_cell, 5 + self.window_size,
            dtype=torch.float32
        )
        target_info = {'objectness': 0.0}

        if label == 1:
            x_norm, y_norm = x / W, y / H
            ws_in = row['waggle_start_in_window']
            we_in = row['waggle_end_in_window']

            dir_norm = np.sqrt(dir_x**2 + dir_y**2)
            if dir_norm > 0:
                dir_x_n, dir_y_n = dir_x / dir_norm, dir_y / dir_norm
            else:
                dir_x_n, dir_y_n = 1.0, 0.0

            grid_x = max(0, min(self.grid_size - 1, int(x_norm * self.grid_size)))
            grid_y = max(0, min(self.grid_size - 1, int(y_norm * self.grid_size)))
            cell_x = (x_norm * self.grid_size) - grid_x
            cell_y = (y_norm * self.grid_size) - grid_y

            target_tensor[grid_y, grid_x, 0, 0] = 1.0
            target_tensor[grid_y, grid_x, 0, 1] = cell_x
            target_tensor[grid_y, grid_x, 0, 2] = cell_y
            target_tensor[grid_y, grid_x, 0, 3] = dir_x_n
            target_tensor[grid_y, grid_x, 0, 4] = dir_y_n

            frame_labels = [0] * self.window_size
            if ws_in != -1 and we_in != -1:
                for t, fidx in enumerate(sampled_idxs):
                    real_frame = start_frame + fidx
                    if ws_in <= real_frame <= we_in:
                        target_tensor[grid_y, grid_x, 0, 5 + t] = 1.0
                        frame_labels[t] = 1

            target_info = {
                'objectness': 1.0,
                'grid_cell': [grid_y, grid_x],
                'cell_coords': [round(cell_x, 4), round(cell_y, 4)],
                'position_norm': [round(x_norm, 4), round(y_norm, 4)],
                'direction': [round(float(dir_x_n), 4), round(float(dir_y_n), 4)],
                'temporal_frame_labels': frame_labels,
                'waggle_start_in_window': int(ws_in),
                'waggle_end_in_window': int(we_in),
            }

        return {
            'raw_frames': raw_frames,
            'crop_params': {
                'x_min': x_min, 'y_min': y_min,
                'x_max': x_max, 'y_max': y_max,
                'offset_x': offset_x, 'offset_y': offset_y,
                'crop_mode': crop_mode,
            },
            'cropped_np': cropped_np,
            'transformed': transformed,
            'sampled': sampled,
            'sampled_indices': sampled_idxs,
            'sampling_method': sampling_method,
            'frames_tensor': frames_tensor,
            'target_tensor': target_tensor,
            'target_info': target_info,
            'metadata': {
                'video_name': video_name,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'resolution': [int(original_w), int(original_h)],
                'crop_size': [crop_w, crop_h],
                'window_size': self.window_size,
                'n_frames_loaded': n_loaded,
                'gt_position': [float(gt_x), float(gt_y)],
                'gt_direction': [float(row.get('direction_x', 1.0)),
                                 float(row.get('direction_y', 0.0))],
                'bee_in_crop': [round(float(x), 1), round(float(y), 1)],
                'is_waggle': int(label),
                'run_id': int(row.get('waggle_run_id', -1)),
            },
        }
