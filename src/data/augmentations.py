import random
import torchvision.transforms.functional as F
import torch.nn.functional as F_torch

class WaggleAugmentations:
    """
        Data augmentation pipeline for waggle-dance video clips with synchronized
        updates to both frames and ground-truth annotations.

        This class applies a set of spatial augmentations to a sequence of frames,
        while consistently updating:
            • (x, y) target position  
            • direction vector (dir_x, dir_y)  
            • image size–dependent coordinate clamping  

        The augmentations include horizontal/vertical flips, 90/180/270-degree
        rotations, and scale (zoom in/out). All augmentations are designed to work
        on temporally stacked frame tensors.

        Two operation modes exist:
            1. **Mutual-exclusive mode** (`mutual_exclusive=True`):  
            exactly one augmentation is sampled from a predefined set and applied
            with probability `prob`. If not selected, no augmentation is applied.

            2. **Independent mode** (`mutual_exclusive=False`):  
            each augmentation has an independent chance `prob` of being applied.

        Coordinates and direction vectors are transformed analytically to preserve
        geometric correctness (e.g., flip inversion, rotation mapping, scale around
        image center). Scaling uses reflection padding when zooming out to avoid
        unnatural borders.

        Parameters
        ----------
        width : int, default=224
            Expected input frame width.
        height : int, default=224
            Expected input frame height.
        prob : float, default=0.3
            Probability of applying an augmentation (per event or per-group depending
            on `mutual_exclusive`).
        position_jitter : bool, default=True
            Whether to enable translation jitter (currently unused because dataset
            cropping handles shifts).
        max_jitter_ratio : float, default=0.08
            Maximum jitter expressed as a fraction of image size.
        scale_range : tuple(float, float), default=(0.9, 1.1)
            Min/max scaling factors for the zoom augmentation.
        mutual_exclusive : bool, default=True
            If True, selects a single augmentation from a predefined set.
            If False, evaluates each augmentation independently.
        debug : bool, default=False
            If True, prints applied augmentation info and updated coordinates.

        Returns
        -------
        frames : list[Tensor]
            Augmented frame sequence, each shaped (C, H, W).
        target : dict
            Updated target dictionary containing new position and direction.
        aug_info : list[str]
            Human-readable list of applied augmentations.
    """
    def __init__(self, width=224, height=224, prob=0.3,
                 position_jitter=True, max_jitter_ratio=0.08,
                 scale_range=(0.9, 1.1),
                 mutual_exclusive=True,  
                 debug=False):
        self.width = width
        self.height = height
        self.prob = prob
        self.position_jitter = position_jitter
        self.max_jitter_ratio = max_jitter_ratio
        self.scale_range = scale_range
        self.mutual_exclusive = mutual_exclusive
        self.debug = debug

    def _clamp_coordinates(self, x, y, W, H):
        x = float(max(0, min(W - 1, x)))
        y = float(max(0, min(H - 1, y)))
        return x, y

    def _pad_with_reflection(self, frame, left, right, top, bottom):
        """
        Pad frame using reflection padding (more natural than tiling)
        frame: (C, h, w)
        returns: (C, h+top+bottom, w+left+right)
        """
        padded = F_torch.pad(frame, (left, right, top, bottom), mode='reflect')
        return padded

    def __call__(self, frames, target):
        x = target["x"]
        y = target["y"]
        dir_x = target["dir_x"]
        dir_y = target["dir_y"]

        aug_info = []
        H, W = self.height, self.width

        ref_frame = frames[0].clone().detach()
        if ref_frame.ndim == 4:
            ref_frame = ref_frame[0]

        if self.mutual_exclusive:
            # Randomly choose ONE augmentation to apply
            aug_options = ['flip_h', 'flip_v', 'rotate', 'scale'] 
            if random.random() < self.prob:
                chosen_aug = random.choice(aug_options)
            else:
                chosen_aug = 'None'
        else:
            # Original behavior: each augmentation independently at self.prob
            chosen_aug = None

        # ----- Horizontal flip -----
        if (self.mutual_exclusive and chosen_aug == 'flip_h') or \
           (not self.mutual_exclusive and random.random() < self.prob):
            frames = [F.hflip(f) for f in frames]
            ref_frame = F.hflip(ref_frame)  
            x = W - 1 - x
            dir_x = -dir_x
            aug_info.append("Horizontal_Flip_Augmentation")

        # ----- Vertical flip -----
        if (self.mutual_exclusive and chosen_aug == 'flip_v') or \
           (not self.mutual_exclusive and random.random() < self.prob):
            frames = [F.vflip(f) for f in frames]
            ref_frame = F.vflip(ref_frame)  
            y = H - 1 - y
            dir_y = -dir_y
            aug_info.append("Vertical_Flip_Augmentation")

        # ----- Rotation (90/180/270 degrees) -----
        if (self.mutual_exclusive and chosen_aug == 'rotate') or \
           (not self.mutual_exclusive and random.random() < self.prob):
            angle = random.choice([90, 180, 270])
            frames = [F.rotate(f, angle) for f in frames]
            ref_frame = F.rotate(ref_frame, angle) 

            if angle == 90:
                x_new, y_new = y, W - x - 1
                dir_x_new, dir_y_new = dir_y, -dir_x
                x, y = x_new, y_new
                dir_x, dir_y = dir_x_new, dir_y_new
                H, W = W, H
                aug_info.append("Rotate_Augmentation_90")

            elif angle == 180:
                x = W - x - 1
                y = H - y - 1
                dir_x = -dir_x
                dir_y = -dir_y
                aug_info.append("Rotate_Augmentation_180")

            elif angle == 270:
                x_new, y_new = H - y - 1, x
                dir_x_new, dir_y_new = -dir_y, dir_x
                x, y = x_new, y_new
                dir_x, dir_y = dir_x_new, dir_y_new
                H, W = W, H
                aug_info.append("Rotate_Augmentation_270")

            x, y = self._clamp_coordinates(x, y, W, H)

        # ----- Scale augmentation (zoom in/out) -----
        if (self.mutual_exclusive and chosen_aug == 'scale') or \
            (not self.mutual_exclusive and random.random() < self.prob):
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            aug_info.append(f"Scale_Augmentation_{scale:.2f}")

            scaled_frames = []
            for f in frames:
                _, orig_H, orig_W = f.shape

                new_H = int(round(orig_H * scale))
                new_W = int(round(orig_W * scale))
                new_H = max(1, new_H)
                new_W = max(1, new_W)

                scaled = F.resize(f, (new_H, new_W))

                if scale < 1.0:
                    # Zooming out: pad to maintain size
                    diff_h = orig_H - new_H
                    diff_w = orig_W - new_W
                    top = diff_h // 2
                    bottom = diff_h - top
                    left = diff_w // 2
                    right = diff_w - left

                    f_final = self._pad_with_reflection(
                        scaled, left=left, right=right, top=top, bottom=bottom
                    )
                else:
                    # Zooming in: center crop to original size
                    f_final = F.center_crop(scaled, [orig_H, orig_W])

                # Ensure final shape
                if f_final.shape[1] != orig_H or f_final.shape[2] != orig_W:
                    f_final = F.resize(f_final, (orig_H, orig_W))

                scaled_frames.append(f_final)

            frames = scaled_frames

            # Update coordinates
            cx, cy = W / 2.0, H / 2.0
            
            if scale < 1.0:
                # Zooming out: coordinates move toward center
                x = cx + (x - cx) * scale
                y = cy + (y - cy) * scale
            else:
                # Zooming in with center crop: coordinates stay at same relative position
                x_scaled = x * scale
                y_scaled = y * scale
            
                # This is the difference in size divided by 2.
                offset_w = (W * scale - W) / 2.0
                offset_h = (H * scale - H) / 2.0
                
                x = x_scaled - offset_w
                y = y_scaled - offset_h
            
            x, y = self._clamp_coordinates(x, y, W, H)

        # Currently not in use because the cropping logic used in the dataset class handles randomising gt location within the crop
        # ----- Position jitter (translation) -----
        # if self.position_jitter and \
        #     ((self.mutual_exclusive and chosen_aug == 'jitter') or \
        #         (not self.mutual_exclusive and random.random() < self.prob)):
        #     max_jitter_x = int(W * self.max_jitter_ratio)
        #     max_jitter_y = int(H * self.max_jitter_ratio)

        #     jitter_x = random.randint(-max_jitter_x, max_jitter_x)
        #     jitter_y = random.randint(-max_jitter_y, max_jitter_y)

        #     aug_info.append(f"Position_Augmentation_jitter_x{jitter_x}_y{jitter_y}")

        #     padded_frames = []
        #     pad_left = max(0, -jitter_x)
        #     pad_right = max(0, jitter_x)
        #     pad_top = max(0, -jitter_y)
        #     pad_bottom = max(0, jitter_y)

        #     for f in frames:
        #         padded = self._pad_with_reflection(
        #             f, left=pad_left, right=pad_right, top=pad_top, bottom=pad_bottom
        #         )

        #         start_x = pad_left + max(0, jitter_x)
        #         start_y = pad_top + max(0, jitter_y)

        #         start_x = int(start_x)
        #         start_y = int(start_y)

        #         end_x = start_x + W
        #         end_y = start_y + H

        #         f_shifted = padded[:, start_y:end_y, start_x:end_x]

        #         if f_shifted.shape[1] != H or f_shifted.shape[2] != W:
        #             f_shifted = F.resize(f_shifted, (H, W))

        #         padded_frames.append(f_shifted)

        #     frames = padded_frames

        #     # Correct coordinate update: add the jitter, not subtract
        #     x = x + jitter_x
        #     y = y + jitter_y
        #     x, y = self._clamp_coordinates(x, y, W, H)

        # Update target
        target["x"] = x
        target["y"] = y
        target["dir_x"] = dir_x
        target["dir_y"] = dir_y

        if self.debug:
            print(f"Augmentations: {', '.join(aug_info) if aug_info else 'none'}")
            print(f"Position: ({x:.2f}, {y:.2f}), direction: ({dir_x:.3f}, {dir_y:.3f})")

        return frames, target, aug_info