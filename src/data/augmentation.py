import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F_torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
cv2.setNumThreads(4)

class WaggleAugmentations:
    """
        Data augmentation pipeline for waggle-dance video clips with synchronized
        updates to both frames and ground-truth annotations.
        
        Includes spatial, color, and texture augmentations:
        - Spatial: Horizontal/Vertical flips, rotation (-45 to 45 degrees), scaling, translation
        - Color: HSV adjustments, color shuffle, posterize, greyscale
        - Texture/Quality: Blur, CLAHE
        - Normalization: Applied as final step (after all augmentations)

        Parameters
        ----------
        width : int, default=224
            Expected input frame width.
        height : int, default=224
            Expected input frame height.
        prob_flip_h : float, default=0.3
            Probability of applying horizontal flip.
        prob_flip_v : float, default=0.3
            Probability of applying vertical flip.
        prob_rotate : float, default=0.3
            Probability of applying rotation.
        rotate_range : tuple(float, float), default=(-45, 45)
            Min/max rotation angles in degrees.
        prob_scale : float, default=0.3
            Probability of applying scale augmentation.
        scale_range : tuple(float, float), default=(0.9, 1.1)
            Min/max scaling factors for the zoom augmentation.
        prob_translate : float, default=0.3
            Probability of applying translation augmentation.
        translate_range : float, default=0.1
            Maximum translation as fraction of image dimensions (0.1 = ±10%).
        prob_hsv : float, default=0.3
            Probability of applying HSV augmentation.
        hsv_hue : float, default=0.1
            Hue adjustment range (fraction of 360 degrees).
        hsv_saturation : float, default=0.2
            Saturation adjustment range (multiplier).
        hsv_value : float, default=0.2
            Value adjustment range (multiplier).
        prob_blur : float, default=0.3
            Probability of applying blur augmentation.
        blur_range : tuple(float, float), default=(0.5, 2.0)
            Min/max blur radius.
        prob_clahe : float, default=0.3
            Probability of applying CLAHE augmentation.
        clahe_clip_limit : float, default=2.0
            CLAHE clip limit for contrast limiting.
        clahe_tile_grid_size : tuple(int, int), default=(8, 8)
            CLAHE tile grid size.
        prob_color_shuffle : float, default=0.3
            Probability of applying color channel shuffle.
        prob_posterize : float, default=0.3
            Probability of applying posterize augmentation.
        posterize_bits : tuple(int, int), default=(4, 7)
            Range of bits to keep for posterization.
        prob_greyscale : float, default=0.3
            Probability of applying greyscale augmentation.
        normalize : bool, default=True
            If True, applies normalization as final step.
        mean : list[float], default=[0.485, 0.456, 0.406]
            Mean values for normalization (ImageNet defaults).
        std : list[float], default=[0.229, 0.224, 0.225]
            Standard deviation values for normalization (ImageNet defaults).
        debug : bool, default=False
            If True, prints applied augmentation info and updated coordinates.
    """
    def __init__(self, width=224, height=224, 
                 prob_flip_h=0.5, prob_flip_v=0.0, 
                 prob_rotate=0.3, rotate_range=(-45, 45), 
                 prob_scale=1.0, scale_range=(0.9, 1.1),
                 prob_translate=0.3, translate_range=0.1,
                 prob_hsv=0.0, hsv_hue=0.1, hsv_saturation=0.9, hsv_value=0.9,
                 prob_brightness=1.0, brightness_range=0.4, 
                 prob_contrast=1.0, contrast_range=0.4,
                 prob_gamma=0.0, gamma_range=(0.8, 1.2),
                 prob_blur=0.1, blur_range=(0.5, 2.0),
                 prob_clahe=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                 prob_color_shuffle=0.0,
                 prob_posterize=0.0, posterize_bits=(4, 7),
                 prob_greyscale=0.0,
                 normalize=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 debug=False):
        
        self.width = width
        self.height = height
        
        # Spatial augmentations
        self.prob_flip_h = prob_flip_h
        self.prob_flip_v = prob_flip_v
        self.prob_rotate = prob_rotate
        self.rotate_range = rotate_range
        self.prob_scale = prob_scale
        self.scale_range = scale_range
        self.prob_translate = prob_translate
        self.translate_range = translate_range
        
        # Color augmentations
        self.prob_hsv = prob_hsv
        self.hsv_hue = hsv_hue
        self.hsv_saturation = hsv_saturation
        self.hsv_value = hsv_value
        self.prob_color_shuffle = prob_color_shuffle
        self.prob_posterize = prob_posterize
        self.posterize_bits = posterize_bits
        self.prob_greyscale = prob_greyscale

        # Greyscaled img augmentations
        self.prob_brightness = prob_brightness
        self.brightness_range = brightness_range
        self.prob_contrast = prob_contrast
        self.contrast_range = contrast_range
        self.prob_gamma = prob_gamma
        self.gamma_range = gamma_range
        
        # Texture/Quality augmentations
        self.prob_blur = prob_blur
        self.blur_range = blur_range
        self.prob_clahe = prob_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        
        # Normalization
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
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

    def _apply_translation(self, frames, tx, ty):
        """Apply translation to frames."""
        translated_frames = []
        
        for frame in frames:
            # Get frame dimensions
            _, H, W = frame.shape
            
            # Calculate padding needed (positive = pad, negative = crop)
            pad_left = max(0, -tx)
            pad_right = max(0, tx)
            pad_top = max(0, -ty)
            pad_bottom = max(0, ty)
            
            # Pad with reflection
            padded = self._pad_with_reflection(
                frame, 
                left=pad_left, right=pad_right, 
                top=pad_top, bottom=pad_bottom
            )
            
            # Calculate crop region
            start_x = pad_left + max(0, tx)
            start_y = pad_top + max(0, ty)
            end_x = start_x + W
            end_y = start_y + H
            
            # Crop to original size
            translated = padded[:, start_y:end_y, start_x:end_x]
            
            # Ensure correct size (in case of rounding errors)
            if translated.shape[1] != H or translated.shape[2] != W:
                translated = F.resize(translated, (H, W))
            
            translated_frames.append(translated)
        
        return translated_frames

    def _apply_hsv_augmentation(self, frames):
        """Apply HSV color space augmentations."""
        augmented_frames = []
        
        # Sample augmentation parameters once for all frames
        hue_shift = random.uniform(-self.hsv_hue, self.hsv_hue) * 179
        sat_factor = random.uniform(1 - self.hsv_saturation, 1 + self.hsv_saturation)
        val_factor = random.uniform(1 - self.hsv_value, 1 + self.hsv_value)
        
        for frame in frames:
            # Convert to numpy array (0-255 uint8)
            if isinstance(frame, torch.Tensor):
                if frame.max() <= 1.0:
                    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    frame_np = frame.permute(1, 2, 0).numpy().astype(np.uint8)
            elif isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame
            
            # Convert RGB to HSV (OpenCV uses H: 0-179, S: 0-255, V: 0-255)
            hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply the SAME augmentation to all frames
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
            
            # Convert back to RGB
            hsv = hsv.astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Convert back to original format
            if isinstance(frame, torch.Tensor):
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
                if frame.max() <= 1.0:
                    rgb_tensor = rgb_tensor / 255.0
                augmented_frames.append(rgb_tensor)
            elif isinstance(frame, Image.Image):
                augmented_frames.append(Image.fromarray(rgb))
            else:
                augmented_frames.append(rgb)
        
        return augmented_frames

    def _apply_color_shuffle(self, frames):
        """Shuffle color channels."""
        augmented_frames = []
        
        # Random permutation of channel indices once for all frames
        channel_order = [0, 1, 2]
        random.shuffle(channel_order)
        
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                # Reorder channels
                shuffled = frame[channel_order, :, :]
                augmented_frames.append(shuffled)
            else:
                # For PIL Image, split and merge
                r, g, b = frame.split()
                channels = [r, g, b]
                shuffled_channels = [channels[i] for i in channel_order]
                shuffled = Image.merge('RGB', shuffled_channels)
                augmented_frames.append(shuffled)
        
        return augmented_frames


    def _apply_posterize(self, frames):
        """Apply posterization (reduce color depth)."""
        augmented_frames = []
        
        # Sample bits once for all frames 
        bits = random.randint(self.posterize_bits[0], self.posterize_bits[1])
        
        for frame in frames:
            # Convert to PIL Image
            if isinstance(frame, torch.Tensor):
                frame_pil = F.to_pil_image(frame)
            else:
                frame_pil = frame
            
            # Apply posterize
            posterized = ImageOps.posterize(frame_pil, bits)
            
            # Convert back to tensor if needed
            if isinstance(frame, torch.Tensor):
                posterized_tensor = F.to_tensor(posterized)
                augmented_frames.append(posterized_tensor)
            else:
                augmented_frames.append(posterized)
        
        return augmented_frames

    def _apply_blur_augmentation(self, frames):
        """Apply blur augmentation."""
        augmented_frames = []
        blur_radius = random.uniform(self.blur_range[0], self.blur_range[1])
        
        for frame in frames:
            # Convert to numpy for OpenCV blur
            if isinstance(frame, torch.Tensor):
                if frame.max() <= 1.0:
                    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    frame_np = frame.permute(1, 2, 0).numpy().astype(np.uint8)
            elif isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame
            
            # Apply Gaussian blur using OpenCV
            # Kernel size must be odd
            kernel_size = int(blur_radius * 2) * 2 + 1  # Ensure odd number
            if kernel_size < 1:
                kernel_size = 1
            
            blurred = cv2.GaussianBlur(frame_np, (kernel_size, kernel_size), blur_radius)
            
            # Convert back to original format
            if isinstance(frame, torch.Tensor):
                blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).float()
                if frame.max() <= 1.0:
                    blurred_tensor = blurred_tensor / 255.0
                augmented_frames.append(blurred_tensor)
            elif isinstance(frame, Image.Image):
                augmented_frames.append(Image.fromarray(blurred))
            else:
                augmented_frames.append(blurred)
        
        return augmented_frames

    def _apply_clahe_augmentation(self, frames):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        augmented_frames = []
        
        for frame in frames:
            # Convert to numpy array
            if isinstance(frame, torch.Tensor):
                if frame.max() <= 1.0:
                    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    frame_np = frame.permute(1, 2, 0).numpy().astype(np.uint8)
            else:
                frame_np = np.array(frame)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(frame_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, 
                                   tileGridSize=self.clahe_tile_grid_size)
            l_clahe = clahe.apply(l)
            
            # Merge back
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # Convert back to RGB
            rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            
            # Convert back to tensor
            if isinstance(frame, torch.Tensor):
                rgb_tensor = torch.from_numpy(rgb_clahe).permute(2, 0, 1).float()
                if frame.max() <= 1.0:
                    rgb_tensor = rgb_tensor / 255.0
                augmented_frames.append(rgb_tensor)
            else:
                augmented_frames.append(Image.fromarray(rgb_clahe))
        
        return augmented_frames


    def _apply_greyscale(self, frames):
        """Convert to greyscale."""
        augmented_frames = []
        
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                # Convert to greyscale using weighted sum
                # Using standard RGB to grayscale weights: 0.299R + 0.587G + 0.114B
                greyscale = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
                # Repeat across 3 channels for compatibility
                greyscale = greyscale.unsqueeze(0).repeat(3, 1, 1)
                augmented_frames.append(greyscale)
            else:
                greyscale = frame.convert('L').convert('RGB')
                augmented_frames.append(greyscale)
        
        return augmented_frames

    def _apply_normalization(self, frames):
        """
        Apply normalization as final step.
        Expects frames to be tensors in [0, 1] range.
        """
        normalized_frames = []
        
        for frame in frames:
            if not isinstance(frame, torch.Tensor):
                raise TypeError("Normalization requires tensor input")
            
            # Apply normalization: (x - mean) / std
            normalized = (frame - self.mean) / self.std
            normalized_frames.append(normalized)
        
        return normalized_frames
    
    def _apply_brightness_augmentation(self, frames):
        """Apply brightness adjustment."""
        augmented_frames = []
        
        # Sample brightness factor once for all frames
        brightness_factor = random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
        
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                # Clamp to [0, 1] range if normalized, or [0, 255] if not
                max_val = 1.0 if frame.max() <= 1.0 else 255.0
                brightened = torch.clamp(frame * brightness_factor, 0, max_val)
                augmented_frames.append(brightened)
            else:
                enhancer = ImageEnhance.Brightness(frame)
                augmented_frames.append(enhancer.enhance(brightness_factor))
        
        return augmented_frames

    def _apply_contrast_augmentation(self, frames):
        """Apply contrast adjustment."""
        augmented_frames = []
        
        # Sample contrast factor once for all frames
        contrast_factor = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                mean = frame.mean()
                contrasted = torch.clamp((frame - mean) * contrast_factor + mean, 0, 
                                        1.0 if frame.max() <= 1.0 else 255.0)
                augmented_frames.append(contrasted)
            else:
                enhancer = ImageEnhance.Contrast(frame)
                augmented_frames.append(enhancer.enhance(contrast_factor))
        
        return augmented_frames

    def _apply_gamma_augmentation(self, frames):
        """Apply gamma correction."""
        augmented_frames = []
        
        # Sample gamma once for all frames
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                # Normalize to [0, 1] for gamma
                if frame.max() > 1.0:
                    frame_normalized = frame / 255.0
                    gamma_corrected = torch.pow(frame_normalized, gamma) * 255.0
                else:
                    gamma_corrected = torch.pow(frame, gamma)
                augmented_frames.append(gamma_corrected)
            else:
                frame_np = np.array(frame).astype(np.float32) / 255.0
                gamma_corrected = np.power(frame_np, gamma)
                gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
                augmented_frames.append(Image.fromarray(gamma_corrected))
        
        return augmented_frames

    def __call__(self, frames, target):
        x = target["x"]
        y = target["y"]
        dir_x = target["dir_x"]
        dir_y = target["dir_y"]

        aug_info = []
        H, W = self.height, self.width

        # Geometric transforms (change pixel positions) 
        # Apply in logical order: Scale -> Translate -> Rotate -> Flips
        # Because they're not commutative, order matters
        
        #  Scale augmentation (zoom in/out) 
        # Should be first: affects all other geometric transformations
        if random.random() < self.prob_scale:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            aug_info.append(f"Scale_{scale:.2f}")

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

        #  Translation augmentation 
        # Should be second: translate after scaling
        if random.random() < self.prob_translate:
            # Calculate translation as fraction of image size
            tx = random.uniform(-self.translate_range, self.translate_range) * W
            ty = random.uniform(-self.translate_range, self.translate_range) * H
            
            # Apply translation to frames using OpenCV
            translation_matrix = np.array([
                [1, 0, tx],
                [0, 1, ty]
            ], dtype=np.float32)
            
            translated_frames = []
            for frame in frames:
                # Convert to numpy if needed
                if isinstance(frame, Image.Image):
                    frame_np = np.array(frame)
                else:  # tensor
                    frame_np = frame.permute(1, 2, 0).numpy()
                
                # Apply translation
                translated = cv2.warpAffine(frame_np, translation_matrix, (W, H))
                
                # Convert back to original format
                if isinstance(frame, Image.Image):
                    translated_frames.append(Image.fromarray(translated))
                else:
                    translated_frames.append(torch.from_numpy(translated).permute(2, 0, 1))
            
            frames = translated_frames
            
            # Update coordinates (simple addition for translation)
            x = x + tx
            y = y + ty
            
            # Clamp coordinates after translation
            x, y = self._clamp_coordinates(x, y, W, H)
            aug_info.append(f"Translate_tx{tx:.1f}_ty{ty:.1f}")

        #  Rotation 
        # Should be third: rotate after scaling and translation
        if random.random() < self.prob_rotate:
            angle = random.uniform(self.rotate_range[0], self.rotate_range[1])
            
            # Get rotation matrix from OpenCV
            cx, cy = W / 2.0, H / 2.0
            rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
            
            # Rotate frames using OpenCV
            rotated_frames = []
            for frame in frames:
                # Convert to numpy if needed (PIL -> numpy or tensor -> numpy)
                if isinstance(frame, Image.Image):
                    frame_np = np.array(frame)
                else:  # tensor
                    frame_np = frame.permute(1, 2, 0).numpy()
                
                # Rotate
                rotated = cv2.warpAffine(frame_np, rotation_matrix, (W, H))
                
                # Convert back to original format
                if isinstance(frame, Image.Image):
                    rotated_frames.append(Image.fromarray(rotated))
                else:
                    rotated_frames.append(torch.from_numpy(rotated).permute(2, 0, 1))
            
            frames = rotated_frames
            
            # Rotate point using the SAME rotation matrix
            point = np.array([x, y, 1.0])  # Homogeneous coordinates
            rotated_point = rotation_matrix @ point
            x, y = rotated_point[0], rotated_point[1]
            
            # Rotate direction vector (no translation, just rotation part)
            direction = np.array([dir_x, dir_y])
            rotation_only = rotation_matrix[:, :2]  # Extract 2x2 rotation part
            rotated_direction = rotation_only @ direction
            
            # Normalize direction
            norm = np.linalg.norm(rotated_direction)
            if norm > 0:
                rotated_direction /= norm
            dir_x, dir_y = rotated_direction[0], rotated_direction[1]
            
            # Clamp coordinates
            x, y = self._clamp_coordinates(x, y, W, H)
            aug_info.append(f"Rotate_{angle:.1f}deg")

        #  Horizontal flip 
        # Should be fourth: simple coordinate flip after other transformations
        if random.random() < self.prob_flip_h:
            frames = [F.hflip(f) for f in frames]
            x = W - 1 - x
            dir_x = -dir_x
            aug_info.append("Horizontal_Flip")

        #  Vertical flip 
        if random.random() < self.prob_flip_v:
            frames = [F.vflip(f) for f in frames]
            y = H - 1 - y
            dir_y = -dir_y
            aug_info.append("Vertical_Flip")

        # --- Photometric Transforms (color/texture only)
        # Apply after geometric transformations, before normalization
        
        #  HSV Augmentation 
        if random.random() < self.prob_hsv:
            frames = self._apply_hsv_augmentation(frames)
            aug_info.append("HSV_Adjustment")

        #  Blur Augmentation 
        if random.random() < self.prob_blur:
            frames = self._apply_blur_augmentation(frames)
            aug_info.append("Blur")

        #  CLAHE Augmentation 
        if random.random() < self.prob_clahe:
            frames = self._apply_clahe_augmentation(frames)
            aug_info.append("CLAHE")

        #  Color Shuffle 
        if random.random() < self.prob_color_shuffle:
            frames = self._apply_color_shuffle(frames)
            aug_info.append("Color_Shuffle")

        #  Posterize 
        if random.random() < self.prob_posterize:
            frames = self._apply_posterize(frames)
            aug_info.append("Posterize")

        #  Greyscale 
        if random.random() < self.prob_greyscale:
            frames = self._apply_greyscale(frames)
            aug_info.append("Greyscale")

        # Brightness Augmentation
        if random.random() < self.prob_brightness:
            frames = self._apply_brightness_augmentation(frames)
            aug_info.append("Brightness")

        # Contrast Augmentation
        if random.random() < self.prob_contrast:
            frames = self._apply_contrast_augmentation(frames)
            aug_info.append("Contrast")

        # Gamma Augmentation
        if random.random() < self.prob_gamma:
            frames = self._apply_gamma_augmentation(frames)
            aug_info.append("Gamma")

        # Normalize last 
        if self.normalize:
            frames = self._apply_normalization(frames)
            aug_info.append("Normalize")

        # Update target
        target["x"] = x
        target["y"] = y
        target["dir_x"] = dir_x
        target["dir_y"] = dir_y

        if self.debug:
            print(f"Augmentations: {', '.join(aug_info) if aug_info else 'none'}")
            print(f"Position: ({x:.2f}, {y:.2f}), direction: ({dir_x:.3f}, {dir_y:.3f})")

        return frames, target, aug_info