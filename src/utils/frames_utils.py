import numpy as np
import torch 
import cv2

def sample_frames(frames, target_length):
    num_frames = len(frames)

    if num_frames >= target_length:
        indices = np.linspace(0, num_frames - 1, target_length, dtype=int)
        return [frames[i] for i in indices]
    else:
        repeated_frames = frames * (target_length // num_frames + 1)
        return repeated_frames[:target_length]

def preprocess_frames(frames, transform, device, clip_len=16):
    all_frames = []
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        all_frames.append(frame)

    sampled_frames = sample_frames(all_frames, clip_len)
    frames_tensor = torch.stack(sampled_frames).permute(1, 0, 2, 3)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    del all_frames, sampled_frames
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return frames_tensor