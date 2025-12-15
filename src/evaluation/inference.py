import cv2
import numpy as np
import torch
from ..data import preprocess_frames
from .postprocessing import  extract_detections, nms_spatial, nms_spatiotemporal
import gc
import time
from tqdm import tqdm

def process_video_with_detections(model, video_path, device, window_size=16, stride=1, confidence=0.5, resize=False):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    import torchvision.transforms as T
    transform_list = [
        T.ToPILImage()
    ]

    if resize:
        transform_list.append(T.Resize((224, 224)))

    transform_list.extend([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transforms = T.Compose(transform_list)

    model.eval()
    frame_idx = 0

    # Number of windows = ceil((total_frames - window_size) / stride) + 1
    num_windows = max(1, (total_frames - window_size) // stride + 1)

    all_detections = []
    for _ in tqdm(range(num_windows), desc="Processing windows", ncols=100):

        if frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        window_frames = []

        for i in range(window_size):
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            window_frames.append(frame)

        if len(window_frames) == 0:
            break

        x = preprocess_frames(window_frames, transforms, device, clip_len=window_size)

        # print((width, height), (frame_idx, frame_idx + len(window_frames)))

        with torch.no_grad():
            model_output = model(x)

            detections = extract_detections(
                model_output,
                window_frames=(frame_idx, frame_idx + len(window_frames)),
                window_size=window_size,
                frame_idx=frame_idx,
                confidence_threshold=confidence,
                original_size=(width, height)
            )
            detections = nms_spatial(detections)
            all_detections.extend(detections)

        del model_output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        frame_idx += stride

    cap.release()
    all_detections = nms_spatiotemporal(all_detections)
    return all_detections, fps

def preprocess_batch_tiles(tile_windows, transform, device, clip_len=16):
    """OPTIMIZED: Preprocess multiple tiles into a single batch tensor"""
    batch_tensors = []
    
    for tile_frames in tile_windows:
        processed_frames = []
        for frame in tile_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb)
            processed_frames.append(frame_tensor)
        
        # Sample to clip_len
        if len(processed_frames) >= clip_len:
            indices = np.linspace(0, len(processed_frames) - 1, clip_len, dtype=int)
            sampled = [processed_frames[i] for i in indices]
        else:
            sampled = processed_frames * (clip_len // len(processed_frames) + 1)
            sampled = sampled[:clip_len]
        
        tile_tensor = torch.stack(sampled).permute(1, 0, 2, 3)
        batch_tensors.append(tile_tensor)
    
    return torch.stack(batch_tensors).to(device)

def process_video_with_detections_tiles(model, video_path, device, window_size=16, stride=4, confidence=0.5, tile_size=(224,224), tile_overlap=0.05, batch_size=16):
    """
    OPTIMIZED VERSION: Fast batched tile processing for single GPU
    
    Key optimizations:
    - Batch processing: Process multiple tiles in one forward pass
    - Increased default stride: 4 instead of 1
    - Reduced overlap: 0.05 instead of 0.1
    - No threading overhead
    - Direct GPU batching
    
    Args:
        batch_size: Number of tiles to process in one batch (higher = faster, more memory)
        stride: Process every Nth frame (higher = faster)
        tile_overlap: Overlap between tiles (lower = faster)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print(f"FAST TILE-BASED INFERENCE")
    print(f"{'='*60}")
    print(f"Video: {width}x{height}, {total_frames} frames @ {fps} FPS")
    print(f"Stride: {stride} (processing every {stride} frames)")
    print(f"Tile size: {tile_size}, overlap: {tile_overlap*100:.0f}%")
    print(f"Batch size: {batch_size} tiles per forward pass")

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    import torchvision.transforms as T
    transforms = T.Compose([
        T.ToPILImage(),
        # T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

    model.eval()
    frame_idx = 0

    all_detections = []

    tile_w, tile_h = tile_size
    step_x = int(tile_w * (1 - tile_overlap))
    step_y = int(tile_h * (1 - tile_overlap))

    # Pre-compute tile positions
    tile_positions = []
    for y in range(0, height - tile_h + 1, step_y):
        for x in range(0, width - tile_w + 1, step_x):
            tile_positions.append((x, y))
    
    print(f"Total tiles per frame: {len(tile_positions)}")
    print(f"Estimated batches: {(total_frames // stride) * len(tile_positions) // batch_size}")
    print(f"{'='*60}\n")

    processed_windows = 0
    start_time = time.time()

    with torch.no_grad():  # Global no_grad for efficiency
        while frame_idx < total_frames:

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            window_frames = []

            for i in range(window_size):
                ret, frame = cap.read()
                if not ret:
                    break
                window_frames.append(frame)

            if len(window_frames) == 0:
                break
            
            # Extract all tiles for this window
            tile_windows = []
            for x, y in tile_positions:
                cropped_window = [f[y:y+tile_h, x:x+tile_w] for f in window_frames]
                tile_windows.append(cropped_window)

            # Process tiles in BATCHES (KEY OPTIMIZATION)
            detections_window = []
            
            for batch_idx in range(0, len(tile_windows), batch_size):
                batch_tiles = tile_windows[batch_idx:batch_idx+batch_size]
                batch_coords = tile_positions[batch_idx:batch_idx+batch_size]
                
                # Preprocess entire batch at once
                batch_tensor = preprocess_batch_tiles(batch_tiles, transforms, device, window_size)
                
                # Single forward pass for entire batch
                batch_output = model(batch_tensor)
                # combined_results=[]
                # Extract detections for each tile in batch
                for i, (x, y) in enumerate(batch_coords):
                    tile_output = batch_output[i:i+1]  # safe slice
                    # combined_results.append((x, y, tile_output))
                    
                    detections = extract_detections(
                        tile_output,
                        window_frames=(frame_idx, frame_idx + len(window_frames)-1),
                        window_size=window_size,
                        frame_idx=frame_idx,
                        confidence_threshold=confidence,
                        original_size=(tile_h, tile_w)
                    )


                    # Adjust coordinates to full frame
                    for det in detections:
                        det["position"][0] += x
                        det["position"][1] += y

                    detections = nms_spatial(detections)

                    detections_window.extend(detections)

            detections_window = nms_spatiotemporal(detections_window)
            all_detections.extend(detections_window)

            del detections_window
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            frame_idx += stride
            processed_windows += 1
            
            # Progress reporting
            if processed_windows % 10 == 0:
                elapsed = time.time() - start_time
                fps_proc = processed_windows / elapsed
                eta = (total_frames / stride - processed_windows) / fps_proc
                print(f"Processed {processed_windows} windows | "
                      f"{frame_idx}/{total_frames} frames | "
                      f"{fps_proc:.1f} windows/sec | "
                      f"ETA: {eta/60:.1f} min")

    cap.release()
    # out.release()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing completed in {total_time/60:.2f} minutes")
    print(f"Average speed: {processed_windows/total_time:.2f} windows/sec")
    # print(f"Annotated video saved to {output_path}")
    print(f"{'='*60}\n")
    
    return all_detections, fps