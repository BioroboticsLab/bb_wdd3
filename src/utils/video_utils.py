import cv2
import torch
import gc
import torchvision.transforms as T
#from utils.nms import nms_spatiotemporal
from utils.frames_utils import preprocess_frames
import os 
import numpy as np 
from PIL import Image
cv2.setNumThreads(4)

def process_video_with_detections(model, video_path, device, window_size=16, stride=1, top_k=7, output_path="annotated_video.mp4", confidence=0.5):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    model.eval()
    frame_idx = 0

    all_detections = []
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

        x = preprocess_frames(window_frames, transform, device, clip_len=window_size)

        with torch.no_grad():
            model_output = model(x)

            detections = model.extract_detections(
                model_output,
                window_frames=(frame_idx, frame_idx + len(window_frames)-1),
                window_size=window_size,
                frame_idx=frame_idx,
                confidence_threshold=confidence,
                original_size=(width, height)
            )

            detections_nms = nms_spatiotemporal(detections)

            all_detections.extend(detections_nms)

        frame_disp = window_frames[0].copy()
        for det in detections_nms:
            pos_x, pos_y = det['position']
            dir_x, dir_y = det['direction']
            
            conf = det['confidence']
            start_frame, end_frame = det['temporal_offsets']

            cv2.circle(frame_disp, (int(pos_x), int(pos_y)), 8, (0, 255, 0), -1)
            arrow_length = 40
            end_x, end_y = int(pos_x + dir_x * arrow_length), int(pos_y + dir_y * arrow_length)
            cv2.arrowedLine(frame_disp, (int(pos_x), int(pos_y)), (int(end_x), int(end_y)), (0, 200, 0), 2)
            cv2.putText(frame_disp, f"P:{conf:.2f}", (int(pos_x) + 10, int(pos_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_disp, f"[{start_frame},{end_frame}]", (int(pos_x)+5, int(pos_y)+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        out.write(frame_disp)

        del model_output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        frame_idx += stride

    cap.release()
    out.release()
    print(f'Detections Preview: type {type(all_detections)} : {len(all_detections)}: {all_detections[:3]}')
    print(f"Annotated video saved to {output_path}")
    return all_detections

def visualize_preds_and_gt(video_path, preds_csv, gt_csv=None, output_path="vis_output.mp4"):
    #print('visualize_preds_and_gt - video_path:', video_path)
    #print(f'visualize_preds_and_gt - preds_csv : type {type(preds_csv)} head: {preds_csv.head()}')
    #print('visualize_preds_and_gt - gt_csv:', gt_csv.head())

    # --- Load video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Load preds ---
    preds_by_frame = {}
    for _, row in preds_csv.iterrows():
        for f in range(int(row.start), int(row.end) + 1):
            preds_by_frame.setdefault(f, []).append(row)

    # --- Load gt if given ---
    gts_by_frame = {}
    #if gt_csv:
    if gt_csv is not None and not gt_csv.empty:
      for _, row in gt_csv.iterrows():
          for f in range(int(row.start_frame), int(row.end_frame) + 1):
              gts_by_frame.setdefault(f, []).append(row)

    # --- Setup writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw preds (green + arrow for direction)
        if frame_idx in preds_by_frame:
            for det in preds_by_frame[frame_idx]:
                x, y = det.x, det.y
                x, y = int(float(x)), int(float(y))
                dx, dy = det.dir_x, det.dir_y
                dx, dy = float(dx), float(dy)

                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # green dot
                arrow_length = 40
                end_x, end_y = int(x + dx * arrow_length), int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 200, 0), 2)
                cv2.putText(frame, f"P:{det.confidence:.2f}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw GT (blue + arrow)
        if frame_idx in gts_by_frame:
            for gt in gts_by_frame[frame_idx]:
                x, y = int(gt.x1), int(gt.y1)
                dx, dy = float(gt.direction_x), float(gt.direction_y)

                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)  # blue dot
                arrow_length = 40
                end_x, end_y = int(x + dx * arrow_length), int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (200, 0, 0), 2)
                cv2.putText(frame, "GT", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Info overlay
        info_text = f"Frame {frame_idx}/{total_frames-1}"
        cv2.putText(frame, info_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Visualization saved to {output_path}")


def frames_to_video(frames_data, output_name, output_dir='./outputs/vids', fps=15):
    """
    Convert frames to video file - processes all batches into one video
    
    Args:
        frames_data: Can be either:
                    - Tensor of shape [batch, channels, time, height, width]
                    - List of batches, where each batch is a list of frames (PIL Images or arrays)
        output_name: Name for the output video file
        output_dir: Directory where to save the video (default: 'vids')
        fps: Frames per second for output video
    """
    
    # Handle both tensor and list inputs
    if isinstance(frames_data, list):
        # Input is a list of batches (each batch is a list of frames)
        all_frames = []
        for batch in frames_data:
            for frame in batch:
                # Convert PIL Image to numpy array if needed
                if isinstance(frame, Image.Image):
                    # Convert PIL Image to numpy array
                    frame = np.array(frame)
                    # Convert RGBA to RGB if needed
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    # Ensure it's RGB (PIL uses RGB by default)
                    elif frame.shape[-1] == 3:
                        # Already RGB, but OpenCV needs BGR later
                        pass
                # Handle tensor inputs within the list
                elif hasattr(frame, 'cpu'):
                    frame = frame.cpu().numpy()
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    if frame.shape[0] == 3:  # [3, H, W] format
                        frame = frame.transpose(1, 2, 0)
                
                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                all_frames.append(frame)
    else:
        # Input is a tensor [batch, channels, time, height, width]
        batch_size = frames_data.shape[0]
        
        # Process all batches and concatenate their frames
        all_frames = []
        for i in range(batch_size):
            # Process each batch: [3, 16, 224, 224] -> [16, 224, 224, 3]
            frames = frames_data[i].permute(1, 2, 3, 0)  # [16, 224, 224, 3]
            frames = (frames * 255).byte().cpu().numpy()  # Convert to uint8
            all_frames.extend([frame for frame in frames])
    
    # Get video dimensions from first frame
    if len(all_frames) == 0:
        raise ValueError("No frames to process")
    
    height, width = all_frames[0].shape[:2]
    
    # Create output path in the specified folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Add .mp4 extension if not already present
    if not output_name.endswith('.mp4'):
        output_path = os.path.join(output_dir, f"{output_name}.mp4")
    else:
        output_path = os.path.join(output_dir, output_name)
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames
    for frame in all_frames:
        # Convert from RGB to BGR for OpenCV
        if frame.shape[2] == 3:  # RGB format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {output_path}")
    print(f"Total frames: {len(all_frames)}")

def get_video_category(base_video_name: str) -> str:
    """
    Categorize video based on naming convention.
    
    Categories:
        - "0": Videos starting with 0
        - "T1": Videos starting with T1
        - "T_other": Videos starting with T(other number)
        - "C": Videos starting with C
        - "other": Everything else
    
    Args:
        base_video_name: Base video name
        
    Returns:
        Category string
    """
    if base_video_name.startswith('0'):
        return "0"
    elif base_video_name.startswith('T'):
        return "T"
    elif base_video_name.startswith('C'):
        return "C"
    else:
        return "other"