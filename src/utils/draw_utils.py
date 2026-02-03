import numpy as np
import torch
import cv2
import os 

def draw_waggle(frames, detections, ground_truths, start_frame_idx, output_dir, create_mp4=True):
    """
    Draw waggle detections and ground truth on frames using OpenCV Only single waggle
    GT: Red filled circle
    Predictions: Green filled circle
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PIL frames to OpenCV format (BGR)
    cv_frames = []
    for pil_frame in frames:
        cv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        cv_frames.append(cv_frame)
    
    # Group predictions by frame
    preds_by_frame = {}
    for detection in detections:
        start_offset, end_offset = detection['temporal_offsets']
        for frame_offset in range(start_offset, end_offset + 1):
            local_frame_idx = frame_offset - start_frame_idx
            if 0 <= local_frame_idx < len(cv_frames):
                if local_frame_idx not in preds_by_frame:
                    preds_by_frame[local_frame_idx] = []
                detection['type'] = 'prediction'
                preds_by_frame[local_frame_idx].append(detection)
    
    # Group ground truth by frame
    gt_by_frame = {}
    for gt in ground_truths:
        start_offset, end_offset = gt['temporal_offsets']
        for frame_offset in range(start_offset, end_offset + 1):
            local_frame_idx = frame_offset - start_frame_idx
            if 0 <= local_frame_idx < len(cv_frames):
                if local_frame_idx not in gt_by_frame:
                    gt_by_frame[local_frame_idx] = []
                gt['type'] = 'ground_truth'
                gt_by_frame[local_frame_idx].append(gt)
    
    # Process frames
    saved_frame_paths = []
    for local_frame_idx in range(len(cv_frames)):
        frame = cv_frames[local_frame_idx].copy()
        
        # Draw ground truth first (red filled)
        if local_frame_idx in gt_by_frame:
            for gt in gt_by_frame[local_frame_idx]:
                x, y = gt['position']
                x, y = int(x), int(y)
                dx, dy = gt['direction']
                
                # Red for ground truth - filled
                color = (0, 0, 255)  # Red in BGR
                
                # Draw filled circle (same size as predictions)
                cv2.circle(frame, (x, y), 6, color, -1)  # Radius 6, filled (-1)
                
                # Draw arrow (same thickness as predictions)
                arrow_length = 30
                end_x = int(x + dx * arrow_length)
                end_y = int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
                
                # Add "GT" text
                cv2.putText(frame, "GT", (x + 12, y - 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw predictions (green filled)
        if local_frame_idx in preds_by_frame:
            for detection in preds_by_frame[local_frame_idx]:
                x, y = detection['position']
                x, y = int(x), int(y)
                dx, dy = detection['direction']
                confidence = detection['confidence']
                
                # Green for predictions - filled
                color = (0, 255, 0)  # Green in BGR
                
                # Draw filled circle (same size as GT)
                cv2.circle(frame, (x, y), 6, color, -1)  # Radius 6, filled (-1)
                
                # Draw arrow (same thickness as GT)
                arrow_length = 25
                end_x = int(x + dx * arrow_length)
                end_y = int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
                
                # Add confidence text
                text = f"{confidence:.2f}"
                cv2.putText(frame, text, (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Count statistics
        num_gt = len(gt_by_frame.get(local_frame_idx, []))
        num_pred = len(preds_by_frame.get(local_frame_idx, []))
        
        # Add frame info text
        info_text = f"GT: {num_gt} | Pred: {num_pred}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save frame
        global_frame_idx = start_frame_idx + local_frame_idx
        filename = f"waggle_detection_{global_frame_idx:06d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_frame_paths.append(filepath)
        
        print(f"Frame {global_frame_idx}: {num_gt} GT, {num_pred} Pred")
    
    # Create MP4 video
    if create_mp4 and len(saved_frame_paths) > 0:
        # Get frame dimensions from first frame
        first_frame = cv2.imread(saved_frame_paths[0])
        height, width = first_frame.shape[:2]
        
        # Define video writer
        mp4_path = os.path.join(output_dir, "waggle_detection_animation.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5  # 5 frames per second
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        
        # Write all frames to video
        for filepath in saved_frame_paths:
            frame = cv2.imread(filepath)
            out.write(frame)
        
        out.release()
        print(f"Created MP4: {mp4_path}")
    
    return saved_frame_paths


def draw_waggle_batch(all_frames, all_detections, all_ground_truths, all_start_frame_idxs, file_name=None, output_dir='vids', fps=5):
    """
    Draw waggle detections and ground truth on frames for multiple batches and create video
    
    Args:
        all_frames: List of batches, where each batch is a list of frames
        all_detections: List of detections for each batch
        all_ground_truths: List of ground truths for each batch  
        all_start_frame_idxs: List of start frame indices for each batch
        file_name: None for default or string to i.e,. denote post processed data
        output_dir: Output directory for video
        fps: Frames per second for output video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video dimensions from first frame of first batch
    first_frame = cv2.cvtColor(np.array(all_frames[0][0]), cv2.COLOR_RGB2BGR)
    height, width = first_frame.shape[:2]
    
    # Define video writer
    if file_name is None:
        mp4_path = os.path.join(output_dir, "all_batches_waggle_detection.mp4")
    else:
        mp4_path = os.path.join(output_dir, f"all_batches_waggle_detection_{file_name}.mp4")


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # Process each batch
    for batch_idx, (frames, detections, ground_truths, start_frame_idx) in enumerate(zip(
        all_frames, all_detections, all_ground_truths, all_start_frame_idxs
    )):
        #print(f"Processing batch {batch_idx} with {len(frames)} frames...")
        
        # Convert PIL frames to OpenCV format (BGR)
        cv_frames = []
        for pil_frame in frames:
            cv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            cv_frames.append(cv_frame)
        
        # Group predictions by frame
        preds_by_frame = {}
        for detection in detections:
            start_offset, end_offset = detection['temporal_offsets']
            for frame_offset in range(start_offset, end_offset + 1):
                local_frame_idx = frame_offset - start_frame_idx
                if 0 <= local_frame_idx < len(cv_frames):
                    if local_frame_idx not in preds_by_frame:
                        preds_by_frame[local_frame_idx] = []
                    detection['type'] = 'prediction'
                    preds_by_frame[local_frame_idx].append(detection)
        
        # Group ground truth by frame
        gt_by_frame = {}
        for gt in ground_truths:
            start_offset, end_offset = gt['temporal_offsets']
            for frame_offset in range(start_offset, end_offset + 1):
                local_frame_idx = frame_offset - start_frame_idx
                if 0 <= local_frame_idx < len(cv_frames):
                    if local_frame_idx not in gt_by_frame:
                        gt_by_frame[local_frame_idx] = []
                    gt['type'] = 'ground_truth'
                    gt_by_frame[local_frame_idx].append(gt)
        
        # Process frames for this batch and write directly to video
        for local_frame_idx in range(len(cv_frames)):
            frame = cv_frames[local_frame_idx].copy()
            
            # Draw ground truth first (red filled)
            if local_frame_idx in gt_by_frame:
                for gt in gt_by_frame[local_frame_idx]:
                    x, y = gt['position']
                    x, y = int(x), int(y)
                    dx, dy = gt['direction']
                    
                    # Red for ground truth - filled
                    color = (0, 0, 255)  # Red in BGR
                    
                    # Draw filled circle (same size as predictions)
                    cv2.circle(frame, (x, y), 6, color, 2)  # Radius 6, filled (-1) or thickness i.e., 2
                    
                    # Draw arrow (same thickness as predictions)
                    arrow_length = 30
                    end_x = int(x + dx * arrow_length)
                    end_y = int(y + dy * arrow_length)
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
                    
                    # Add "GT" text
                    cv2.putText(frame, "GT", (x + 12, y - 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw predictions (green filled)
            if local_frame_idx in preds_by_frame:
                for detection in preds_by_frame[local_frame_idx]:
                    x, y = detection['position']
                    x, y = int(x), int(y)
                    dx, dy = detection['direction']
                    confidence = detection['confidence']
                    
                    # Green for predictions - filled
                    color = (0, 255, 0)  # Green in BGR
                    
                    # Draw filled circle (same size as GT)
                    cv2.circle(frame, (x, y), 6, color, 2)  # Radius 6, filled (-1)
                    
                    # Draw arrow (same thickness as GT)
                    arrow_length = 25
                    end_x = int(x + dx * arrow_length)
                    end_y = int(y + dy * arrow_length)
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
                    
                    # Add confidence text
                    text = f"{confidence:.2f}"
                    cv2.putText(frame, text, (x + 8, y - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Count statistics
            num_gt = len(gt_by_frame.get(local_frame_idx, []))
            num_pred = len(preds_by_frame.get(local_frame_idx, []))
            
            # Add frame info text
            info_text = f"GT: {num_gt} | Pred: {num_pred}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add global frame info
            global_frame_idx = start_frame_idx + local_frame_idx
            frame_text = f"Global Frame: {global_frame_idx}"
            cv2.putText(frame, frame_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame directly to video
            out.write(frame)
            frame_count += 1
            
            #print(f"Batch {batch_idx}, Frame {global_frame_idx}: {num_gt} GT, {num_pred} Pred")
    
    out.release()
    print(f"Created MP4 with {frame_count} frames: {mp4_path}")
    
    return mp4_path


def draw_waggle_batch_union(all_frames, all_detections, all_ground_truths, all_start_frame_idxs, file_name=None, output_dir='vids', fps=5):
    """
    Draw waggle detections and ground truth on frames for multiple batches using Union of All Predictions
    
    Args:
        all_frames: List of batches, where each batch is a list of frames
        all_detections: List of detections for each batch
        all_ground_truths: List of ground truths for each batch  
        all_start_frame_idxs: List of start frame indices for each batch
        file_name: None for default or string to i.e,. denote post processed data
        output_dir: Output directory for video
        fps: Frames per second for output video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Build mapping of all global frames and their sources
    global_frame_sources = {}  # global_idx -> list of (batch_idx, local_idx)
    global_frame_images = {}   # global_idx -> OpenCV frame (we'll use the first occurrence)
    
    for batch_idx, (frames, start_idx) in enumerate(zip(all_frames, all_start_frame_idxs)):
        for local_idx in range(len(frames)):
            global_idx = start_idx + local_idx
            
            # Store the frame source
            if global_idx not in global_frame_sources:
                global_frame_sources[global_idx] = []
            global_frame_sources[global_idx].append((batch_idx, local_idx))
            
            # Store the frame image (use first occurrence)
            if global_idx not in global_frame_images:
                pil_frame = frames[local_idx]
                cv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                global_frame_images[global_idx] = cv_frame
    
    # Step 2: Aggregate ALL predictions and ground truths per global frame
    global_predictions = {}
    global_ground_truths = {}
    
    # Aggregate predictions from ALL batches
    for batch_idx, (detections, start_idx) in enumerate(zip(all_detections, all_start_frame_idxs)):
        for detection in detections:
            start_offset, end_offset = detection['temporal_offsets']
            for frame_offset in range(start_offset, end_offset + 1):
                global_idx = frame_offset #start_idx + (frame_offset - start_offset)
                if global_idx not in global_predictions:
                    global_predictions[global_idx] = []
                # Add batch info to detection for tracking
                detection_with_batch = detection.copy()
                detection_with_batch['batch_idx'] = batch_idx
                global_predictions[global_idx].append(detection_with_batch)
    
    # Aggregate ground truths from ALL batches
    for batch_idx, (ground_truths, start_idx) in enumerate(zip(all_ground_truths, all_start_frame_idxs)):
        for gt in ground_truths:
            start_offset, end_offset = gt['temporal_offsets']
            for frame_offset in range(start_offset, end_offset + 1):
                global_idx = frame_offset #start_idx + (frame_offset - start_offset)
                if global_idx not in global_ground_truths:
                    global_ground_truths[global_idx] = []
                gt_with_batch = gt.copy()
                gt_with_batch['batch_idx'] = batch_idx
                global_ground_truths[global_idx].append(gt_with_batch)
    
    # Step 3: Get video dimensions and create video writer
    global_indices = sorted(global_frame_images.keys())
    if not global_indices:
        raise ValueError("No frames found!")
    
    first_frame = global_frame_images[global_indices[0]]
    height, width = first_frame.shape[:2]
    
    # Define video writer
    if file_name is None:
        mp4_path = os.path.join(output_dir, "union_waggle_detection.mp4")
    else:
        mp4_path = os.path.join(output_dir, f"union_waggle_detection_{file_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    
    # Step 4: Process each global frame in order
    for global_idx in global_indices:
        frame = global_frame_images[global_idx].copy()
        
        # DEDUPLICATE ground truths to remove duplicates from overlapping windows
        if global_idx in global_ground_truths:
            unique_gts = []
            seen_positions = set()
            
            for gt in global_ground_truths[global_idx]:
                # Create a position key (round to handle small variations)
                pos_key = (round(gt['position'][0]), round(gt['position'][1]))
                
                if pos_key not in seen_positions:
                    unique_gts.append(gt)
                    seen_positions.add(pos_key)
            
            global_ground_truths[global_idx] = unique_gts
        
        # Draw ALL ground truths from ALL batches for this global frame
        if global_idx in global_ground_truths:
            for gt in global_ground_truths[global_idx]:
                x, y = gt['position']
                x, y = int(x), int(y)
                dx, dy = gt['direction']
                
                # Red for ground truth - filled
                color = (0, 0, 255)  # Red in BGR
                
                # Draw filled circle
                cv2.circle(frame, (x, y), 6, color, -1)
                
                # Draw arrow
                arrow_length = 30
                end_x = int(x + dx * arrow_length)
                end_y = int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
                
                # Add "GT" text
                batch_info = f"GT"
                cv2.putText(frame, batch_info, (x + 12, y - 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ALL predictions from ALL batches for this global frame
        if global_idx in global_predictions:
            for detection in global_predictions[global_idx]:
                x, y = detection['position']
                x, y = int(x), int(y)
                dx, dy = detection['direction']
                confidence = detection['confidence']
                
                # Green for predictions - filled
                color = (0, 255, 0)  # Green in BGR
                
                # Draw filled circle
                cv2.circle(frame, (x, y), 6, color, -1)
                
                # Draw arrow
                arrow_length = 25
                end_x = int(x + dx * arrow_length)
                end_y = int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
                
                # Add confidence text
                text = f"{confidence:.2f}"
                cv2.putText(frame, text, (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Count statistics 
        num_gt = len(global_ground_truths.get(global_idx, []))
        num_pred = len(global_predictions.get(global_idx, []))
        num_batches = len(global_frame_sources.get(global_idx, []))
        
        # Add frame info text
        info_text = f"Frame {global_idx} | GT: {num_gt} | Pred: {num_pred}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        #print(f"Global Frame {global_idx}: {num_gt} GT, {num_pred} Pred")
    
    out.release()
    print(f"Created Union MP4 with {len(global_indices)} unique frames: {mp4_path}")
    
    return mp4_path