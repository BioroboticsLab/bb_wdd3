import cv2
import numpy as np
from typing import List, Optional


class VideoFrameCache:
    """
    Efficient video frame loader that caches video captures (not frames)
    and decodes on-demand with optimized frame seeking.
    
    The cache uses an LRU (Least Recently Used) eviction policy to manage
    memory while maintaining fast access to frequently used videos.
    
    Args:
        cache_size: Maximum number of VideoCapture objects to keep in memory
        
    """
    
    def __init__(self, cache_size: int = 10):
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
    
    def _get_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """
        Get or create a cv2.VideoCapture object with LRU caching.
        
        Args:
            video_path: Path to video file
            
        Returns:
            cv2.VideoCapture object
            
        Raises:
            FileNotFoundError: If video cannot be opened
        """
        if video_path in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(video_path)
            self._access_order.append(video_path)
            return self._cache[video_path]
        
        # Create new capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        # Add to cache
        self._cache[video_path] = cap
        self._access_order.append(video_path)
        
        # Evict oldest if cache is full
        if len(self._cache) > self.cache_size:
            oldest = self._access_order.pop(0)
            self._cache[oldest].release()
            del self._cache[oldest]
        
        return cap
    
    def load_frame_range(
        self, 
        video_path: str, 
        start_frame: int, 
        end_frame: int,
        convert_rgb: bool = True
    ) -> List[np.ndarray]:
        """
        Load specific frame range efficiently.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive)
            convert_rgb: Whether to convert BGR to RGB
            
        Returns:
            List of frames as numpy arrays
            
        """
        cap = self._get_video_capture(video_path)
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return frames
    
    def load_full_video(
        self, 
        video_path: str,
        convert_rgb: bool = True
    ) -> List[np.ndarray]:
        """
        Load all frames from a video.
        
        Args:
            video_path: Path to video file
            convert_rgb: Whether to convert BGR to RGB
            
        Returns:
            List of all frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata without loading frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with 'fps', 'frame_count', 'width', 'height'
        """
        cap = self._get_video_capture(video_path)
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        return info
    
    def clear_cache(self):
        """Release all cached video captures and clear the cache."""
        for cap in self._cache.values():
            cap.release()
        self._cache.clear()
        self._access_order.clear()
    
    def __del__(self):
        """Cleanup all video captures on deletion."""
        self.clear_cache()


# Global instance for easy access
_global_video_cache = VideoFrameCache(cache_size=10)


def load_video_frames(
    video_path: str, 
    start_frame: Optional[int] = None, 
    end_frame: Optional[int] = None,
    convert_rgb: bool = True,
    use_cache: bool = True
) -> List[np.ndarray]:
    """
    Load video frames efficiently with optional caching.
    
    This is a convenience function that uses the global VideoFrameCache
    for efficient frame loading.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index (optional, inclusive)
        end_frame: Ending frame index (optional, exclusive)
        convert_rgb: Whether to convert BGR to RGB
        use_cache: Whether to use the global cache
    
    Returns:
        List of frames as numpy arrays

    """
    if use_cache:
        cache = _global_video_cache
    else:
        cache = VideoFrameCache(cache_size=1)
    
    # Load specific frame range
    if start_frame is not None and end_frame is not None:
        return cache.load_frame_range(video_path, start_frame, end_frame, convert_rgb)
    
    # Load all frames
    frames = cache.load_full_video(video_path, convert_rgb)
    
    # Apply slicing if only one bound is specified
    if start_frame is not None:
        frames = frames[start_frame:]
    if end_frame is not None:
        frames = frames[:end_frame]
    
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with 'fps', 'frame_count', 'width', 'height'
    """
    return _global_video_cache.get_video_info(video_path)