"""
Video playback manager with pre-loaded frame buffering.

Pre-loads all video frames into memory for instant, smooth playback
without seeking or I/O during rendering.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
from contextlib import contextmanager
import sys


@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide FFmpeg warnings."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class VideoManager:
    """Manages video loading and playback with pre-loaded frames."""
    
    def __init__(self, videos_dir: str):
        """Initialize video manager.
        
        Args:
            videos_dir: Directory containing video files
        """
        self.videos_dir = videos_dir
        self._videos: Dict[str, Dict] = {}
        
        # Set up logging
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from logger import get_logger
            self.logger = get_logger("Video Manager")
        except ImportError:
            # Logger not available, use print fallback
            self.logger = None
    
    def load_video(self, action_name: str, video_filename: str, max_frames: Optional[int] = None) -> bool:
        """Load and pre-load all frames from a video file.
        
        Args:
            action_name: Name identifier for this video
            video_filename: Filename of video in videos_dir
            
        Returns:
            True if loaded successfully, False otherwise
        """
        video_path = os.path.join(self.videos_dir, video_filename)
        
        if not os.path.exists(video_path):
            if self.logger:
                self.logger.error(f"Video not found: {video_path}")
            else:
                print(f"  ✗ Video not found: {video_path}")
            return False
        
        try:
            with suppress_stderr():
                video_capture = cv2.VideoCapture(video_path)
            
            if not video_capture.isOpened():
                if self.logger:
                    self.logger.error(f"Could not open video: {video_path}")
                else:
                    print(f"  ✗ Could not open video: {video_path}")
                return False
            
            # Get video metadata
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count_meta = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count_meta / fps if fps > 0 else 0
            
            # Pre-load frames into memory (up to max_frames if specified)
            if self.logger:
                if max_frames:
                    self.logger.info(f"Pre-loading up to {max_frames} frames for '{action_name}'...")
                else:
                    self.logger.info(f"Pre-loading frames for '{action_name}'...")
            else:
                if max_frames:
                    print(f"    Pre-loading up to {max_frames} frames for '{action_name}'...")
                else:
                    print(f"    Pre-loading frames for '{action_name}'...")
            frames: List[np.ndarray] = []
            frame_idx = 0
            
            with suppress_stderr():
                while True:
                    # Stop if we've reached max_frames
                    if max_frames and frame_idx >= max_frames:
                        break
                    ret, frame = video_capture.read()
                    if not ret or frame is None:
                        break
                    # Convert BGR to RGB and store
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_idx += 1
                    if frame_idx % 30 == 0:
                        if self.logger:
                            self.logger.debug(f"Loaded {frame_idx} frames...")
                        else:
                            print(f"      Loaded {frame_idx} frames...")
            
            # Reset video capture
            with suppress_stderr():
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Store video data
            self._videos[action_name] = {
                'capture': video_capture,  # Keep for metadata
                'frames': frames,  # Pre-loaded frames array
                'fps': fps,
                'frame_count': len(frames),
                'duration': len(frames) / fps if fps > 0 else 0,
                'current_frame': 0,
                'last_elapsed': 0.0,
            }
            
            actual_duration = len(frames) / fps if fps > 0 else 0
            if self.logger:
                if max_frames:
                    self.logger.info(f"Pre-loaded {len(frames)} frames for '{action_name}' (limited to {max_frames}, {fps:.2f} fps, {actual_duration:.2f}s)")
                else:
                    self.logger.info(f"Pre-loaded {len(frames)} frames for '{action_name}' ({len(frames)} frames, {fps:.2f} fps, {actual_duration:.2f}s)")
            else:
                if max_frames:
                    print(f"      ✓ Pre-loaded {len(frames)} frames (limited to {max_frames})")
                    print(f"  ✓ Loaded '{action_name}': {video_filename} "
                          f"({len(frames)}/{max_frames} frames, {fps:.2f} fps, {actual_duration:.2f}s)")
                else:
                    print(f"      ✓ Pre-loaded {len(frames)} frames")
                    print(f"  ✓ Loaded '{action_name}': {video_filename} "
                          f"({len(frames)} frames, {fps:.2f} fps, {actual_duration:.2f}s)")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading video '{action_name}' ({video_filename}): {e}")
            else:
                print(f"  ✗ Error loading video '{action_name}' ({video_filename}): {e}")
            return False
    
    def load_videos(self, video_config: Dict[str, str], max_frames_config: Optional[Dict[str, int]] = None) -> None:
        """Load multiple videos from config.
        
        Args:
            video_config: Dict mapping action names to video filenames
            max_frames_config: Optional dict mapping action names to max frame limits
        """
        for action_name, video_filename in video_config.items():
            max_frames = None
            if max_frames_config and action_name in max_frames_config:
                max_frames = max_frames_config[action_name]
            self.load_video(action_name, video_filename, max_frames=max_frames)
    
    def get_frame(self, action_name: str, elapsed: float) -> Optional[np.ndarray]:
        """Get video frame at specified elapsed time.
        
        Args:
            action_name: Name of video action
            elapsed: Elapsed time since start (seconds)
            
        Returns:
            Video frame as numpy array (RGB), or None if not available
        """
        if action_name not in self._videos:
            return None
        
        video_info = self._videos[action_name]
        frames = video_info.get('frames')
        
        if frames is None or len(frames) == 0:
            return None
        
        fps = video_info['fps']
        frame_count = video_info['frame_count']
        duration = video_info['duration']
        
        # Calculate target frame - respect duration exactly (don't clamp to show last frame indefinitely)
        if duration > 0:
            # If elapsed exceeds duration, return None to indicate video has ended
            if elapsed > duration:
                return None
            video_time = elapsed  # Use elapsed directly, don't clamp
            target_frame_number = int(video_time * fps)
            target_frame_number = min(target_frame_number, frame_count - 1)
            target_frame_number = max(0, target_frame_number)
        else:
            target_frame_number = 0
        
        # Reset if elapsed went backwards (new status started)
        if elapsed < video_info['last_elapsed']:
            video_info['current_frame'] = 0
            video_info['last_elapsed'] = elapsed
            target_frame_number = 0
        
        # Direct frame access (instant, no I/O)
        video_info['current_frame'] = target_frame_number
        video_info['last_elapsed'] = elapsed
        
        return frames[target_frame_number]
    
    def get_duration(self, action_name: str) -> float:
        """Get video duration.
        
        Args:
            action_name: Name of video action
            
        Returns:
            Duration in seconds, or 0.0 if not found
        """
        if action_name in self._videos:
            return self._videos[action_name]['duration']
        return 0.0
    
    def has_video(self, action_name: str) -> bool:
        """Check if video is loaded.
        
        Args:
            action_name: Name of video action
            
        Returns:
            True if video is loaded
        """
        return action_name in self._videos
    
    def cleanup(self) -> None:
        """Release all video resources."""
        for action_name, video_info in self._videos.items():
            capture = video_info.get('capture')
            if capture is not None:
                capture.release()
            if self.logger:
                self.logger.info(f"Released video: {action_name}")
            else:
                print(f"Released video: {action_name}")
        self._videos.clear()

