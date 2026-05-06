import gym
import numpy as np
import cv2
from memory_diffusion_policy.real_world.video_recorder import VideoRecorder

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0
        self.text_overlay = None  # Optional text overlay for current frame (LSTM progression)
        self.step_based_overlay = None  # Optional step-based progression overlay

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            
            # Apply text overlays if provided
            if self.text_overlay is not None:
                frame = self._add_text_overlay(frame, self.text_overlay, position='top-left')
            
            if self.step_based_overlay is not None:
                frame = self._add_text_overlay(frame, self.step_based_overlay, position='top-right')
            
            self.video_recoder.write_frame(frame)
        return result
    
    def _add_text_overlay(self, frame: np.ndarray, text: str, position: str = 'top-left') -> np.ndarray:
        """
        Add text overlay to frame.
        
        Args:
            frame: RGB image (H, W, 3) uint8
            text: Text to display
            position: 'top-left' or 'top-right'
        
        Returns:
            Frame with text overlay
        """
        frame = frame.copy()  # Don't modify original
        
        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness)
        
        # Calculate position with tighter padding
        padding = 2
        if position == 'top-right':
            # Top-right corner
            x1 = frame.shape[1] - text_width - 2 * padding - 2
            x2 = frame.shape[1] - padding
            text_x = frame.shape[1] - text_width - padding - 1
        else:  # top-left
            # Top-left corner
            x1 = padding
            x2 = text_width + 2 * padding
            text_x = padding 
        
        # Draw black background rectangle
        cv2.rectangle(
            frame,
            (x1, padding),
            (x2, text_height + 2 * padding),
            bg_color,
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (text_x, text_height + padding + 1),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
