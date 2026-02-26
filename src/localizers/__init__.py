"""
Chest localization methods
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
from .utils import clip_to_mask_smart
import numpy as np
from typing import Deque

class BaseLocalizer(ABC):
    """
    Abstract base class for chest localization
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def locate(self, frame_buffer: Deque[np.ndarray], hand_mask, bird_mask) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate chest region within bird mask
        
        Args:
            frame_buffer: list of buffers
            hand_mask: Binary mask of hand
            bird_mask: Binary mask of bird
            **kwargs: Additional arguments
        
        Returns:
            chest_roi: (x, y, w, h) or None
        """
        pass

class CustomRobustLocalizer(BaseLocalizer):
    """
    Locate chest using pixel variance - chest has highest temporal variance
    Requires multiple frames to analyze motion
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Frame preprocessing
        self.smooth_kernel_size = config.get('smooth_kernel_size', 21)

        # Working buffers
        self.frame_buffer = []

        # Frame coordinates (set during locate())
        self.bx = None
        self.by = None
        self.bw = None
        self.bh = None

        # Hand mask (set during locate())
        self.hand_mask_resized = None

        print(f"✓ CustomRobustLocalizer initialized (smooth_kernel={self.smooth_kernel_size})")

    def locate(self, frame_buffer: Deque[np.ndarray], hand_mask, bird_mask):
        return self.locate_w_bird_mask(frame_buffer, hand_mask, bird_mask)
        
    def _preprocess_buffer(self):
        """
        Crop frames to reference hand mask and smooth with Gaussian filter
        """
        kernel = (self.smooth_kernel_size, self.smooth_kernel_size)

        assert self.bh is not None and self.bw is not None, "Reference shape not set"
        assert self.bx is not None and self.by is not None, "Reference bbox not set"
        
        for i in range(len(self.frame_buffer)):
            # 1. Grab the frame
            frame = self.frame_buffer[i]
            
            # 2. Process (Crop -> Gray -> Blur -> Float)
            try:
                processed = cv2.cvtColor(frame[self.by:self.by+self.bh, self.bx:self.bx+self.bw], cv2.COLOR_RGB2GRAY)
                processed = cv2.GaussianBlur(processed, kernel, 0)
            except Exception as e:
                raise Exception( f"Error processing frame {i} with shape {frame.shape}: {e}" )

            # 3. Overwrite the element in the list
            self.frame_buffer[i] = processed.astype(np.float32)

    def locate_w_bird_mask(self, frame_buffer: Deque[np.ndarray], hand_mask, bird_mask):
        """
        This method uses less post processing of the energy map since we rely on the quality of hand_mask
        Works entirely in FULL frame coordinates - no cropping/resizing

        :param frame_buffer: Buffer of frames (full size)
        :param hand_mask: Hand mask at full resolution
        :param bird_mask: Bird mask at full resolution (reliable, pre-processed)
        :return: ROI tuple (x, y, w, h) in FULL frame coordinates
        """
        # Locates using given MASK (consider it already pre-processed and reliable)

        # bird mask must be valid and reliable!
        assert bird_mask is not None
        assert np.sum(bird_mask) > 0

        # 0. Store masks and setup frame coordinates
        h_full, w_full = bird_mask.shape

        # Working in full frame coordinates (no cropping/offset)
        self.bx = 0
        self.by = 0
        self.bw = w_full
        self.bh = h_full
        self.hand_mask_resized = hand_mask

        # Preprocess frame buffer
        self.frame_buffer = list(frame_buffer)
        self._preprocess_buffer()

        # Convert bird mask to binary float
        bird_binary = (bird_mask.astype(np.uint8) > 0).astype(np.float32)

        # Initialize search boundaries to full frame
        core_left   = 0
        core_right  = w_full - 1
        core_top    = 0
        core_bottom = h_full - 1

        # Safety mask: inverse of bird mask (areas to avoid)
        safety_mask = 1 - bird_binary

        # Find largest rectangle within bird mask
        core_left, core_right, core_top, core_bottom = clip_to_mask_smart(
            safety_mask,
            core_left, core_right, core_top, core_bottom
        )

        # Convert to (x, y, w, h) format
        roi_w = core_right - core_left
        roi_h = core_bottom - core_top

        return (core_left, core_top, roi_w, roi_h)

def get_localizer(config: dict) -> BaseLocalizer:
    """
    Factory function to get localizer based on config

    Args:
        config: Configuration dictionary

    Returns:
        CustomRobustLocalizer instance
    """
    return CustomRobustLocalizer(config)
