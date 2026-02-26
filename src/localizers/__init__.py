"""
Chest localization methods
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
from .utils import clip_to_mask_smart, plot_matrices
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

        # Get variance-specific settings
        self.debug = config.get('debug', False) # add tons of prints
        self.debug_plots = config.get('debug_plots', False)# Show intermediate plots
        self.smooth_kernel_size = config.get('smooth_kernel_size', 21)


        custom_localizer_config = config.get('custom_localizer', {})
        print("custom_localizer_config: ", custom_localizer_config)

        self.use_hull_mask = custom_localizer_config.get('use_hull_mask', False)
        self.priorize_energy = custom_localizer_config.get('priorize_energy', False) # Force to avoid hands even if loosing some energy
        self.buffer_size = custom_localizer_config.get('buffer_frames', 30)

        # Hand mask buffer size - smaller to avoid over-accumulating hand movement
        # Default to 1/3 of frame buffer to balance coverage vs. precision
        self.hand_mask_buffer_size = custom_localizer_config.get('hand_mask_buffer_frames', self.buffer_size // 3)

        # Get ROI size constraints
        roi_size = config.get('roi_size', {})
        self.min_width = roi_size.get('min_width', 30)
        self.min_height = roi_size.get('min_height', 30)

        # Get tracking start frame (wait for camera stabilization)
        tracking_config = config.get('tracking', {})
        self.start_frame = tracking_config.get('start_frame', 0)

        self.start_frame = config.get('start_frame', 200)


        self.frame_buffer = []
        self.hand_mask_buffer = []  # Buffer to accumulate hand masks over time

        self.frames_rgb = []

        # Get reference shape FROM HAND MASK
        self.by = None
        self.bx = None
        self.bw = None
        self.bh = None


        # Post segmentation refinement params
        post_segmentation_config = config.get('segmentation_enhancement', {})
        self.hsv_hue_min = post_segmentation_config.get('hsv_hue_min', 0)
        self.hsv_hue_max = post_segmentation_config.get('hsv_hue_max', 20)
        self.hsv_sat_min = post_segmentation_config.get('hsv_sat_min', 20)
        self.hsv_val_min = post_segmentation_config.get('hsv_val_min', 70)
        self.ycrcb_cr_min = post_segmentation_config.get('ycrcb_cr_min', 133)
        self.ycrcb_cr_max = post_segmentation_config.get('ycrcb_cr_max', 173)
        self.ycrcb_cb_min = post_segmentation_config.get('ycrcb_cb_min', 77)
        self.ycrcb_cb_max = post_segmentation_config.get('ycrcb_cb_max', 127)

        self.hand_mask_resized = None # Main hand mask to use accross localization
        self.hand_mask_tuned = None # Main hand mask -processed- in ORIGINAL SIZE


        self.reference_bbox = None  # Store first bbox to ensure consistent frame sizes
        self.reference_shape = None  # Store expected frame shape (h, w)
        self.frame_count = 0  # Track current frame number

        print(f"✓ CustomRobustLocalizer initialized (buffer={self.buffer_size}, "
              f"hand_mask_buffer={self.hand_mask_buffer_size}, "
              f"use_hull_mask={self.use_hull_mask}, "
              f"priorize_energy={self.priorize_energy}, "
              f"start_frame={self.start_frame})")

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

        # bird mask MUST BE VALID and reliable!!!!!!!!!
        assert bird_mask is not None
        assert np.sum(bird_mask) > 0


        if self.debug_plots:
            print("---- hand mask buffers: ", len(self.hand_mask_buffer))
            if len(self.hand_mask_buffer)>0:
                plot_matrices([
                    (self.hand_mask_buffer[-1].mask, "self.hand_mask_buffer[-1]"),
                ])

        # 0. Store masks at full size (no cropping/reshaping)
        h_full, w_full = bird_mask.shape

        # No offset - working in full frame coordinates
        self.bx = 0
        self.by = 0
        self.bw = w_full
        self.bh = h_full

        self.hand_mask_resized = hand_mask  # Keep full size
        self.hand_mask_tuned = hand_mask    # FULL SIZE
        self.bird_mask = bird_mask          # FULL SIZE

        self.frame_buffer = list(frame_buffer)

        # Preprocess frame buffer (frames are already full size)
        self._preprocess_buffer()

        bird_binary = (bird_mask.astype(np.uint8) > 0).astype(np.float32)

        # Initialize core boundaries to full frame
        frame_h, frame_w = self.hand_mask_resized.shape
        core_left   = 0
        core_right  = frame_w - 1
        core_top    = 0
        core_bottom = frame_h - 1

        # Safety mask: areas to avoid (inverse of bird mask)
        safety_mask = 1 - bird_binary

        # Find largest rectangle within bird mask that avoids hand regions
        core_left, core_right, core_top, core_bottom = clip_to_mask_smart(
            safety_mask,
            core_left, core_right, core_top, core_bottom
        )

       
        final_left = core_left
        final_right = core_right
        final_top = core_top
        final_bottom = core_bottom
        final_bw = final_right - final_left
        final_bh = final_bottom - final_top

        return (final_left, final_top, final_bw, final_bh)

def get_localizer(config: dict) -> BaseLocalizer:
    """
    Factory function to get localizer based on config

    Args:
        config: Configuration dictionary

    Returns:
        Localizer instance
    """
    method = config.get('method', 'simple')


    if method == 'custom':
        return CustomRobustLocalizer(config)
    else:
        print(f"⚠ Unknown localization method '{method}', defaulting to 'custom'")
        return CustomRobustLocalizer(config)
