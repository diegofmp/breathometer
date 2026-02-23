"""
Chest localization methods
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
from .utils import _find_energy_center, clip_to_mask_smart, get_breathing_energy_map, plot_matrices
import numpy as np
import matplotlib.pyplot as plt
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
        self.hard_constrain_mode = custom_localizer_config.get('hard_constrain_mode', "energy22") # What to focus on when shrinking ROI: energy or area
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
              f"hard_constrain_mode={self.hard_constrain_mode}, "
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

        # 1. Get Energy map (now at FULL resolution)
        divergence_energy, raw_divergence = get_breathing_energy_map(
            self.frame_buffer,
            self.hand_mask_resized,
            self.bird_mask,
            visualize=self.debug_plots
        )

        bird_binary = (bird_mask.astype(np.uint8) > 0).astype(np.float32)

        if self.debug_plots:
            plot_matrices(
                [
                    (divergence_energy, "divergence_energy"),
                    (raw_divergence, "raw_divergence"),
                    (raw_divergence*bird_binary, "raw_divergence cleaned")
                ], show_axis=True)

        # Use raw divergence energy
        energy_map = raw_divergence

        # Remove areas outside bird mask
        energy_map = energy_map * bird_binary

        if np.sum(energy_map) < 1:
            print("VERY LOW ENERGY MAP!!!: ", np.sum(energy_map) , " - ---> <1!")

        reliable_bird_mask = True

        # 2. Find energy mass center (now returns coordinates in FULL frame space)
        cx_full, cy_full, energy_cleaned_v2 = _find_energy_center(
            energy_map,
            self.hand_mask_resized,
            self.bird_mask,
            self.bw,
            self.bh,
            visualize=self.debug_plots,
            reliable_bird_mask=reliable_bird_mask
        )

        if self.debug:
            print("cx_full: ", cx_full, " (no offset needed)")
            print("cy_full: ", cy_full, " (no offset needed)")

        # CORE ##
        # ============================================================================
        # PHASE 1: Extract High-Energy Points
        # ============================================================================

        # Find top 5% energy points (now in FULL frame coordinates)
        percentile = 99
        threshold_val = np.percentile(energy_cleaned_v2[energy_cleaned_v2 > 0], percentile)
        y_high, x_high = np.where(energy_cleaned_v2 >= threshold_val)

        if self.debug_plots:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.imshow(energy_cleaned_v2, cmap='magma')
            ax.set_title(f'All Energy (energy_cleaned_v2) w percentile {percentile}')
            ax.scatter(cx_full, cy_full, c='cyan', s=300, marker='*', label='Center of Mass')
            ax.scatter(x_high, y_high, c='cyan', s=2, alpha=0.5, label='Valid points')
            ax.legend()
            fig.suptitle(f"PHASE 1 - Full Frame Coordinates")
            plt.tight_layout()
            plt.show()

        # ============================================================================
        # PHASE 2: Filter Points by Line-of-Sight from Center (SKIPPED)
        # ============================================================================
        
        # Kepp it simple. We assume mask is limited to real bird center, not head nor tail. 
        valid_x = x_high
        valid_y = y_high
            
        # ============================================================================
        # PHASE 3: Remove Statistical Outliers : TODO: does it make sense to keep this?
        # ============================================================================

        # Calculate distances from energy center
        dist_from_center = np.sqrt((valid_x - cx_full)**2 + (valid_y - cy_full)**2)
        median_dist = np.median(dist_from_center)
        std_dist = np.std(dist_from_center)

        # Keep only points within 1.5 std deviations from median
        is_not_outlier = dist_from_center < (median_dist + 1.5 * std_dist)

        # just for dbugging, show outliers
        invalid_x = valid_x[~is_not_outlier]
        invalid_y = valid_y[~is_not_outlier]
        ########

        valid_x = valid_x[is_not_outlier]
        valid_y = valid_y[is_not_outlier]
      
        if self.debug_plots:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.imshow(energy_cleaned_v2, cmap='magma')
            ax.set_title(f'All Energy (energy_cleaned_v2) w percentile {percentile}')
            ax.scatter(cx_full, cy_full, c='cyan', s=300, marker='*', label='Center of Mass - BASIC')
            ax.scatter(valid_x, valid_y, c='cyan', s=2, alpha=0.5, label=f'Valid points {len(valid_x)}')
            ax.scatter(invalid_x, invalid_y, c='lime', s=2, alpha=0.5, label=f'Invalid points {len(invalid_x)}')
            ax.legend()
            fig.suptitle(f"PHASE 3: remove outliers")
            plt.tight_layout()
            plt.show()

        #TODO ADD VALIDATION!!

        # ============================================================================
        # PHASE 4: Create Valid Energy Map Using Convex Hull
        # ============================================================================

        valid_energy_map = np.zeros_like(energy_cleaned_v2)
        valid_x_local = valid_x - self.bx
        valid_y_local = valid_y - self.by

        # Initialize functional energy map
        if isinstance(raw_divergence, list):
            functional_energy_map = raw_divergence[-1] if len(raw_divergence) > 0 else energy_cleaned_v2.copy()
        else:
            functional_energy_map = raw_divergence

        # no need to groupping or so ever. keep maps unchanged
        valid_energy_map = energy_cleaned_v2
        

        # ============================================================================
        # PHASE 5: Initialize Core Bounding Box from Percentiles
        # ============================================================================
        # Gonna start exploring the whole mask content
        bird_mask_uint8 = (self.bird_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(bird_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        region_contour = max(contours, key=cv2.contourArea)

        bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(region_contour)
        core_left = bb_x
        core_right = bb_x + bb_w
        core_top = bb_y
        core_bottom = bb_y + bb_h

        if self.debug_plots:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.imshow(energy_cleaned_v2, cmap='magma')
            ax.scatter(cx_full, cy_full, c='cyan', s=300, marker='*', label='Center of Mass - BASIC')
            ax.set_title(f'All Energy (energy_cleaned_v2) w percentile {percentile}')
            ax.scatter(valid_x, valid_y, c='cyan', s=2, alpha=0.5, label='Valid points')
            ax.scatter(invalid_x, invalid_y, c='red', s=2, alpha=0.5, label='Invalid points')

            # Draw the percentile boundaries
            ax.axvline(core_left, color='lime', linestyle='-', linewidth=3, label=f'core_left (2nd %ile={core_left})')
            ax.axvline(core_right, color='green', linestyle='-', linewidth=3, label=f'core_right (98th %ile={core_right})')
            ax.axhline(core_top, color='cyan', linestyle='-', linewidth=2, alpha=0.7, label=f'core_top (98th %ile={core_top})')
            ax.axhline(core_bottom, color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'core_bottom (98th %ile={core_bottom})')


            ax.legend()
            fig.suptitle(f"PHASE 5: SET CORE BBOX")
            plt.tight_layout()
            plt.show()

        if self.debug:
            print(f"CORE Initial (from percentiles): left={core_left}, right={core_right}, top={core_top}, bottom={core_bottom}")

        
        # ============================================================================
        # PHASE 6: Recenter Core Box on Center of Mass
        # ============================================================================

        #print("SKIPPING PHASE 6: Recenter Core Box on Center of Mass")
        
        # ============================================================================
        # PHASE 7: Optimize ROI - Avoid Hand While Maximizing Energy Coverage
        # ============================================================================
        FUNCTIONAL_ENERGY_RATIO = 0.3 # How much of the REAL energy to consider TODO: move to params!

        # Shrink core until it's completely outside hand mask
        hard_constraint_energy = valid_energy_map.copy()
        
        if self.hard_constrain_mode == "energy":
            # Use hybrid energy map for better coverage
            hard_constraint_energy = (valid_energy_map * (1-FUNCTIONAL_ENERGY_RATIO)) + (functional_energy_map * FUNCTIONAL_ENERGY_RATIO)


        # Clamp to frame boundaries
        frame_h, frame_w = self.hand_mask_resized.shape  # or use self.hand_mask_tuned.shape
        core_left   = max(0, core_left)
        core_right  = min(frame_w - 1, core_right)
        core_top    = max(0, core_top)
        core_bottom = min(frame_h - 1, core_bottom)

        old_core_left = core_left
        old_core_right = core_right
        old_core_top = core_top
        old_core_bottom = core_bottom

        safety_mask = 1 - bird_binary
        
        core_left, core_right, core_top, core_bottom = clip_to_mask_smart(
            safety_mask, hard_constraint_energy, 
            core_left, core_right, core_top, core_bottom, 
            self.bx, self.by, mode=self.hard_constrain_mode
        )

        if self.debug_plots:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.imshow(hard_constraint_energy, cmap='magma')
            ax.scatter(cx_full, cy_full, c='cyan', s=300, marker='*', label='Center of Mass - BASIC')
            ax.set_title(f'All Energy (energy_cleaned_v2) w percentile {percentile}')
            ax.scatter(valid_x, valid_y, c='cyan', s=2, alpha=0.5, label='Valid points')
            ax.scatter(invalid_x, invalid_y, c='red', s=2, alpha=0.5, label='Invalid points')

            # Draw the percentile boundaries
            ax.axvline(core_left, color='lime', linestyle='-', linewidth=3, label=f'core_left ({core_left})')
            ax.axvline(core_right, color='green', linestyle='-', linewidth=3, label=f'core_right ({core_right})')
            ax.axhline(core_top, color='cyan', linestyle='-', linewidth=3, alpha=0.7, label=f'core_top ({core_top})')
            ax.axhline(core_bottom, color='blue', linestyle='-', linewidth=3, alpha=0.7, label=f'core_bottom ({core_bottom})')

            ax.axvline(old_core_left, color='lime', linestyle='-', linewidth=1, alpha=0.4,label=f'old core_left (2nd %ile={old_core_left})')
            ax.axvline(old_core_right, color='green', linestyle='-', linewidth=1, alpha=0.4, label=f'old core_right (98th %ile={old_core_right})')
            ax.axhline(old_core_top, color='cyan', linestyle='-', linewidth=1, alpha=0.4, label=f'old core_top (98th %ile={old_core_top})')
            ax.axhline(old_core_bottom, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'old core_bottom (98th %ile={old_core_bottom})')


            ax.legend()
            fig.suptitle(f"PHASE 7: Optimized ROI - constraint mode: {self.hard_constrain_mode}")
            plt.tight_layout()
            plt.show()

       
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
