"""
Chest localization methods
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np


class BaseLocalizer(ABC):
    """
    Abstract base class for chest localization
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def locate(self, bird_mask: np.ndarray, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate chest region within bird mask
        
        Args:
            bird_mask: Binary mask of bird
            **kwargs: Additional arguments
        
        Returns:
            chest_roi: (x, y, w, h) or None
        """
        pass


class SimpleLocalizer(BaseLocalizer):
    """
    Simple anatomical heuristic for chest localization
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Get ratios from config
        ratios = config.get('simple_ratios', {})
        self.y_start = ratios.get('y_start', 0.25)
        self.y_end = ratios.get('y_end', 0.80)
        self.x_margin = ratios.get('x_margin', 0.15)
        self.width = ratios.get('width', 0.70)

        # Get ROI size constraints
        roi_size = config.get('roi_size', {})
        self.min_width = roi_size.get('min_width', 30)
        self.min_height = roi_size.get('min_height', 30)

        print(f"✓ SimpleLocalizer initialized (y_range: {self.y_start}-{self.y_end}, "
              f"x_margin: {self.x_margin}, width: {self.width})")

    def locate(self, bird_mask: np.ndarray, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate chest using anatomical heuristics
        """
        # Find bird bounding box
        contours, _ = cv2.findContours(
            bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        # Get largest contour (bird)
        bird_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(bird_contour)

        # Calculate chest position based on ratios
        chest_y_start = y + int(self.y_start * h)
        chest_y_end = y + int(self.y_end * h)
        chest_x_start = x + int(self.x_margin * w)
        chest_w = int(self.width * w)

        chest_h = chest_y_end - chest_y_start

        # Apply minimum size constraints
        chest_w = max(chest_w, self.min_width)
        chest_h = max(chest_h, self.min_height)

        return (chest_x_start, chest_y_start, chest_w, chest_h)


class ContourLocalizer(BaseLocalizer):
    """
    Locate chest using contour analysis - finds widest part of bird body
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Get contour-specific settings
        contour_config = config.get('contour', {})
        self.width_ratio = contour_config.get('width_ratio', 0.70)
        self.height_ratio = contour_config.get('height_ratio', 0.55)
        self.x_margin = contour_config.get('x_margin', 0.15)
        self.scan_range = contour_config.get('scan_range', [0.15, 0.85])

        # Get ROI size constraints
        roi_size = config.get('roi_size', {})
        self.min_width = roi_size.get('min_width', 30)
        self.min_height = roi_size.get('min_height', 30)

        print(f"✓ ContourLocalizer initialized (width: {self.width_ratio}, height: {self.height_ratio})")

    def locate(self, bird_mask: np.ndarray, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        """
        Find chest by analyzing contour width at different heights
        The chest is typically the widest part of the bird's body
        """
        contours, _ = cv2.findContours(
            bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        # Get largest contour (bird)
        bird_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(bird_contour)

        # Analyze horizontal width at different vertical positions
        widths = []
        y_positions = []

        # Sample using configurable scan range
        start_y = y + int(h * self.scan_range[0])
        end_y = y + int(h * self.scan_range[1])

        for y_scan in range(start_y, end_y, max(1, h // 20)):
            # Find leftmost and rightmost points at this height
            mask_row = bird_mask[y_scan, :]
            nonzero = np.nonzero(mask_row)[0]

            if len(nonzero) > 0:
                width = nonzero[-1] - nonzero[0]
                widths.append(width)
                y_positions.append(y_scan)

        if len(widths) == 0:
            # Fallback using configured ratios
            chest_w = max(int(w * self.width_ratio), self.min_width)
            chest_h = max(int(h * self.height_ratio), self.min_height)
            return (x + int(w * self.x_margin), y + int(h * 0.35), chest_w, chest_h)

        # Find the region with maximum width (chest)
        max_width_idx = np.argmax(widths)
        chest_y_center = y_positions[max_width_idx]

        # Create ROI centered around maximum width region using configured sizes
        chest_h = max(int(h * self.height_ratio), self.min_height)
        chest_y = max(y, chest_y_center - chest_h // 2)
        chest_x = x + int(w * self.x_margin)
        chest_w = max(int(w * self.width_ratio), self.min_width)

        return (chest_x, chest_y, chest_w, chest_h)


class VarianceLocalizer(BaseLocalizer):
    """
    Locate chest using pixel variance - chest has highest temporal variance
    Requires multiple frames to analyze motion
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Get variance-specific settings
        variance_config = config.get('variance', {})
        self.buffer_size = variance_config.get('buffer_frames', 30)
        self.grid_size = variance_config.get('grid_size', 4)
        self.roi_cells = variance_config.get('roi_cells', 3)

        # Get ROI size constraints
        roi_size = config.get('roi_size', {})
        self.min_width = roi_size.get('min_width', 30)
        self.min_height = roi_size.get('min_height', 30)

        self.frame_buffer = []
        print(f"✓ VarianceLocalizer initialized (buffer={self.buffer_size}, "
              f"grid={self.grid_size}x{self.grid_size}, roi={self.roi_cells}x{self.roi_cells} cells)")

    def locate(self, bird_mask: np.ndarray, hand_mask: np.ndarray = None, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        """
        Find chest by analyzing temporal variance across frames
        Chest region has highest variance due to breathing motion

        Args:
            bird_mask: Bird segmentation mask (or hand mask if segmentation disabled)
            hand_mask: Optional hand mask for additional constraint
            **kwargs: frame, fps
        """
        frame = kwargs.get('frame')

        if frame is None:
            # Fallback to simple method if no frame provided
            contours, _ = cv2.findContours(
                bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                return None
            bird_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(bird_contour)
            return (x + int(w * 0.25), y + int(h * 0.35), int(w * 0.5), int(h * 0.35))

        # Get analysis region bounding box (bird or hand)
        contours, _ = cv2.findContours(
            bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        region_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(region_contour)

        # Extract region from frame WITHOUT zeroing pixels
        # This allows motion detection to work on all chest colors (including black)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        region = gray[y:y+h, x:x+w].copy()
        mask_region = bird_mask[y:y+h, x:x+w]

        # Keep all pixel values for motion analysis - don't zero out!

        # Add to buffer
        self.frame_buffer.append(region)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # Need enough frames for variance calculation
        if len(self.frame_buffer) < self.buffer_size:
            # Return simple estimate while collecting frames (use configured ratios)
            chest_w = max(int(w * 0.70), self.min_width)
            chest_h = max(int(h * 0.55), self.min_height)
            return (x + int(w * 0.15), y + int(h * 0.25), chest_w, chest_h)

        # Calculate temporal variance
        buffer_array = np.array(self.frame_buffer, dtype=np.float32)
        variance_map = np.var(buffer_array, axis=0)

        # Divide into grid and find region with highest variance
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        max_variance = 0
        best_cell_y = self.grid_size // 2
        best_cell_x = self.grid_size // 2

        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                cell_y_start = gy * cell_h
                cell_y_end = min((gy + 1) * cell_h, h)
                cell_x_start = gx * cell_w
                cell_x_end = min((gx + 1) * cell_w, w)

                cell_variance = np.mean(variance_map[cell_y_start:cell_y_end,
                                                     cell_x_start:cell_x_end])

                if cell_variance > max_variance:
                    max_variance = cell_variance
                    best_cell_y = gy
                    best_cell_x = gx

        # Create ROI around highest variance cell (configurable size)
        half_roi = self.roi_cells // 2
        chest_y = y + max(0, best_cell_y * cell_h - half_roi * cell_h)
        chest_x = x + max(0, best_cell_x * cell_w - half_roi * cell_w)
        chest_h = max(min(cell_h * self.roi_cells, h), self.min_height)
        chest_w = max(min(cell_w * self.roi_cells, w), self.min_width)

        return (chest_x, chest_y, chest_w, chest_h)


class MotionLocalizer(BaseLocalizer):
    """
    Locate chest using optical flow magnitude - chest has periodic motion
    Requires frame buffer to analyze motion over time
    """

    def __init__(self, config: dict):
        super().__init__(config)
        motion_config = config.get('motion', {})
        self.buffer_frames = motion_config.get('buffer_frames', 60)
        self.grid_size = motion_config.get('grid_size', 5)
        self.roi_cells = motion_config.get('roi_cells', 3)

        # Get ROI size constraints
        roi_size = config.get('roi_size', {})
        self.min_width = roi_size.get('min_width', 30)
        self.min_height = roi_size.get('min_height', 30)

        self.frame_buffer = []
        print(f"✓ MotionLocalizer initialized (buffer={self.buffer_frames}, "
              f"grid={self.grid_size}x{self.grid_size}, roi={self.roi_cells}x{self.roi_cells} cells)")

    def locate(self, bird_mask: np.ndarray, hand_mask: np.ndarray = None, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        """
        Find chest by analyzing optical flow patterns
        Chest has periodic motion from breathing

        Args:
            bird_mask: Bird segmentation mask (or hand mask if segmentation disabled)
            hand_mask: Optional hand mask for additional constraint
            **kwargs: frame, fps
        """
        frame = kwargs.get('frame')

        if frame is None:
            # Fallback
            contours, _ = cv2.findContours(
                bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                return None
            region_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(region_contour)
            return (x + int(w * 0.25), y + int(h * 0.35), int(w * 0.5), int(h * 0.35))

        # Get analysis region bounding box (bird or hand)
        contours, _ = cv2.findContours(
            bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        region_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(region_contour)

        # Extract region from frame WITHOUT zeroing pixels
        # This allows motion detection to work on all chest colors (including black)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        region = gray[y:y+h, x:x+w].copy()
        mask_region = bird_mask[y:y+h, x:x+w]

        # Keep all pixel values for motion analysis - don't zero out!

        # Add to buffer
        self.frame_buffer.append(region)
        if len(self.frame_buffer) > self.buffer_frames:
            self.frame_buffer.pop(0)

        # Need at least 2 frames for optical flow
        if len(self.frame_buffer) < 2:
            chest_w = max(int(w * 0.70), self.min_width)
            chest_h = max(int(h * 0.55), self.min_height)
            return (x + int(w * 0.15), y + int(h * 0.25), chest_w, chest_h)

        # Calculate optical flow magnitude for recent frames
        flow_magnitudes = []
        for i in range(len(self.frame_buffer) - 1):
            prev_region = self.frame_buffer[i]
            curr_region = self.frame_buffer[i + 1]

            flow = cv2.calcOpticalFlowFarneback(
                prev_region, curr_region, None,
                pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )

            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(magnitude)

        # Average flow magnitude across all frames
        region_flow = np.mean(flow_magnitudes, axis=0)

        # Divide into grid
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        max_flow = 0
        best_cell_y = self.grid_size // 2
        best_cell_x = self.grid_size // 2

        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                cell_y_start = gy * cell_h
                cell_y_end = min((gy + 1) * cell_h, h)
                cell_x_start = gx * cell_w
                cell_x_end = min((gx + 1) * cell_w, w)

                cell_flow = np.mean(region_flow[cell_y_start:cell_y_end,
                                                 cell_x_start:cell_x_end])

                if cell_flow > max_flow:
                    max_flow = cell_flow
                    best_cell_y = gy
                    best_cell_x = gx

        # Create ROI around highest flow cell (configurable size)
        half_roi = self.roi_cells // 2
        chest_y = y + max(0, best_cell_y * cell_h - half_roi * cell_h)
        chest_x = x + max(0, best_cell_x * cell_w - half_roi * cell_w)
        chest_h = max(min(cell_h * self.roi_cells, h), self.min_height)
        chest_w = max(min(cell_w * self.roi_cells, w), self.min_width)

        return (chest_x, chest_y, chest_w, chest_h)


class OpticalFlowLocalizer(BaseLocalizer):
    """
    Advanced optical flow-based localization with frequency analysis
    Finds region with breathing-rate frequency motion
    """

    def __init__(self, config: dict):
        super().__init__(config)
        motion_config = config.get('motion', {})
        self.buffer_frames = motion_config.get('buffer_frames', 60)
        self.grid_size = motion_config.get('grid_size', 5)
        self.roi_cells = motion_config.get('roi_cells', 3)
        self.freq_range = motion_config.get('freq_range', [0.5, 4.0])  # Hz

        # Get ROI size constraints
        roi_size = config.get('roi_size', {})
        self.min_width = roi_size.get('min_width', 30)
        self.min_height = roi_size.get('min_height', 30)

        self.frame_buffer = []
        self.motion_history = []
        print(f"✓ OpticalFlowLocalizer initialized (buffer={self.buffer_frames}, "
              f"grid={self.grid_size}x{self.grid_size}, roi={self.roi_cells}x{self.roi_cells} cells, "
              f"freq_range={self.freq_range} Hz)")

    def locate(self, bird_mask: np.ndarray, hand_mask: np.ndarray = None, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        """
        Find chest using frequency analysis of optical flow
        Identifies region with periodic motion at breathing frequency

        Args:
            bird_mask: Bird segmentation mask (or hand mask if segmentation disabled)
            hand_mask: Optional hand mask for additional constraint
            **kwargs: frame, fps
        """
        frame = kwargs.get('frame')
        fps = kwargs.get('fps', 30)

        if frame is None:
            # Fallback
            contours, _ = cv2.findContours(
                bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                return None
            region_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(region_contour)
            return (x + int(w * 0.25), y + int(h * 0.35), int(w * 0.5), int(h * 0.35))

        # Get analysis region bounding box (bird or hand)
        contours, _ = cv2.findContours(
            bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        region_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(region_contour)

        # Extract region from frame WITHOUT zeroing pixels
        # This allows motion detection to work on all chest colors (including black)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        region = gray[y:y+h, x:x+w].copy()
        mask_region = bird_mask[y:y+h, x:x+w]

        # Keep all pixel values for motion analysis - don't zero out!

        # Add to buffer
        self.frame_buffer.append(region)
        if len(self.frame_buffer) > self.buffer_frames:
            self.frame_buffer.pop(0)

        # Need full buffer for frequency analysis
        if len(self.frame_buffer) < self.buffer_frames:
            chest_w = max(int(w * 0.70), self.min_width)
            chest_h = max(int(h * 0.55), self.min_height)
            return (x + int(w * 0.15), y + int(h * 0.25), chest_w, chest_h)

        # Divide analysis region into grid
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        # For each grid cell, compute motion time series
        max_power = 0
        best_cell_y = self.grid_size // 2
        best_cell_x = self.grid_size // 2

        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                # Use region coordinates (bird or hand, not full frame)
                cell_y_start = gy * cell_h
                cell_x_start = gx * cell_w
                cell_y_end = min((gy + 1) * cell_h, h)
                cell_x_end = min((gx + 1) * cell_w, w)

                # Extract motion time series for this cell (from frame buffer)
                motion_series = []
                for i in range(len(self.frame_buffer) - 1):
                    prev = self.frame_buffer[i][cell_y_start:cell_y_end, cell_x_start:cell_x_end]
                    curr = self.frame_buffer[i + 1][cell_y_start:cell_y_end, cell_x_start:cell_x_end]

                    if prev.size > 0 and curr.size > 0:
                        diff = np.abs(curr.astype(float) - prev.astype(float))
                        motion_series.append(np.mean(diff))
                    else:
                        motion_series.append(0)

                if len(motion_series) < 10:
                    continue

                # FFT to find power in breathing frequency range
                fft_result = np.fft.fft(motion_series)
                frequencies = np.fft.fftfreq(len(motion_series), 1/fps)

                # Power in breathing range
                mask = (frequencies >= self.freq_range[0]) & (frequencies <= self.freq_range[1])
                power = np.sum(np.abs(fft_result[mask]))

                if power > max_power:
                    max_power = power
                    best_cell_y = gy
                    best_cell_x = gx

        # Create ROI around best cell (configurable size)
        half_roi = self.roi_cells // 2
        chest_y = y + max(0, best_cell_y * cell_h - half_roi * cell_h)
        chest_x = x + max(0, best_cell_x * cell_w - half_roi * cell_w)
        chest_h = max(min(cell_h * self.roi_cells, h), self.min_height)
        chest_w = max(min(cell_w * self.roi_cells, w), self.min_width)

        return (chest_x, chest_y, chest_w, chest_h)


def get_localizer(config: dict) -> BaseLocalizer:
    """
    Factory function to get localizer based on config

    Args:
        config: Configuration dictionary

    Returns:
        Localizer instance
    """
    method = config.get('method', 'simple')

    if method == 'simple':
        return SimpleLocalizer(config)
    elif method == 'contour':
        return ContourLocalizer(config)
    elif method == 'variance':
        return VarianceLocalizer(config)
    elif method == 'motion':
        return MotionLocalizer(config)
    elif method == 'optical_flow':
        return OpticalFlowLocalizer(config)
    else:
        print(f"⚠ Unknown localization method '{method}', defaulting to 'simple'")
        return SimpleLocalizer(config)
