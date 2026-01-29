"""
Optical flow-based stabilization using full frame motion
"""

import cv2
import numpy as np
from .base_stabilizer import BaseStabilizer


class OpticalFlowStabilizer(BaseStabilizer):
    """
    Stabilize chest region using optical flow computed on full frame
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.prev_gray = None
        print(f"✓ OpticalFlowStabilizer initialized")

    def stabilize(self, chest_region: np.ndarray, frame: np.ndarray = None,
                 hand_bbox: tuple = None, prev_hand_bbox: tuple = None,
                 prev_frame: np.ndarray = None) -> np.ndarray:
        """
        Stabilize using optical flow from full frame

        Args:
            chest_region: Chest ROI to stabilize
            frame: Current full frame
            prev_frame: Previous full frame

        Returns:
            Stabilized chest region
        """
        if prev_frame is None or frame is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
            return chest_region

        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute median flow (global camera motion)
        flow_median = np.median(flow.reshape(-1, 2), axis=0)

        # Compensate for global motion in chest region
        M = np.float32([[1, 0, -flow_median[0]],
                        [0, 1, -flow_median[1]]])

        h, w = chest_region.shape[:2]
        chest_stabilized = cv2.warpAffine(chest_region, M, (w, h),
                                         borderMode=cv2.BORDER_REPLICATE)

        self.prev_gray = curr_gray
        return chest_stabilized

    def reset(self):
        """Reset optical flow state"""
        self.prev_gray = None
