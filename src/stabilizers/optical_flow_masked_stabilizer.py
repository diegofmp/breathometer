"""
Optical flow-based stabilization using masked region (hand area only)
"""

import cv2
import numpy as np
from .base_stabilizer import BaseStabilizer


class OpticalFlowMaskedStabilizer(BaseStabilizer):
    """
    Stabilize chest region using optical flow computed only in hand region
    This focuses on motion within the bird/hand area and ignores background
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.prev_gray = None
        print(f"✓ OpticalFlowMaskedStabilizer initialized")

    def stabilize(self, chest_region: np.ndarray, frame: np.ndarray = None,
                 hand_bbox: tuple = None, prev_hand_bbox: tuple = None,
                 prev_frame: np.ndarray = None) -> np.ndarray:
        """
        Stabilize using optical flow computed within hand bounding box mask

        Args:
            chest_region: Chest ROI to stabilize
            frame: Current full frame
            prev_frame: Previous full frame
            hand_bbox: Current hand bounding box (to create mask)

        Returns:
            Stabilized chest region
        """
        if prev_frame is None or frame is None or hand_bbox is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
            return chest_region

        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Create mask for hand region
        mask = np.zeros(curr_gray.shape, dtype=np.uint8)
        x, y, w, h = [int(v) for v in hand_bbox]

        # Expand mask slightly beyond hand bbox
        margin = 0.1
        x_start = max(0, int(x - w * margin))
        y_start = max(0, int(y - h * margin))
        x_end = min(curr_gray.shape[1], int(x + w * (1 + margin)))
        y_end = min(curr_gray.shape[0], int(y + h * (1 + margin)))

        mask[y_start:y_end, x_start:x_end] = 255

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

        # Extract flow only in masked region
        flow_masked = flow[mask > 0]

        if len(flow_masked) > 0:
            # Compute median flow in hand region (hand motion)
            flow_median = np.median(flow_masked, axis=0)
        else:
            flow_median = np.array([0.0, 0.0])

        # Compensate for hand motion in chest region
        M = np.float32([[1, 0, -flow_median[0]],
                        [0, 1, -flow_median[1]]])

        h_chest, w_chest = chest_region.shape[:2]
        chest_stabilized = cv2.warpAffine(chest_region, M, (w_chest, h_chest),
                                         borderMode=cv2.BORDER_REPLICATE)

        self.prev_gray = curr_gray
        return chest_stabilized

    def reset(self):
        """Reset optical flow state"""
        self.prev_gray = None
