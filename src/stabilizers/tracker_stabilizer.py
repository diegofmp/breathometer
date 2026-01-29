"""
Tracker-based stabilization using hand bounding box motion
"""

import cv2
import numpy as np
from .base_stabilizer import BaseStabilizer


class TrackerPositionStabilizer(BaseStabilizer):
    """
    Stabilize chest region by compensating for hand tracker motion
    """

    def __init__(self, config: dict):
        super().__init__(config)
        print(f"✓ TrackerPositionStabilizer initialized")

    def stabilize(self, chest_region: np.ndarray, frame: np.ndarray = None,
                 hand_bbox: tuple = None, prev_hand_bbox: tuple = None,
                 prev_frame: np.ndarray = None) -> np.ndarray:
        """
        Stabilize by compensating for hand center motion

        Args:
            chest_region: Chest ROI to stabilize
            hand_bbox: Current hand bounding box
            prev_hand_bbox: Previous hand bounding box

        Returns:
            Stabilized chest region
        """
        if prev_hand_bbox is None or hand_bbox is None:
            return chest_region

        # Compute hand motion (center difference)
        prev_center = np.array([
            prev_hand_bbox[0] + prev_hand_bbox[2]/2,
            prev_hand_bbox[1] + prev_hand_bbox[3]/2
        ])
        curr_center = np.array([
            hand_bbox[0] + hand_bbox[2]/2,
            hand_bbox[1] + hand_bbox[3]/2
        ])

        hand_motion = curr_center - prev_center

        # Warp chest to compensate for hand motion
        M = np.float32([[1, 0, -hand_motion[0]],
                        [0, 1, -hand_motion[1]]])

        h, w = chest_region.shape[:2]
        chest_stabilized = cv2.warpAffine(chest_region, M, (w, h),
                                         borderMode=cv2.BORDER_REPLICATE)

        return chest_stabilized
