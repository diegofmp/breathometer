"""
Base class for stabilization methods
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseStabilizer(ABC):
    """
    Base class for chest region stabilization
    """

    def __init__(self, config: dict):
        """
        Initialize stabilizer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.method = config.get('method', 'none')

    @abstractmethod
    def stabilize(self, chest_region: np.ndarray, frame: np.ndarray = None,
                 hand_bbox: tuple = None, prev_hand_bbox: tuple = None,
                 prev_frame: np.ndarray = None) -> np.ndarray:
        """
        Stabilize chest region

        Args:
            chest_region: Chest ROI to stabilize
            frame: Full frame (optional, for optical flow methods)
            hand_bbox: Current hand bounding box (optional, for tracker method)
            prev_hand_bbox: Previous hand bounding box (optional, for tracker method)
            prev_frame: Previous full frame (optional, for optical flow methods)

        Returns:
            Stabilized chest region
        """
        pass

    def reset(self):
        """Reset stabilizer state"""
        pass
