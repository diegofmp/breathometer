"""
Base segmenter interface
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSegmenter(ABC):
    """
    Abstract base class for bird segmentation
    """
    
    def __init__(self, config: dict):
        """
        Initialize segmenter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def segment(self, hand_region: np.ndarray, hand_mask_local: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Segment bird within hand region

        Args:
            hand_region: Cropped hand region (BGR image)
            hand_mask_local: Optional hand segmentation mask in local coordinates (same size as hand_region)
                           Used to exclude hand pixels from bird segmentation
            **kwargs: Additional arguments

        Returns:
            bird_mask: Binary mask (same size as hand_region)
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"
