"""
Base detector interface for hand detection
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for hand detectors
    
    All detectors must implement the detect() method
    """
    
    def __init__(self, config: dict):
        """
        Initialize detector with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]:
        """
        Detect hand in frame

        Args:
            frame: Input frame (BGR image)

        Returns:
            hand_bbox: (x, y, w, h) or None if not detected
            confidence: Detection confidence (0-1)
            hand_mask: Binary mask of hand pixels (same size as frame), or None if not available
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"
