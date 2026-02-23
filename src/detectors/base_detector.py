"""
Base detector interface for hand detection
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for detectors (actually, segmentator models)
    
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
        Detect element in frame

        Args:
            frame: Input frame (BGR image)

        Returns:
            bbox: (x, y, w, h) or None if not detected
            confidence: Detection confidence (0-1)
            mask: Binary mask of found pixels (same size as frame), or None if not available
        """
        pass

    @abstractmethod
    def post_process(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], mask: np.ndarray) -> np.ndarray:
        """
        Post-process 
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"
