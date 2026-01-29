"""
Breathing measurement methods
"""

import cv2
import numpy as np
from typing import Optional


class IntensityMeasurement:
    """
    Measure breathing using intensity difference
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.prev_chest = None
        print("✓ IntensityMeasurement initialized")
    
    def measure(self, chest_region: np.ndarray) -> float:
        """
        Measure breathing from chest region
        
        Args:
            chest_region: Current chest region (grayscale)
        
        Returns:
            breathing_magnitude: Scalar breathing measurement
        """
        if self.prev_chest is None or self.prev_chest.shape != chest_region.shape:
            self.prev_chest = chest_region.copy()
            return 0.0
        
        # Intensity difference
        diff = cv2.absdiff(chest_region, self.prev_chest)
        breathing = np.mean(diff)
        
        # Update
        self.prev_chest = chest_region.copy()
        
        return float(breathing)
    
    def reset(self):
        """Reset state"""
        self.prev_chest = None


class OpticalFlowMeasurement:
    """
    Measure breathing using optical flow
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.prev_chest = None
        print("✓ OpticalFlowMeasurement initialized")
    
    def measure(self, chest_region: np.ndarray) -> float:
        """
        Measure breathing using optical flow
        """
        if self.prev_chest is None or self.prev_chest.shape != chest_region.shape:
            self.prev_chest = chest_region.copy()
            return 0.0
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_chest, chest_region, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        breathing = np.mean(magnitude)
        
        # Update
        self.prev_chest = chest_region.copy()
        
        return float(breathing)
    
    def reset(self):
        """Reset state"""
        self.prev_chest = None


def get_measurement(config: dict):
    """
    Factory function to get measurement method
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Measurement instance
    """
    method = config.get('method', 'intensity')
    
    if method == 'intensity':
        return IntensityMeasurement(config)
    elif method == 'optical_flow':
        return OpticalFlowMeasurement(config)
    else:
        return IntensityMeasurement(config)
