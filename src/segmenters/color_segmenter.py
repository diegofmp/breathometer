"""
Color-based bird segmentation
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .base_segmenter import BaseSegmenter


class ColorSegmenter(BaseSegmenter):
    """
    Segment bird using HSV color thresholding
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Get color ranges from config
        self.color_ranges = config.get('color_range', {})
        self.default_color = config.get('color_range', {}).get('default', 'green')
        
        print(f"✓ ColorSegmenter initialized (default: {self.default_color})")
    
    def segment(self, hand_region: np.ndarray, hand_mask_local: np.ndarray = None, color: str = None, **kwargs) -> np.ndarray:
        """
        Segment bird using color thresholding

        Args:
            hand_region: Hand region image
            hand_mask_local: Optional hand segmentation mask in local coordinates
            color: Color name ('green', 'blue', 'yellow') or None for default
            **kwargs: Can include 'color_range' as tuple of (lower, upper) HSV

        Returns:
            bird_mask: Binary mask
        """
        # Get color range
        if 'color_range' in kwargs:
            lower, upper = kwargs['color_range']
            lower_color = np.array(lower, dtype=np.uint8)
            upper_color = np.array(upper, dtype=np.uint8)
        else:
            color = color or self.default_color
            if color in self.color_ranges:
                lower_color = np.array(self.color_ranges[color][0], dtype=np.uint8)
                upper_color = np.array(self.color_ranges[color][1], dtype=np.uint8)
            else:
                # Default green
                lower_color = np.array([35, 40, 40], dtype=np.uint8)
                upper_color = np.array([85, 255, 255], dtype=np.uint8)

        # Convert to HSV
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)

        # Threshold
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # If hand mask is provided, only keep bird pixels inside the hand region
        if hand_mask_local is not None:
            mask = cv2.bitwise_and(mask, hand_mask_local)

        # Clean up
        mask = self._clean_mask(mask)

        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean mask using morphological operations
        """
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes
        mask = binary_fill_holes(mask).astype(np.uint8) * 255
        
        return mask
    
    def get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract largest connected component
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels <= 1:
            return mask
        
        # Find largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create mask with only largest component
        largest_mask = (labels == largest_label).astype(np.uint8) * 255
        
        return largest_mask
