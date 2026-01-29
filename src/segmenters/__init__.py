"""
Bird segmenters module
"""

from .base_segmenter import BaseSegmenter
from .color_segmenter import ColorSegmenter
from .grabcut_segmenter import GrabCutSegmenter

__all__ = ['BaseSegmenter', 'ColorSegmenter', 'GrabCutSegmenter']


def get_segmenter(config: dict) -> BaseSegmenter:
    """
    Factory function to get segmenter based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Segmenter instance
    """
    method = config.get('method', 'color')
    
    if method == 'color':
        return ColorSegmenter(config)
    elif method == 'grabcut':
        return GrabCutSegmenter(config)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def convert_mask_to_frame_coords(mask_local: 'np.ndarray', hand_bbox: tuple, frame_shape: tuple) -> 'np.ndarray':
    """
    Convert mask from hand region coordinates to full frame coordinates
    
    Args:
        mask_local: Binary mask in hand region coordinates
        hand_bbox: (x, y, w, h) of hand region
        frame_shape: (height, width) of full frame
    
    Returns:
        mask_full: Binary mask in frame coordinates
    """
    import numpy as np
    
    x, y, w, h = [int(v) for v in hand_bbox]
    
    mask_full = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    # Handle edge cases
    x_end = min(x + w, frame_shape[1])
    y_end = min(y + h, frame_shape[0])
    
    mask_h = y_end - y
    mask_w = x_end - x
    
    mask_full[y:y_end, x:x_end] = mask_local[:mask_h, :mask_w]
    
    return mask_full
