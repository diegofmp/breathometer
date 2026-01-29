"""
Hand detectors module
"""

from .base_detector import BaseDetector
from .yolo_detector import YOLODetector

__all__ = ['BaseDetector', 'YOLODetector']


def get_detector(config: dict) -> BaseDetector:
    """
    Factory function to get detector based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Detector instance
    """
    # For now, only YOLO is implemented
    return YOLODetector(config)
