"""
Hand detectors module
"""

from .base_detector import BaseDetector
from .manual_detector import ManualDetector

# Optional: RF-DETR detector (requires 'rfdetr' package)
try:
    from .rfdetr_detector import RFDETRDetector
    _RFDETR_DETECTOR_AVAILABLE = True
except ImportError:
    _RFDETR_DETECTOR_AVAILABLE = False
    RFDETRDetector = None  # type: ignore

__all__ = ['BaseDetector', 'ManualDetector', 'RFDETRDetector',]

def get_detector(config: dict) -> BaseDetector:
    """
    Factory function to get detector based on config

    Args:
        config: Configuration dictionary

    Returns:
        Detector instance
    """
    mode = config.get('mode', 'rfdetr')

    if mode == 'manual':
        return ManualDetector(config)
    elif mode == 'rfdetr':
        if not _RFDETR_DETECTOR_AVAILABLE:
            raise ImportError(
                "RF-DETR detector requested (mode='rfdetr') but the 'rfdetr' package "
                "is not installed. Install it with: pip install rfdetr"
            )
        return RFDETRDetector(config)
    else:
        raise ValueError(
            f"Unknown detection mode: '{mode}'. "
            f"Supported modes: 'manual', 'rfdetr'"
        )
