"""
Stabilization methods for chest region
"""

from .base_stabilizer import BaseStabilizer
from .tracker_stabilizer import TrackerPositionStabilizer
from .optical_flow_stabilizer import OpticalFlowStabilizer
from .optical_flow_masked_stabilizer import OpticalFlowMaskedStabilizer

__all__ = [
    'BaseStabilizer',
    'TrackerPositionStabilizer',
    'OpticalFlowStabilizer',
    'OpticalFlowMaskedStabilizer'
]


def get_stabilizer(config: dict):
    """
    Factory function to get stabilizer based on config

    Args:
        config: Configuration dictionary

    Returns:
        Stabilizer instance or None if stabilization disabled
    """
    if not config.get('enabled', False):
        return None

    method = config.get('method', 'tracker_position')

    if method == 'tracker_position':
        return TrackerPositionStabilizer(config)
    elif method == 'optical_flow':
        return OpticalFlowStabilizer(config)
    elif method == 'optical_flow_masked':
        return OpticalFlowMaskedStabilizer(config)
    else:
        raise ValueError(f"Unknown stabilization method: {method}")
