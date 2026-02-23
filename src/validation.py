"""
Motion validation utilities for breathing measurement

This module provides validators to detect and reject non-breathing motion
artifacts (e.g., hand movements, bird repositioning) that can contaminate
the breathing signal.
"""

import numpy as np
from typing import Optional, Tuple


class RadialConsistencyValidator:
    """
    Validates breathing motion by checking radial consistency.

    Breathing produces radially-symmetric expansion/contraction from the chest center.
    Non-breathing motion (hand movements, bird repositioning) produces directional
    or sliding motion that lacks radial symmetry.

    This validator computes:
    1. Consistency: How much of total motion is radially aligned (vs tangential)
    2. Coherence: Whether radial motion is uniform (all expansion or all contraction)
    3. Motion magnitude: Whether motion is within physiological breathing range

    Based on the principle that valid breathing should have:
    - High radial consistency (>0.4): motion is organized radially, not sliding
    - High coherence (>0.5): pixels move in same direction (all expand or all contract)
    - Moderate magnitude (0.1-2.0): not too small (noise) or too large (artifact)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Validation configuration with optional parameters:
                - consistency_threshold: Minimum radial consistency (default: 0.4)
                - coherence_threshold: Minimum radial coherence (default: 0.5)
                - max_motion_magnitude: Maximum median motion (default: 2.0)
                - min_motion_magnitude: Minimum total motion (default: 0.1)
                - fallback_mode: What to return on rejection: 'zero', 'previous', 'none'
                  (default: 'previous')
        """
        self.consistency_threshold = config.get('consistency_threshold', 0.4)
        #self.coherence_threshold = config.get('coherence_threshold', 0.5)
        self.coherence_threshold = config.get('coherence_threshold', 0.3)
        self.max_motion_magnitude = config.get('max_motion_magnitude', 2.0)
        self.min_motion_magnitude = config.get('min_motion_magnitude', 0.1)
        self.fallback_mode = config.get('fallback_mode', 'previous')

        # Statistics tracking
        self.total_frames = 0
        self.rejected_frames = 0
        self.rejection_reasons = {
            'low_consistency': 0,
            'low_coherence': 0,
            'excessive_motion': 0,
            'insufficient_motion': 0
        }

        print(f"✓ RadialConsistencyValidator initialized")
        print(f"  Consistency threshold: {self.consistency_threshold}")
        print(f"  Coherence threshold: {self.coherence_threshold}")
        print(f"  Motion range: [{self.min_motion_magnitude}, {self.max_motion_magnitude}]")
        print(f"  Fallback mode: {self.fallback_mode}")

    def validate(self,
                 u_x_res: np.ndarray,
                 u_y_res: np.ndarray,
                 signal_value: float,
                 fallback_value: float = 0.0,
                 center = None) -> Tuple[float, bool, dict]:
        """
        Validate breathing motion using radial consistency analysis.

        Args:
            u_x_res: Residual horizontal optical flow (after rigid motion removal)
            u_y_res: Residual vertical optical flow (after rigid motion removal)
            signal_value: Proposed breathing signal value
            fallback_value: Value to return if motion is rejected (default: 0.0)
            center: center of the bbox. Ideally the proper center of energy

        Returns:
            Tuple of:
            - validated_signal: Either signal_value (if valid) or fallback_value (if rejected)
            - is_valid: Boolean indicating if motion passed validation
            - metrics: Dictionary with validation metrics for debugging:
                - consistency: Radial consistency score (0-1)
                - coherence: Radial coherence score (0-1)
                - median_magnitude: Median motion magnitude
                - total_motion: Total motion magnitude
                - rejection_reason: String description if rejected, None if valid
        """
        self.total_frames += 1

        # Calculate motion magnitude field
        mag = np.sqrt(u_x_res**2 + u_y_res**2)
        total_motion = np.sum(mag)
        median_mag = np.median(mag)

        # Calculate radial unit vectors from center
        h, w = u_x_res.shape

        if center is None:
            center_x, center_y = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            (center_x, center_y) = center
            
        y, x = np.indices((h, w))
        rel_x, rel_y = x - center_x, y - center_y
        dist = np.sqrt(rel_x**2 + rel_y**2) + 1e-9
        unit_rx, unit_ry = rel_x / dist, rel_y / dist

        # Compute radial projection: how much motion is aligned with radial direction
        radial_projection = (u_x_res * unit_rx) + (u_y_res * unit_ry)

        # Metrics:
        # 1. Consistency: fraction of motion that is radially aligned (vs tangential)
        #    High consistency (>0.4) means motion is organized radially
        radial_magnitude = np.sum(np.abs(radial_projection))
        consistency = radial_magnitude / (total_motion + 1e-9)

        # 2. Coherence: whether radial motion is uniform (all expansion or all contraction)
        ##### V1 WORKING!!
        # #    High coherence (>0.5) means pixels move in same radial direction
        # radial_signed_sum = np.sum(radial_projection)
        # coherence = np.abs(radial_signed_sum) / (radial_magnitude + 1e-9)

        # --- UPDATED COHERENCE LOGIC -------------------------------------------
        # Separate the 'push' from the 'pull'
        expansion = np.sum(radial_projection[radial_projection > 0])
        contraction = np.abs(np.sum(radial_projection[radial_projection < 0]))

        # Coherence is now: "Is the motion dominated by ONE radial direction?"
        # This prevents symmetric expansion from canceling itself out.
        coherence = max(expansion, contraction) / (expansion + contraction + 1e-9)
        # --- UPDATED COHERENCE LOGIC -------------------------------------

        # Validation checks
        rejection_reason = None

        if total_motion < self.min_motion_magnitude:
            rejection_reason = 'insufficient_motion'
            self.rejection_reasons['insufficient_motion'] += 1
        elif median_mag > self.max_motion_magnitude:
            rejection_reason = 'excessive_motion'
            self.rejection_reasons['excessive_motion'] += 1
        elif consistency < self.consistency_threshold:
            rejection_reason = 'low_consistency'
            self.rejection_reasons['low_consistency'] += 1
        elif coherence < self.coherence_threshold:
            rejection_reason = 'low_coherence'
            self.rejection_reasons['low_coherence'] += 1

        # Determine result
        if rejection_reason is None:
            # Valid breathing motion
            validated_signal = signal_value
            is_valid = True
        else:
            # Rejected - use fallback
            self.rejected_frames += 1
            validated_signal = fallback_value
            is_valid = False

        # Metrics for debugging
        metrics = {
            'consistency': consistency,
            'coherence': coherence,
            'median_magnitude': median_mag,
            'total_motion': total_motion,
            'rejection_reason': rejection_reason
        }

        return validated_signal, is_valid, metrics

    def get_statistics(self) -> dict:
        """
        Get validation statistics.

        Returns:
            Dictionary with:
            - total_frames: Total frames processed
            - rejected_frames: Number of rejected frames
            - rejection_rate: Fraction of frames rejected
            - rejection_reasons: Breakdown of rejection reasons
        """
        rejection_rate = self.rejected_frames / max(self.total_frames, 1)

        return {
            'total_frames': self.total_frames,
            'rejected_frames': self.rejected_frames,
            'rejection_rate': rejection_rate,
            'rejection_reasons': self.rejection_reasons.copy()
        }

    def reset_statistics(self):
        """Reset validation statistics counters"""
        self.total_frames = 0
        self.rejected_frames = 0
        self.rejection_reasons = {
            'low_consistency': 0,
            'low_coherence': 0,
            'excessive_motion': 0,
            'insufficient_motion': 0
        }

    def reset(self):
        """Reset validator state (currently just statistics)"""
        self.reset_statistics()
