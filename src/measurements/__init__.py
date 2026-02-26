"""
Breathing measurement methods
"""

import cv2
import numpy as np
from typing import Optional
import pandas as pd

class OpticalFlowDivergenceRobustMeasurement:
    """
    Robust breathing measurement using optical flow radial projection.

    Pipeline:
    1. Dense optical flow (Farnebäck)
    2. Remove dominant rigid motion (affine or median translation)
    3. Compute radial projection (expansion/contraction) of residual flow
    4. Validate radial consistency (optional, rejects hand movements)
    5. Aggregate signal globally or patch-wise (optional)

    Radial projection detects breathing motion as expansion/contraction
    from a center point (either energy-weighted or geometric center).

    Patch-wise mode increases robustness against local disturbances
    (e.g. fingers, feathers, partial occlusion).

    Radial consistency validation rejects non-breathing artifacts like
    hand movements by checking if motion is radially symmetric.
    """

    def __init__(self, config: dict):
        self.config = config
        self.prev_chest = None
        self.last_valid_breathing = 0.0  # For fallback when motion is rejected

        # Center calculation method
        self.center_type = config.get('center_method', "energy_center")

        # Optical flow parameters
        self.pyr_scale = config.get('pyr_scale', 0.5)
        self.levels = config.get('levels', 3)
        self.winsize = config.get('winsize', 15)
        self.iterations = config.get('iterations', 3)
        self.poly_n = config.get('poly_n', 5)
        self.poly_sigma = config.get('poly_sigma', 1.2)

        # Aggregation method (default: mean)
        self.use_median = config.get('use_median', False)

        # Patch-based measurement (default: enabled)
        self.use_patches = config.get('use_patches', True)
        self.patch_rows = config.get('patch_rows', 3)
        self.patch_cols = config.get('patch_cols', 3)

        # Radial consistency validation (always enabled)
        from src.validation import RadialConsistencyValidator
        validation_config = config.get('validation', {})
        self.validator = RadialConsistencyValidator(validation_config)

        print("✓ OpticalFlowDivergenceRobustMeasurement initialized")
        print("  → Radial consistency validation enabled")

    def _remove_affine_motion(self, u_x, u_y, mask):
        """Helper to estimate and subtract 2D affine rigid motion."""
        h, w = u_x.shape
        y, x = np.indices((h, w))
        
        mask_idx = mask > 0
        if not np.any(mask_idx):
            return u_x - np.median(u_x), u_y - np.median(u_y)

        # Features: [x, y, 1] for the affine model u = ax + by + c
        features = np.stack([x[mask_idx], y[mask_idx], np.ones_like(x[mask_idx])], axis=1)
        targets_x = u_x[mask_idx]
        targets_y = u_y[mask_idx]
        
        # Solve least squares for x and y flow independently
        sol_x, _, _, _ = np.linalg.lstsq(features, targets_x, rcond=None)
        sol_y, _, _, _ = np.linalg.lstsq(features, targets_y, rcond=None)
        
        # Reconstruct the global rigid field
        full_features = np.stack([x.flatten(), y.flatten(), np.ones(h*w)], axis=1)
        rigid_u_x = (full_features @ sol_x).reshape(h, w)
        rigid_u_y = (full_features @ sol_y).reshape(h, w)
        
        return u_x - rigid_u_x, u_y - rigid_u_y

    def measure(self, chest_region: np.ndarray, removal_type: str="affine"):
        """
        Args:
            chest_region: current mask/image area
            removal_type: 'median' for simple translation, 'affine' for translation+rotation+scaling

        Returns:
            tuple: (breathing_value, metadata_dict)
                - breathing_value (float): The measured breathing signal value
                - metadata_dict (dict): Contains quality info and diagnostics
        """
        if self.prev_chest is None or self.prev_chest.shape != chest_region.shape:
            # Build metadata for invalid measurement
            metadata = {
                'quality': 'invalid',
                'reason': 'shape_mismatch' if self.prev_chest is not None else 'initialization',
                'prev_shape': None if self.prev_chest is None else self.prev_chest.shape,
                'curr_shape': chest_region.shape
            }

            if self.prev_chest is None:
                print("self.prev_chest is None")
            else:
                print(f"inconsistent prev and new chest regions!!!: prev_chest.shape ({self.prev_chest.shape}) != chest_region.shape ({chest_region.shape})")

            self.prev_chest = chest_region.copy()
            return 0.0, metadata

        # 1. Optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_chest, chest_region, None,
            pyr_scale=self.pyr_scale, levels=self.levels, winsize=self.winsize,
            iterations=self.iterations, poly_n=self.poly_n, poly_sigma=self.poly_sigma,
            flags=0
        )

        u_x = flow[..., 0]
        u_y = flow[..., 1]

        # 2. Rigid motion removal (Optional types)
        if removal_type == "affine":
            # Mask is used to identify the 'support' pixels for the rigid body
            # We assume the bird/hand block moves rigidly together
            mask = (chest_region > 0)
            u_x_res, u_y_res = self._remove_affine_motion(u_x, u_y, mask)
        else:
            # Fallback to your original median subtraction (Translation only)
            u_x_res = u_x - np.median(u_x)
            u_y_res = u_y - np.median(u_y)

        # 3. Generate Signal Map using Radial Projection (Expansion/Contraction)
        h, w = u_x.shape
        y, x = np.indices((h, w))

        # Center calculations
        # --- ENERGY CENTER CALCULATION ---
        if self.center_type=="energy_center":
            # Calculate magnitude of motion to find the "active" center
            mag = np.sqrt(u_x_res**2 + u_y_res**2)
            total_mag = np.sum(mag)

            if total_mag > 1e-6:
                # Weighted center of motion (Energy Center)
                center_x = np.sum(x * mag) / total_mag
                center_y = np.sum(y * mag) / total_mag
            else:
                # Fallback to geometric center if no motion detected
                center_x, center_y = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            #print("USING GEOMETRIC CENTER!")
            center_x, center_y = (w - 1) / 2.0, (h - 1) / 2.0

        rel_x, rel_y = x - center_x, y - center_y
        dist = np.sqrt(rel_x**2 + rel_y**2) + 1e-9
        unit_rx, unit_ry = rel_x / dist, rel_y / dist

        signal_map = (u_x_res * unit_rx) + (u_y_res * unit_ry)

        # 4. Patch-wise Aggregation
        if not self.use_patches:
            breathing = np.median(signal_map) if self.use_median else np.mean(signal_map)
        else:
            h, w = signal_map.shape
            ph, pw = h // self.patch_rows, w // self.patch_cols
            patch_values = []

            for r in range(self.patch_rows):
                for c in range(self.patch_cols):
                    patch = signal_map[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
                    if patch.size == 0: continue

                    val = np.median(patch) if self.use_median else np.mean(patch)
                    patch_values.append(val)

            # Robust consensus: the median value across all patches
            breathing = np.median(patch_values) if patch_values else 0.0

            # 1. Convert to absolute values to find the "loudest" patches
            # We use absolute because breathing is an oscillation (+/-)
            abs_values = np.abs(patch_values)
            
            # 2. Get indices of the top 3 most active patches
            top_indices = np.argsort(abs_values)[-3:] 
            
            # 3. Average those top 3 (original values, not absolute)
            breathing = np.mean([patch_values[i] for i in top_indices])

        # 5. Radial consistency validation
        # Rejects non-breathing motion artifacts (hand movements, repositioning)
        if self.validator is not None:  # Always true, kept for safety
            validated_breathing, is_valid, validator_metrics = self.validator.validate(
                u_x_res, u_y_res,
                signal_value=breathing,
                #fallback_value=self.last_valid_breathing
                fallback_value=np.nan,
                center=(center_x, center_y)
            )

            if not is_valid:
                print("Validator rejected!: ", validator_metrics)

            # Update last valid value if current frame was accepted
            if is_valid:
                self.last_valid_breathing = validated_breathing

            breathing = validated_breathing

            # Build metadata for successful measurement
            metadata = {
                'quality': 'valid' if is_valid else 'invalid',
                'shape': chest_region.shape,
                'breathing_raw': float(breathing),
                'was_validated': True,
                'validator_metrics': validator_metrics if is_valid else ''
            }

        else:
            # Safety fallback (should never happen since validator is always initialized)
            self.last_valid_breathing = breathing

            # Build metadata for successful measurement
            metadata = {
                'quality': 'valid',
                'shape': chest_region.shape,
                'breathing_raw': float(breathing),
                'was_validated': True
            }

        self.prev_chest = chest_region.copy()
        return float(breathing), metadata
    
    def post_processing(self, raw_signal):
        """
        Apply linear interpolation to handle NaNs
        
        :param self: Description
        :param raw_signal: Description
        """
        
        # 1. Convert to pandas for high-performance interpolation
        s = pd.Series(raw_signal)
        
        # 2. Linear interpolation is best for maintaining phase
        # limit_direction='both' handles cases where the video starts/ends with noise
        interp_signal = s.interpolate(method='linear', limit_direction='both').values
        
        # 3. Final safety: fill any remaining NaNs with 0
        return np.nan_to_num(interp_signal)
    
    def reset(self, init_chest_region=None):
        """Reset internal state"""
        self.prev_chest = init_chest_region
        self.last_valid_breathing = 0.0
        self.validator.reset()

def get_measurement(config: dict):
    """
    Factory function to get measurement method

    Args:
        config: Configuration dictionary

    Returns:
        OpticalFlowDivergenceRobustMeasurement instance
    """
    return OpticalFlowDivergenceRobustMeasurement(config)
