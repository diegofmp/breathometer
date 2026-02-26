"""
Configuration validation utilities.

Provides sanity checks for configuration parameters to ensure required
dependencies are met and prevent runtime errors.

Usage:
    >>> from src.utils import validate_config, ConfigValidationError
    >>> import yaml
    >>> from pathlib import Path
    >>>
    >>> config_path = Path('configs/default.yaml')
    >>> with open(config_path) as f:
    ...     config = yaml.safe_load(f)
    >>>
    >>> try:
    ...     validate_config(config, config_path)
    ... except ConfigValidationError as e:
    ...     print(f"Invalid config: {e}")

Adding new validation rules:
    1. Create a new validation function (e.g., validate_xyz_config)
    2. Add your checks and raise ConfigValidationError with clear messages
    3. Call your function from validate_config()
"""

from pathlib import Path
from typing import Dict, Any, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_roi_localization_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Validate ROI localization configuration parameters.

    Args:
        config: Full configuration dictionary
        config_path: Optional path to config file (for better error messages)

    Raises:
        ConfigValidationError: If configuration is invalid or has missing required parameters
    """
    config_name = f" in {config_path}" if config_path else ""

    roi_loc = config.get('roi_localization', {})
    mode = roi_loc.get('mode', '').lower()

    # Check roi_localization.mode is valid
    valid_modes = ['auto', 'manual']
    if mode not in valid_modes:
        raise ConfigValidationError(
            f"Invalid roi_localization.mode='{roi_loc.get('mode')}'{config_name}. "
            f"Must be one of: {valid_modes}"
        )

    # If mode is 'auto', bird_detector.model_path must be set
    if mode == 'auto':
        bird_det = roi_loc.get('bird_detector', {})
        model_path = bird_det.get('model_path')

        if not model_path:
            raise ConfigValidationError(
                f"Configuration error{config_name}: "
                f"roi_localization.mode='auto' requires bird_detector.model_path to be set. "
                f"Please specify the path to your bird detection model."
            )

        # Check if the model file exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise ConfigValidationError(
                f"Configuration error{config_name}: "
                f"bird_detector.model_path='{model_path}' does not exist. "
                f"Please provide a valid path to the bird detection model."
            )


def validate_tracking_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Validate tracking configuration parameters.

    Args:
        config: Full configuration dictionary
        config_path: Optional path to config file (for better error messages)

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    config_name = f" in {config_path}" if config_path else ""

    tracking = config.get('tracking', {})

    # Validate redetect_interval is non-negative
    redetect_interval = tracking.get('redetect_interval')
    if redetect_interval is not None and redetect_interval < 0:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"tracking.redetect_interval must be >= 0, got {redetect_interval}"
        )

    # Validate start_frame is non-negative
    start_frame = tracking.get('start_frame')
    if start_frame is not None and start_frame < 0:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"tracking.start_frame must be >= 0, got {start_frame}"
        )

    # Validate max_frames is positive if set
    max_frames = tracking.get('max_frames')
    if max_frames is not None and max_frames <= 0:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"tracking.max_frames must be > 0 or null, got {max_frames}"
        )


def validate_signal_processing_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Validate signal processing configuration parameters.

    Args:
        config: Full configuration dictionary
        config_path: Optional path to config file (for better error messages)

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    config_name = f" in {config_path}" if config_path else ""

    sig_proc = config.get('signal_processing', {})

    # Validate FPS
    fps = sig_proc.get('fps')
    if fps is not None and fps <= 0:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"signal_processing.fps must be > 0, got {fps}"
        )

    # Validate bandpass filter frequencies
    bandpass = sig_proc.get('bandpass_filter', {})
    low_freq = bandpass.get('low_freq')
    high_freq = bandpass.get('high_freq')

    if low_freq is not None and low_freq < 0:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"signal_processing.bandpass_filter.low_freq must be >= 0, got {low_freq}"
        )

    if high_freq is not None and high_freq < 0:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"signal_processing.bandpass_filter.high_freq must be >= 0, got {high_freq}"
        )

    if low_freq is not None and high_freq is not None and low_freq >= high_freq:
        raise ConfigValidationError(
            f"Configuration error{config_name}: "
            f"signal_processing.bandpass_filter.low_freq ({low_freq}) must be "
            f"< high_freq ({high_freq})"
        )


def validate_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Perform comprehensive validation of configuration parameters.

    This is the main entry point for config validation. It runs all validation
    checks and raises an error if any issues are found.

    Args:
        config: Full configuration dictionary
        config_path: Optional path to config file (for better error messages)

    Raises:
        ConfigValidationError: If any validation check fails

    Example:
        >>> import yaml
        >>> from pathlib import Path
        >>>
        >>> config_path = Path('configs/default.yaml')
        >>> with open(config_path) as f:
        ...     config = yaml.safe_load(f)
        >>>
        >>> validate_config(config, config_path)
    """
    validate_roi_localization_config(config, config_path)
    validate_tracking_config(config, config_path)
    validate_signal_processing_config(config, config_path)
