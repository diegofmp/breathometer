"""
General signal processing utilities

These are stateless, reusable signal processing functions that can be used
across different modules. They don't depend on configuration or class state.
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter


def bandpass_filter(signal_data: np.ndarray, fps: float, low_freq: float,
                   high_freq: float, order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to signal

    Args:
        signal_data: Input signal array
        fps: Sampling rate (frames per second)
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        order: Filter order (default: 4)

    Returns:
        Filtered signal
    """
    nyquist = fps / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply filter (forward-backward to avoid phase shift)
    filtered = signal.filtfilt(b, a, signal_data)

    return filtered


def remove_outliers(signal_data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove outliers using median absolute deviation (MAD) method
    Replaces outliers with median value to avoid affecting range calculations

    Args:
        signal_data: Input signal
        threshold: Number of MADs to consider as outlier (default: 3.0)

    Returns:
        Signal with outliers replaced by median
    """
    median = np.median(signal_data)
    mad = np.median(np.abs(signal_data - median))

    if mad == 0:
        # No variation, return original
        return signal_data

    # Modified z-score using MAD (more robust than standard deviation)
    modified_z_scores = 0.6745 * (signal_data - median) / mad

    # Create copy and replace outliers with median
    cleaned_signal = signal_data.copy()
    outlier_mask = np.abs(modified_z_scores) > threshold
    cleaned_signal[outlier_mask] = median

    return cleaned_signal


def apply_savgol_filter(signal_data: np.ndarray, window_size: int,
                       polyorder: int) -> np.ndarray:
    """
    Apply Savitzky-Golay filter (preserves peaks better than moving average)

    Args:
        signal_data: Input signal
        window_size: Window size (must be odd)
        polyorder: Polynomial order

    Returns:
        Filtered signal
    """
    # Ensure window size is odd and valid
    if window_size % 2 == 0:
        window_size += 1

    if window_size > len(signal_data):
        window_size = len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1

    if window_size < polyorder + 2:
        return signal_data  # Can't apply filter

    return savgol_filter(signal_data, window_size, polyorder)


def normalize_signal(signal_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal to consistent range

    Args:
        signal_data: Input signal
        method: 'zscore', 'minmax', or 'robust'

    Returns:
        Normalized signal
    """
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std > 0:
            return (signal_data - mean) / std
        return signal_data - mean

    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        if max_val > min_val:
            return (signal_data - min_val) / (max_val - min_val)
        return signal_data - min_val

    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        if mad > 0:
            return (signal_data - median) / mad
        return signal_data - median

    return signal_data


def interpolate_nans(signal_data: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Handle NaN values in signal using linear interpolation

    Args:
        signal_data: Input signal that may contain NaN values

    Returns:
        Tuple of (interpolated_signal, nan_count)
    """
    nan_count = np.sum(np.isnan(signal_data))

    if nan_count == 0:
        return signal_data, 0

    # Simple linear interpolation
    mask = ~np.isnan(signal_data)
    if np.any(mask):
        indices = np.arange(len(signal_data))
        interpolated = np.interp(indices, indices[mask], signal_data[mask])
        return interpolated, nan_count
    else:
        raise ValueError("All signal values are NaN - cannot interpolate")
