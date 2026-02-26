"""
Signal processing for breathing rate estimation
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import spectrogram as scipy_spectrogram
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from .utils import (
    bandpass_filter,
    remove_outliers,
    normalize_signal,
    interpolate_nans
)


class SignalProcessor:
    """
    Process breathing signal to estimate breathing rate
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.fps = config.get('fps', 30)

        filter_config = config.get('bandpass_filter', {})
        self.low_freq = filter_config.get('low_freq', 1.5)
        self.high_freq = filter_config.get('high_freq', 8.0)
        self.order = filter_config.get('order', 2)

        # Load preprocessing config
        self.preprocess_config = config.get('preprocessing', {})

        # Load breath counting config
        breath_config = config.get('breath_counting', {})
        self.max_breathing_rate = breath_config.get('max_breathing_rate_bpm', 360)
        self.peak_prominence_ratio = breath_config.get('peak_prominence_ratio', 0.1)
        self.counting_method = breath_config.get('method', 'autocorrelation_windowed')

        # Autocorrelation parameters
        acf_config = breath_config.get('autocorrelation', {})
        self.acf_min_bpm = acf_config.get('min_breathing_rate_bpm', 90)
        self.acf_max_bpm = acf_config.get('max_breathing_rate_bpm', 400)
        self.acf_min_prominence = acf_config.get('acf_min_prominence', 0.03)
        self.acf_peak_selection = acf_config.get('acf_peak_selection', 'first')
        self.acf_min_confidence = acf_config.get('min_confidence', 0.3)
        self.acf_low_corr_threshold = acf_config.get('low_correlation_threshold', 0.25)
        self.acf_window_size = acf_config.get('window_size', 7.0)  # seconds (for windowed version)
        self.acf_overlap = acf_config.get('overlap', 0.5)

        print(f"✓ SignalProcessor initialized (fps={self.fps}, "
              f"band={self.low_freq}-{self.high_freq} Hz, method={self.counting_method})")
    
    def estimate_breathing_rate(self, breathing_signal: np.ndarray, fps: Optional[float] = None) -> Tuple[float, dict]:
        """
        Estimate breathing rate from signal

        Args:
            breathing_signal: Array of breathing measurements
            fps: Frame rate (overrides config if provided)

        Returns:
            breathing_rate_bpm: Breathing rate in BPM
            info: Dictionary with additional information
        """
        if fps is None:
            fps = self.fps

        if len(breathing_signal) < 30:
            return 0.0, {'error': 'Signal too short'}

        # Unified preprocessing (bandpass → clip → savgol → TKEO → normalize)
        filtered_signal, preprocessing_info = self._preprocess_signal(breathing_signal, fps)

        # Count breaths using configured method (autocorrelation only)
        if self.counting_method == 'autocorrelation':
            breath_count_info = self.count_breaths_autocorrelation(breathing_signal, fps)
        else:  # autocorrelation_windowed (default)
            breath_count_info = self.count_breaths_autocorrelation_windowed(breathing_signal, fps)

        # FFT-based analysis
        fft_result = np.fft.fft(filtered_signal)
        frequencies = np.fft.fftfreq(len(filtered_signal), 1/fps)

        # Positive frequencies only
        positive_mask = frequencies > 0
        positive_freqs = frequencies[positive_mask]
        positive_fft = np.abs(fft_result[positive_mask])

        # Find peak in breathing range
        breathing_mask = (positive_freqs >= self.low_freq) & (positive_freqs <= self.high_freq)

        if not np.any(breathing_mask):
            return 0.0, {'error': 'No peaks in breathing range'}

        breathing_freqs = positive_freqs[breathing_mask]
        breathing_fft = positive_fft[breathing_mask]

        # Find dominant frequency
        peak_idx = np.argmax(breathing_fft)
        breathing_freq_hz = breathing_freqs[peak_idx]
        peak_magnitude = breathing_fft[peak_idx]

        # Use BPM from breath counting method if available
        # FFT/EMV/Adaptive methods compute their own BPM estimates
        if 'breathing_rate_bpm' in breath_count_info and breath_count_info['breathing_rate_bpm'] > 0:
            # Method provides its own BPM estimate (FFT, EMV, adaptive peak)
            breathing_rate_bpm = breath_count_info['breathing_rate_bpm']
            # Use method's confidence if available

            if 'quality' in breath_count_info:
                confidence = breath_count_info['quality'].get('overall_score', 0.0)
            else:
                confidence = breath_count_info.get('mean_confidence', 0.0)

            
        elif self.counting_method == 'adaptive_window':
            # Adaptive windowing: compute BPM from breath count
            signal_duration_minutes = len(breathing_signal) / fps / 60.0
            total_breaths = breath_count_info['breath_counts'].get('total', 0)
            breathing_rate_bpm = total_breaths / signal_duration_minutes if signal_duration_minutes > 0 else 0.0
            confidence = breath_count_info['validation'].get('mean_confidence', 0.0)
        elif 'full' in breath_count_info.get('breath_counts', {}):
            # Traditional peak method: use 'full' duration breath count
            breathing_rate_bpm = breath_count_info['breath_counts']['full']['rate_bpm']
            # Confidence score (ratio of peak to total energy in FFT)
            total_energy = np.sum(breathing_fft)
            confidence = peak_magnitude / total_energy if total_energy > 0 else 0.0
        else:
            # Fallback: compute from breath count if available
            signal_duration_minutes = len(breathing_signal) / fps / 60.0
            breath_counts_dict = breath_count_info.get('breath_counts', {})
            total_breaths = breath_counts_dict.get('total', breath_counts_dict.get('accepted', 0))
            breathing_rate_bpm = total_breaths / signal_duration_minutes if signal_duration_minutes > 0 else 0.0
            # Confidence score (ratio of peak to total energy in FFT)
            total_energy = np.sum(breathing_fft)
            confidence = peak_magnitude / total_energy if total_energy > 0 else 0.0

        info = {
            'frequency_hz': breathing_freq_hz,
            'confidence': confidence,
            'peak_magnitude': peak_magnitude,
            'preprocessed_signal': filtered_signal,   # unified pipeline output
            'filtered_signal': filtered_signal,
            'frequencies': breathing_freqs,
            'fft_magnitude': breathing_fft,
            # Add breath counting information
            'breath_counts': breath_count_info['breath_counts'],
            'breath_intervals': breath_count_info['breath_intervals'],
            'validation': breath_count_info['validation'],
            # Add preprocessing information
            'preprocessing': preprocessing_info,
        }

        # Add method-specific information
        if self.counting_method == 'autocorrelation':
            info['autocorr'] = breath_count_info.get('autocorr', np.array([]))
            info['acf_peaks'] = breath_count_info.get('acf_peaks', [])
            info['period_seconds'] = breath_count_info.get('period_seconds', 0.0)
            info['filtered_signal'] = breath_count_info.get('filtered_signal', breathing_signal)
            info['method'] = 'autocorrelation'
        elif self.counting_method == 'autocorrelation_windowed':
            info['window_estimates'] = breath_count_info.get('window_estimates', [])
            info['num_windows'] = breath_count_info.get('num_windows', 0)
            info['bpm_std'] = breath_count_info.get('bpm_std', 0.0)
            info['bpm_range'] = breath_count_info.get('bpm_range', (0.0, 0.0))
            info['mean_confidence'] = breath_count_info.get('mean_confidence', 0.0)
            info['method'] = 'autocorrelation_windowed'
            info['quality'] = breath_count_info.get('quality', {})  
        else:
            info['peak_frames'] = breath_count_info.get('peak_frames', [])
            info['method'] = 'peak'

        return breathing_rate_bpm, info

    def count_breaths_autocorrelation(
        self,
        breathing_signal: np.ndarray,
        fps: float
    ) -> dict:
        """
        Count breaths using autocorrelation periodicity detection.

        Advantages over other methods:
        - No amplitude assumptions (detects periodicity, not peak height)
        - No hard peak spacing constraints
        - Robust to outliers (averaging effect)
        - Natural periodicity detection

        Algorithm:
        1. Preprocess signal (bandpass filter, center around zero)
        2. Compute autocorrelation for all lags
        3. Find peaks in ACF within valid lag range (based on BPM range)
        4. Select first peak (fundamental frequency, avoids harmonics)
        5. Convert period to BPM

        Args:
            breathing_signal: Raw breathing signal
            fps: Frames per second

        Returns:
            dict with breath counting results including ACF data for visualization
        """
        # 1. Unified preprocessing (bandpass → clip → savgol → TKEO → normalize)
        filtered_signal, _ = self._preprocess_signal(breathing_signal, fps)
        signal_centered = filtered_signal - np.mean(filtered_signal)  # already ~0 after zscore, kept for ACF

        # 2. Compute autocorrelation
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]  # Normalize (correlation coefficient)

        # 3. Calculate valid lag range (from BPM range)
        min_lag = int((60.0 / self.acf_max_bpm) * fps)  # Shortest period
        max_lag = int((60.0 / self.acf_min_bpm) * fps)  # Longest period

        # Edge case: signal too short
        if len(breathing_signal) < max_lag:
            return {
                'breathing_rate_bpm': 0.0,
                'breath_counts': {'total': 0, 'accepted': 0, 'rejected': 0},
                'breath_intervals': [],
                'validation': {
                    'error': 'signal_too_short',
                    'min_length': max_lag,
                    'confidence': 0.0
                },
                'autocorr': autocorr,
                'valid_lags': np.array([]),
                'acf_peaks': [],
                'period_seconds': 0.0,
                'filtered_signal': filtered_signal,
            }

        valid_autocorr = autocorr[min_lag:max_lag+1]
        valid_lags = np.arange(min_lag, max_lag+1)

        # 4. Find peaks in ACF
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            valid_autocorr,
            prominence=self.acf_min_prominence,
            distance=min_lag  # Peaks must be at least min_lag apart
        )

        # Edge case: no peaks found
        if len(peaks) == 0:
            return {
                'breathing_rate_bpm': 0.0,
                'breath_counts': {'total': 0, 'accepted': 0, 'rejected': 0},
                'breath_intervals': [],
                'validation': {
                    'error': 'no_acf_peaks',
                    'confidence': 0.0,
                    'num_acf_peaks': 0
                },
                'autocorr': autocorr,
                'valid_lags': valid_lags,
                'acf_peaks': [],
                'period_seconds': 0.0,
                'filtered_signal': filtered_signal,
            }

        # 5. Select dominant peak
        # 'first' (fundamental, avoids harmonics) or 'highest' (max prominence)
        if self.acf_peak_selection == 'first':
            dominant_peak_idx = 0
        else:
            dominant_peak_idx = np.argmax(properties['prominences'])

        dominant_peak_lag = valid_lags[peaks[dominant_peak_idx]]
        dominant_peak_value = valid_autocorr[peaks[dominant_peak_idx]]
        peak_prominence = properties['prominences'][dominant_peak_idx]

        # 6. Convert to BPM
        period_seconds = dominant_peak_lag / fps
        breathing_rate_bpm = 60.0 / period_seconds

        # 7. Estimate breath count
        signal_duration_seconds = len(breathing_signal) / fps
        breath_count = int(round(breathing_rate_bpm * (signal_duration_seconds / 60.0)))

        # 8. Confidence estimation
        # Peak value (0-1) + prominence normalized
        confidence = 0.6 * dominant_peak_value + 0.4 * min(1.0, peak_prominence / 0.3)

        # Low correlation warning
        if dominant_peak_value < self.acf_low_corr_threshold:
            confidence *= 0.5

        # 9. Breath intervals (assumed uniform for periodic signal)
        breath_intervals = [period_seconds] * (breath_count - 1) if breath_count > 1 else []

        return {
            'breathing_rate_bpm': breathing_rate_bpm,
            'breath_counts': {
                'total': breath_count,
                'accepted': breath_count,
                'rejected': 0,
            },
            'breath_intervals': breath_intervals,
            'validation': {
                'method': 'autocorrelation',
                'confidence': confidence,
                'peak_lag': int(dominant_peak_lag),
                'peak_value': float(dominant_peak_value),
                'peak_prominence': float(peak_prominence),
                'num_acf_peaks': len(peaks),
            },
            # Debug info for visualization
            'autocorr': autocorr,
            'valid_lags': valid_lags,
            'acf_peaks': peaks.tolist(),
            'period_seconds': period_seconds,
            'filtered_signal': filtered_signal,
        }

    def count_breaths_autocorrelation_windowed(
        self,
        breathing_signal: np.ndarray,
        fps: float,
        window_size: Optional[float] = None,
        overlap: Optional[float] = None
    ) -> dict:
        """
        Count breaths using windowed autocorrelation for non-stationary signals.

        Handles breathing rate changes over time by analyzing overlapping
        windows independently. This addresses the limitation of basic ACF
        which assumes stationary (constant breathing rate) signals.

        Args:
            breathing_signal: Raw breathing signal
            fps: Frames per second
            window_size: Window duration in seconds (default: 15.0)
            overlap: Window overlap fraction 0-1 (default: 0.5)

        Returns:
            dict with breath counting results including per-window estimates
        """
        # Use config defaults if not specified
        if window_size is None:
            window_size = getattr(self, 'acf_window_size', 8.0)
        if overlap is None:
            overlap = getattr(self, 'acf_overlap', 0.5)

        # Preprocess
        filtered, _ = self._preprocess_signal(breathing_signal, fps)

        # Calculate window parameters
        window_samples = int(window_size * fps)
        step_samples = int(window_samples * (1 - overlap))

        if len(filtered) < window_samples:
            # Signal too short for windowing, fall back to basic ACF
            return self.count_breaths_autocorrelation(breathing_signal, fps)

        # Split into overlapping windows
        window_estimates = []
        num_windows = 0

        for start_idx in range(0, len(filtered) - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window = filtered[start_idx:end_idx]

            # Apply ACF to this window
            # Use basic ACF logic but on window
            signal_centered = window - np.mean(window)
            autocorr = np.correlate(signal_centered, signal_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] != 0:
                autocorr = autocorr / autocorr[0]

            # Calculate valid lag range from configured BPM bounds
            min_lag = max(1, int(round((60.0 / self.acf_max_bpm) * fps)))
            max_lag = int(round((60.0 / self.acf_min_bpm) * fps))

            # Extend the lower bound by exactly 1 sample so the fundamental is
            # always at position ≥ 1 in valid_autocorr: find_peaks cannot detect
            # a local maximum at position 0 because it has no left neighbour.
            # The old 0.9 factor gave a multi-sample extension, which — combined
            # with distance=0.8*min_lag — let find_peaks suppress the fundamental
            # whenever spurious ACF energy existed in that extension region.
            # A fixed 1-sample extension + distance=min_lag avoids both problems:
            # the fundamental is always visible to find_peaks, and it beats any
            # sub-period noise via the distance-deduplication (ACF[min_lag-1] is
            # on the rising slope toward the peak, so it's always shorter).
            search_min_lag = max(1, min_lag - 1)
            search_max_lag = min(len(autocorr) - 1, int(round(max_lag * 1.1)))

            if len(window) < max_lag:
                continue

            valid_autocorr = autocorr[search_min_lag : search_max_lag + 1]
            valid_lags = np.arange(search_min_lag, search_max_lag + 1)

            # Find peaks — distance=min_lag is the theoretically correct value:
            # ACF harmonics are spaced exactly min_lag apart, so this prevents
            # adjacent harmonics from both surviving while still allowing the
            # fundamental to be found at any lag in [min_lag, max_lag].
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(
                valid_autocorr,
                prominence=self.acf_min_prominence,
                distance=min_lag
            )

            if len(peaks) == 0:
                continue  # Skip windows with no peaks

            # Select first peak (fundamental)
            # dominant_peak_idx = 0 if self.acf_peak_selection == 'first' else np.argmax(properties['prominences'])
            # dominant_peak_lag = valid_lags[peaks[dominant_peak_idx]]
            # dominant_peak_value = valid_autocorr[peaks[dominant_peak_idx]]
            # peak_prominence = properties['prominences'][dominant_peak_idx]

            # Select peak index
            dominant_peak_idx = 0 if self.acf_peak_selection == 'first' else np.argmax(properties['prominences'])
            p_idx = peaks[dominant_peak_idx]  # This is the index in valid_autocorr
            
            # --- PARABOLIC INTERPOLATION ---
            # We look at the neighbor points in valid_autocorr to find the sub-pixel peak
            if 0 < p_idx < len(valid_autocorr) - 1:
                y1 = valid_autocorr[p_idx - 1]
                y2 = valid_autocorr[p_idx]
                y3 = valid_autocorr[p_idx + 1]
                
                # Quadratic peak formula: find the fractional offset from the center
                # offset will be between -0.5 and +0.5
                denom = (2 * y2 - y1 - y3)
                if denom != 0:
                    offset = (y3 - y1) / (2 * denom)
                else:
                    offset = 0.0
            else:
                offset = 0.0

            # Calculate the precise lag (fractional)
            dominant_peak_lag = valid_lags[p_idx] + offset
            # -------------------------------

            dominant_peak_value = valid_autocorr[p_idx]
            peak_prominence = properties['prominences'][dominant_peak_idx]

            # Convert to BPM
            period_seconds = dominant_peak_lag / fps
            window_bpm = 60.0 / period_seconds

            # Calculate confidence
            confidence = 0.6 * dominant_peak_value + 0.4 * min(1.0, peak_prominence / 0.3)
            if dominant_peak_value < self.acf_low_corr_threshold:
                confidence *= 0.5

            window_estimates.append({
                'bpm': window_bpm,
                'confidence': confidence,
                'start_time': start_idx / fps,
                'end_time': end_idx / fps,
                'peak_value': float(dominant_peak_value),
                'peak_prominence': float(peak_prominence),
                'num_peaks': len(peaks)
            })
            num_windows += 1

        if len(window_estimates) == 0:
            # No valid windows
            return {
                'breathing_rate_bpm': 0.0,
                'breath_counts': {'total': 0, 'accepted': 0, 'rejected': 0},
                'breath_intervals': [],
                'validation': {
                    'error': 'no_valid_windows',
                    'confidence': 0.0,
                    'num_windows': 0
                },
                'window_estimates': [],
                'num_windows': 0,
            }

        # Aggregate window estimates using confidence-weighted KDE mode
        bpms = np.array([w['bpm'] for w in window_estimates])
        confidences = np.array([w['confidence'] for w in window_estimates])

        # Confidence-weighted KDE mode: finds where estimate density is highest,
        # not just the 50th percentile. Robust when confidences are similar but
        # BPMs cluster in distinct groups.
        if len(bpms) > 0:
            bandwidth = max(10.0, np.std(bpms) * 0.3)

            # Handle case where all BPMs are identical
            if bpms.min() == bpms.max():
                breathing_rate_bpm = float(bpms[0])
                kde_concentration = 1.0  # Perfect agreement
                kde_second_peak_ratio = 0.0  # No competing peaks
            else:
                bpm_grid = np.linspace(bpms.min(), bpms.max(), 500)
                grid_step = bpm_grid[1] - bpm_grid[0]

                # Ensure grid_step is not zero (shouldn't happen but safeguard)
                if grid_step == 0:
                    breathing_rate_bpm = float(bpms[0])
                    kde_concentration = 1.0
                    kde_second_peak_ratio = 0.0
                else:
                    kde = np.sum(
                        confidences[:, None] * np.exp(-0.5 * ((bpm_grid[None, :] - bpms[:, None]) / bandwidth) ** 2),
                        axis=0
                    )
                    peak_idx = np.argmax(kde)
                    breathing_rate_bpm = bpm_grid[peak_idx]

                    # How concentrated is the mass around the peak (0=flat/uncertain, ~1=sharp/confident)
                    kde_concentration = float(1.0 - kde.mean() / kde[peak_idx]) if kde[peak_idx] > 0 else 0.0

                    # Second-peak ratio: mask out the main peak region, find next peak
                    mask_radius = int(3.0 * bandwidth / grid_step)
                    kde_masked = kde.copy()
                    kde_masked[max(0, peak_idx - mask_radius):peak_idx + mask_radius + 1] = 0.0
                    kde_second_peak_ratio = float(kde_masked.max() / kde[peak_idx]) if kde[peak_idx] > 0 else 0.0
        else:
            breathing_rate_bpm = 0.0
            kde_concentration = 0.0
            kde_second_peak_ratio = 0.0

        # Calculate breath count
        signal_duration_seconds = len(breathing_signal) / fps
        breath_count = int(round(breathing_rate_bpm * (signal_duration_seconds / 60.0)))

        # Calculate statistics
        bpm_std = np.std(bpms) if len(bpms) > 1 else 0.0
        bpm_range = (np.min(bpms), np.max(bpms)) if len(bpms) > 0 else (0.0, 0.0)
        mean_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0

        # Breath intervals (assumed uniform based on aggregated rate)
        if breathing_rate_bpm > 0:
            period_seconds = 60.0 / breathing_rate_bpm
            breath_intervals = [period_seconds] * (breath_count - 1) if breath_count > 1 else []
        else:
            breath_intervals = []

        # ===== QUALITY ASSESSMENT =====
        # Calculate component quality scores (0-1 scale)

        # 1. Confidence score (already 0-1)
        confidence_score = mean_confidence

        # 2. Stability score (based on coefficient of variation)
        cv = bpm_std / breathing_rate_bpm if breathing_rate_bpm > 0 else 1.0
        stability_score = max(0.0, 1.0 - (cv / 0.3))  # 30% CV = 0 score

        # 3. Range consistency score
        range_spread = bpm_range[1] - bpm_range[0] if bpm_range[1] > 0 else 0
        relative_spread = range_spread / breathing_rate_bpm if breathing_rate_bpm > 0 else 1.0
        range_score = max(0.0, 1.0 - (relative_spread / 0.5))  # 50% spread = 0 score

        # 4. Coverage score (minimum 3 windows expected for decent signal)
        coverage_score = min(1.0, num_windows / 3.0) if num_windows > 0 else 0.0

        # 6. KDE Sharpness score (1.0 is a sharp peak, 0.0 is flat)
        # We want to reward high concentration
        kde_sharpness_score = kde_concentration 

        # 7. Multi-modal penalty (1.0 is no other peaks, 0.0 is a huge second peak)
        # This directly attacks the "Octave Jump" problem
        kde_ambiguity_penalty = 1.0 - kde_second_peak_ratio

        # ===== UPDATED OVERALL QUALITY =====
        quality_score = (
            0.40 * confidence_score +      # Base signal strength
            0.20 * kde_sharpness_score +   # How "certain" the mode is
            0.20 * kde_ambiguity_penalty + # Is there a competing harmonic?
            0.10 * stability_score +       # General consistency
            0.10 * coverage_score          # Data quantity
        )

        # # 5. Weighted overall quality score
        # quality_score = (
        #     0.50 * confidence_score +      # 50% weight - most important
        #     0.20 * stability_score +       # 20% weight
        #     0.20 * range_score +           # 20% weight
        #     0.10 * coverage_score          # 10% weight
        # )

        # Quality label and recommendation
        if quality_score >= 0.75:
            quality_label = "EXCELLENT"
            recommendation = "High confidence - use result"
        elif quality_score >= 0.60:
            quality_label = "GOOD"
            recommendation = "Reliable - use result"
        elif quality_score >= 0.40:
            quality_label = "FAIR"
            recommendation = "Moderate confidence - verify if critical"
        elif quality_score >= 0.25:
            quality_label = "POOR"
            recommendation = "Low confidence - manual review recommended"
        else:
            quality_label = "VERY POOR"
            recommendation = "Unreliable - reject or re-measure"

        return {
            'breathing_rate_bpm': breathing_rate_bpm,
            'breath_counts': {
                'total': breath_count,
                'accepted': breath_count,
                'rejected': 0,
            },
            'breath_intervals': breath_intervals,
            'validation': {
                'method': 'autocorrelation_windowed',
                'confidence': mean_confidence,
                'num_windows': num_windows,
                'bpm_std': bpm_std,
                'bpm_range': bpm_range,
            },
            'window_estimates': window_estimates,
            'num_windows': num_windows,
            'bpm_std': bpm_std,
            'bpm_range': bpm_range,
            'mean_confidence': mean_confidence,
            # Quality assessment
            'quality': {
                'overall_score': float(quality_score),
                'label': quality_label,
                'recommendation': recommendation,
                'components': {
                    'confidence': float(confidence_score),
                    'stability': float(stability_score),
                    'range_consistency': float(range_score),
                    'coverage': float(coverage_score),
                },
                'metrics': {
                    'mean_confidence': float(mean_confidence),
                    'bpm_std': float(bpm_std),
                    'cv': float(cv),
                    'relative_spread': float(relative_spread),
                    'num_windows': int(num_windows),
                    'bpm_range': (float(bpm_range[0]), float(bpm_range[1])),
                    'kde_concentration': kde_concentration,      # 0=flat/uncertain, ~1=sharp/confident
                    'kde_second_peak_ratio': kde_second_peak_ratio,  # >0.5 = competing cluster
                }
            }
        }

    def _bandpass_filter(self, signal_data: np.ndarray, fps: float) -> np.ndarray:
        """Apply bandpass filter to signal using instance config"""
        return bandpass_filter(signal_data, fps, self.low_freq, self.high_freq, self.order)

    def _remove_outliers(self, signal_data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using MAD method - delegates to utility function"""
        return remove_outliers(signal_data, threshold)

    def _normalize(self, signal_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normalize signal - delegates to utility function"""
        return normalize_signal(signal_data, method)

    def _preprocess_signal(self, raw_signal: np.ndarray, fps: float) -> Tuple[np.ndarray, dict]:
        """
        Production Pipeline for Optical Flow Divergence:
        1. Handle NaN values
        2. Bandpass (Clean velocity)
        3. Integrate (Velocity -> Displacement/Size)
        4. Second Bandpass (Remove integration drift)
        5. Outlier clip (Protect scale)
        6. Normalize (Z-score)
        """
        info = {'raw': raw_signal.copy()}

        # 1. Handle NaN values with linear interpolation
        if np.any(np.isnan(raw_signal)):
            try:
                raw_signal, nan_count = interpolate_nans(raw_signal)
                print(f"  Warning: {nan_count} NaN values in signal ({nan_count/len(raw_signal)*100:.1f}%)")
            except ValueError as e:
                print(f"✗ Error: {e}")
                return None, {}

        # 2. Bandpass — Remove non-biological wiggles first
        processed = self._bandpass_filter(raw_signal, fps)
        info['after_bandpass'] = processed.copy()

        # 3. INTEGRATION: Convert Velocity to Displacement
        # This merges the 'in' and 'out' spikes into a single 'breath' hump
        processed = np.cumsum(processed)
        info['after_integration'] = processed.copy()

        # 4. Second bandpass to clear integration drift
        processed = self._bandpass_filter(processed, fps)
        info['after_second_bandpass'] = processed.copy()

        # 5. Outlier clip — Cap spikes so they don't dominate
        clip_config = self.preprocess_config.get('outlier_clip', {})
        limit = np.std(processed) * clip_config.get('std_threshold', 3.0)
        processed = np.clip(processed, -limit, limit)
        info['after_clip'] = processed.copy()

        # 6. Normalize
        normalize_config = self.preprocess_config.get('normalize', {})
        if normalize_config.get('enabled', True):
            method = normalize_config.get('method', 'zscore')
            processed = self._normalize(processed, method)
            info['after_normalize'] = processed.copy()

        return processed, info

    def plot_analysis(self, breathing_signal: np.ndarray, save_path: Optional[str] = None,
                     show_outliers: bool = True):
        """
        Plot signal analysis with outlier visualization

        Args:
            breathing_signal: Raw breathing signal
            save_path: Optional path to save plot
            show_outliers: Whether to show outlier detection (default: True)
        """
        

        breathing_rate, info = self.estimate_breathing_rate(breathing_signal)

        # Check if preprocessing was applied
        has_preprocessing = 'preprocessing' in info and len(info['preprocessing']) > 1

        # Determine number of subplots based on options
        n_plots = 4 if show_outliers else 3
        if has_preprocessing:
            n_plots += 1  # Add extra plot for preprocessing steps
        n_plots += 1  # Add spectrogram panel

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Raw signal
        axes[plot_idx].plot(breathing_signal, linewidth=1, color='#2E86AB')
        axes[plot_idx].set_title('Raw Breathing Signal', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Frame')
        axes[plot_idx].set_ylabel('Magnitude')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Preprocessing steps (if enabled)
        if has_preprocessing:
            preprocessing = info['preprocessing']
            colors = ['#2E86AB', '#A23E48', '#FFA630', '#06A77D', '#6A4C93']
            color_idx = 0

            for step_name, step_signal in preprocessing.items():
                if step_name == 'raw':
                    continue  # Skip raw, already shown above

                label = step_name.replace('after_', '').replace('_', ' ').title()
                axes[plot_idx].plot(step_signal, linewidth=1.5,
                                   color=colors[color_idx % len(colors)],
                                   label=label, alpha=0.8)
                color_idx += 1

            axes[plot_idx].set_title('Preprocessing Steps', fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Frame')
            axes[plot_idx].set_ylabel('Magnitude')
            axes[plot_idx].legend(loc='upper right')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Filtered signal with outlier detection
        if 'filtered_signal' in info and show_outliers:
            filtered_signal = info['filtered_signal']
            cleaned_signal = self._remove_outliers(filtered_signal)

            # Detect outliers
            outlier_mask = np.abs(filtered_signal - cleaned_signal) > 0.01
            num_outliers = np.sum(outlier_mask)

            # Plot filtered signal
            axes[plot_idx].plot(filtered_signal, linewidth=1, color='#0066CC',
                               alpha=0.7, label='Filtered Signal')

            # Highlight outliers
            if num_outliers > 0:
                outlier_indices = np.where(outlier_mask)[0]
                outlier_values = filtered_signal[outlier_mask]
                axes[plot_idx].scatter(outlier_indices, outlier_values,
                                      color='red', s=50, zorder=5, marker='x',
                                      linewidth=2, label=f'Outliers ({num_outliers})')

                # Plot cleaned signal
                axes[plot_idx].plot(cleaned_signal, linewidth=1, color='green',
                                   alpha=0.5, linestyle='--', label='Cleaned (for thresholds)')

            axes[plot_idx].set_title(f'Filtered Signal ({self.low_freq}-{self.high_freq} Hz) with Outlier Detection',
                                    fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Frame')
            axes[plot_idx].set_ylabel('Magnitude')
            axes[plot_idx].legend(loc='upper right')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        elif 'filtered_signal' in info:
            # Just show filtered signal without outlier analysis
            axes[plot_idx].plot(info['filtered_signal'], linewidth=1, color='#0066CC')
            axes[plot_idx].set_title(f'Filtered Signal ({self.low_freq}-{self.high_freq} Hz)',
                                    fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Frame')
            axes[plot_idx].set_ylabel('Magnitude')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Filtered signal with detected peaks
        if 'filtered_signal' in info and 'peak_frames' in info:
            filtered_signal = info['filtered_signal']
            peak_frames = info['peak_frames']

            axes[plot_idx].plot(filtered_signal, linewidth=1, color='#0066CC', alpha=0.6)

            if len(peak_frames) > 0:
                peak_values = filtered_signal[peak_frames]
                axes[plot_idx].scatter(peak_frames, peak_values,
                                      color='#06A77D', s=60, zorder=5, marker='v',
                                      label=f'Detected Peaks ({len(peak_frames)})')

            axes[plot_idx].set_title('Filtered Signal with Detected Breath Peaks',
                                    fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Frame')
            axes[plot_idx].set_ylabel('Magnitude')
            axes[plot_idx].legend(loc='upper right')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # FFT
        if 'frequencies' in info and 'fft_magnitude' in info:
            axes[plot_idx].plot(info['frequencies'], info['fft_magnitude'],
                               linewidth=1.5, color='#F77F00')
            axes[plot_idx].axvline(info['frequency_hz'], color='r', linestyle='--',
                           linewidth=2, label=f"Peak: {breathing_rate:.1f} BPM")
            axes[plot_idx].set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Frequency (Hz)')
            axes[plot_idx].set_ylabel('Magnitude')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlim([0, self.high_freq + 0.5])
            plot_idx += 1

        # Spectrogram
        if 'filtered_signal' in info:
            filtered_signal = info['filtered_signal']

            # Generate spectrogram
            f, t, Sxx = scipy_spectrogram(
                filtered_signal,
                fs=self.fps,
                window='hann',
                nperseg=min(256, len(filtered_signal)//4),  # Window size
                noverlap=None,  # 50% overlap (default)
                scaling='density'
            )

            # Filter to breathing range
            freq_mask = (f >= self.low_freq) & (f <= self.high_freq)
            f_breathing = f[freq_mask]
            Sxx_breathing = Sxx[freq_mask, :]

            # Plot as heatmap (convert to dB for better visualization)
            im = axes[plot_idx].pcolormesh(
                t, f_breathing, 10 * np.log10(Sxx_breathing + 1e-10),
                shading='gouraud',
                cmap='viridis'
            )

            # Add colorbar
            plt.colorbar(im, ax=axes[plot_idx], label='Power (dB)')

            # Mark average breathing rate from peak counting
            if 'breath_counts' in info and 'full' in info['breath_counts']:
                avg_rate_hz = info['breath_counts']['full']['rate_bpm'] / 60
                axes[plot_idx].axhline(avg_rate_hz, color='red', linestyle='--',
                                      linewidth=2, alpha=0.8,
                                      label=f'Peak Count: {avg_rate_hz*60:.1f} BPM')
                axes[plot_idx].legend(loc='upper right', fontsize=9)

            axes[plot_idx].set_xlabel('Time (s)', fontsize=11)
            axes[plot_idx].set_ylabel('Frequency (Hz)', fontsize=11)
            axes[plot_idx].set_title('Spectrogram (Time-Frequency Analysis)',
                                     fontsize=12, fontweight='bold')
            axes[plot_idx].set_ylim([self.low_freq, self.high_freq])
            axes[plot_idx].grid(True, alpha=0.2, color='white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    @staticmethod
    def print_quality_report(result: dict, detail_level: str = 'moderate'):
        """
        Print a formatted quality report for ACF windowed results

        Args:
            result: Result dictionary from count_breaths_autocorrelation_windowed
            detail_level: 'simple', 'moderate', or 'full'
        """
        if 'quality' not in result:
            print("⚠ Quality metrics not available in result")
            return

        quality = result['quality']
        bpm = result.get('breathing_rate_bpm', 0.0)

        if detail_level == 'simple':
            # Simple format for non-technical users
            stars = '★' * int(quality['overall_score'] * 5)
            print(f"Breathing Rate: {bpm:.1f} BPM")
            print(f"Quality: {stars} {quality['label']} ({quality['overall_score']*100:.0f}% confidence)")

        elif detail_level == 'moderate':
            # Moderate detail for lab technicians
            metrics = quality['metrics']
            bpm_std = metrics['bpm_std']
            cv = metrics['cv']
            num_windows = metrics['num_windows']

            print("=" * 50)
            print(f"Breathing Rate: {bpm:.1f} ± {bpm_std:.1f} BPM")
            print(f"Quality: {quality['label']} ({quality['overall_score']*100:.0f}% overall)")
            print(f"Confidence: {metrics['mean_confidence']*100:.0f}%")
            print(f"Stability: {'High' if cv < 0.1 else 'Moderate' if cv < 0.2 else 'Low'} (CV: {cv*100:.1f}%)")
            print(f"Windows Analyzed: {num_windows}")
            print(f"Recommendation: {quality['recommendation']}")
            print("=" * 50)

        elif detail_level == 'full':
            # Full detail for researchers
            metrics = quality['metrics']
            components = quality['components']

            print("\n" + "=" * 60)
            print("=== BREATHING RATE MEASUREMENT QUALITY REPORT ===")
            print("=" * 60)
            print(f"\nEstimate: {bpm:.2f} BPM")
            print(f"Method: ACF Windowed")

            print(f"\n--- Overall Quality ---")
            print(f"Quality Score: {quality['overall_score']:.3f} / 1.000")
            print(f"Quality Label: {quality['label']}")
            print(f"Recommendation: {quality['recommendation']}")

            print(f"\n--- Component Scores ---")
            print(f"  Periodicity Strength: {components['confidence']:.3f}")
            print(f"  Measurement Stability: {components['stability']:.3f}")
            print(f"  Range Consistency: {components['range_consistency']:.3f}")
            print(f"  Window Coverage: {components['coverage']:.3f}")

            print(f"\n--- Detailed Metrics ---")
            print(f"  Mean Confidence: {metrics['mean_confidence']:.3f}")
            print(f"  BPM Std Dev: {metrics['bpm_std']:.2f} BPM")
            print(f"  Coefficient of Variation: {metrics['cv']*100:.2f}%")
            print(f"  BPM Range: {metrics['bpm_range'][0]:.1f} - {metrics['bpm_range'][1]:.1f} BPM")
            print(f"  Relative Spread: {metrics['relative_spread']*100:.1f}%")
            print(f"  Valid Windows: {metrics['num_windows']}")

            print("\n" + "=" * 60)

        else:
            print(f"⚠ Unknown detail_level: {detail_level}")
            print("   Valid options: 'simple', 'moderate', 'full'")
