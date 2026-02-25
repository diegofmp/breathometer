"""
Signal processing for breathing rate estimation
"""

import numpy as np
from scipy import signal
from scipy import stats as scipy_stats
from scipy.signal import spectrogram as scipy_spectrogram
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from .utils import (
    bandpass_filter,
    remove_outliers,
    apply_savgol_filter,
    normalize_signal,
    nonlinear_amplification,
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
        self.high_freq = filter_config.get('high_freq', 5.0)
        self.order = filter_config.get('order', 4)

        # Load preprocessing config
        self.preprocess_config = config.get('preprocessing', {})

        # Load breath counting config
        breath_config = config.get('breath_counting', {})
        self.max_breathing_rate = breath_config.get('max_breathing_rate_bpm', 360)
        self.peak_prominence_ratio = breath_config.get('peak_prominence_ratio', 0.1)
        self.counting_method = breath_config.get('method', 'peak')  # 'peak', 'emv_phase', or 'adaptive_window'

        # EMV phase-based parameters
        emv_config = breath_config.get('emv_phase', {})
        self.emv_min_phase_duration = emv_config.get('min_phase_duration', 0.1)  # seconds
        self.emv_zero_crossing_threshold = emv_config.get('zero_crossing_threshold', 0.0)  # signal units
        self.emv_min_cycle_amplitude = emv_config.get('min_cycle_amplitude', 0.1)  # fraction of signal range
        self.emv_ie_ratio_range = emv_config.get('ie_ratio_range', [0.2, 5.0])  # [min, max]

        # Adaptive windowing parameters
        adaptive_config = breath_config.get('adaptive_window', {})
        self.adaptive_window_size = adaptive_config.get('window_size', 10.0)  # seconds
        self.adaptive_overlap = adaptive_config.get('overlap', 0.5)  # 50% overlap
        self.adaptive_base_method = adaptive_config.get('base_method', 'emv_phase')  # method to use per window
        self.adaptive_aggregation = adaptive_config.get('aggregation', 'confidence_weighted')  # 'median', 'mean', or 'confidence_weighted'
        self.adaptive_min_window_breaths = adaptive_config.get('min_window_breaths', 2)  # minimum expected breaths per window

        # FFT-based parameters
        fft_config = breath_config.get('fft_frequency', {})
        self.fft_window_size = fft_config.get('window_size', 15.0)  # seconds (for windowed version)
        self.fft_overlap = fft_config.get('overlap', 0.5)  # 50% overlap
        self.fft_min_snr = fft_config.get('min_snr', 3.0)  # Minimum SNR to trust estimate
        self.fft_wiener_scaling = fft_config.get('wiener_scaling', True)  # Apply sqrt(f) scaling

        # Adaptive peak detection parameters
        adaptive_peak_config = breath_config.get('peak_adaptive', {})
        self.adaptive_peak_rolling_window = adaptive_peak_config.get('rolling_window_breaths', 20)  # Number of recent breaths
        self.adaptive_peak_mad_multiplier = adaptive_peak_config.get('mad_multiplier', 4.0)  # More permissive than global (was 3.0)

        # Autocorrelation parameters
        acf_config = breath_config.get('autocorrelation', {})
        self.acf_min_bpm = acf_config.get('min_breathing_rate_bpm', 90)
        self.acf_max_bpm = acf_config.get('max_breathing_rate_bpm', 312)
        self.acf_min_prominence = acf_config.get('acf_min_prominence', 0.15)
        self.acf_peak_selection = acf_config.get('acf_peak_selection', 'first')
        self.acf_min_confidence = acf_config.get('min_confidence', 0.3)
        self.acf_low_corr_threshold = acf_config.get('low_correlation_threshold', 0.3)
        self.acf_window_size = acf_config.get('window_size', 15.0)  # seconds (for windowed version)
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
        
        if self.counting_method == "fft_windowed":
            filtered_signal, preprocessing_info = self._preprocess_for_fft(breathing_signal, fps)
        else:
            # Unified preprocessing (bandpass → clip → savgol → TKEO → normalize)
            filtered_signal, preprocessing_info = self._preprocess_signal(breathing_signal, fps)

        # Count breaths using configured method (use raw signal to preserve peaks)
        if self.counting_method == 'emv_phase':
            breath_count_info = self.count_breaths_emv(breathing_signal, fps)
        elif self.counting_method == 'fft_frequency':
            breath_count_info = self.count_breaths_fft(breathing_signal, fps)
        elif self.counting_method == 'fft_windowed':
            #breath_count_info = self.count_breaths_fft_windowed(breathing_signal, fps)
            breath_count_info = self.count_breaths_fft_windowed(filtered_signal, fps)
        elif self.counting_method == 'autocorrelation':
            breath_count_info = self.count_breaths_autocorrelation(breathing_signal, fps)
        elif self.counting_method == 'autocorrelation_windowed':
            breath_count_info = self.count_breaths_autocorrelation_windowed(breathing_signal, fps)
        else:
            breath_count_info = self.count_breaths(breathing_signal, fps)

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
            confidence = breath_count_info.get('confidence', 0.0)
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
        if self.counting_method == 'emv_phase':
            info['breath_cycles'] = breath_count_info.get('breath_cycles', [])
            info['avg_ie_ratio'] = breath_count_info.get('avg_ie_ratio', 0.0)
            info['num_zero_crossings'] = breath_count_info.get('num_zero_crossings', 0)
            info['method'] = 'emv_phase'
        elif self.counting_method == 'adaptive_window':
            info['window_estimates'] = breath_count_info.get('window_estimates', [])
            info['num_windows'] = breath_count_info.get('num_windows', 0)
            info['frequency_std'] = breath_count_info.get('frequency_std', 0.0)
            info['method'] = 'adaptive_window'
        elif self.counting_method in ['fft_frequency', 'fft_windowed']:
            # FFT-based methods
            info['window_estimates'] = breath_count_info.get('window_estimates', [])
            info['num_windows'] = breath_count_info.get('num_windows', 0)
            info['bpm_std'] = breath_count_info.get('bpm_std', 0.0)
            info['bpm_range'] = breath_count_info.get('bpm_range', (0.0, 0.0))
            info['mean_snr'] = breath_count_info.get('mean_snr', 0.0)
            info['method'] = self.counting_method
        elif self.counting_method == 'autocorrelation':
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
        else:
            info['peak_frames'] = breath_count_info.get('peak_frames', [])
            info['method'] = 'peak'

        return breathing_rate_bpm, info
    
    def count_breaths_emv(self, breathing_signal: np.ndarray, fps: Optional[float] = None) -> dict:
        """
        Count breaths using EMV (Equal Minute Ventilation) phase-based method.

        This method segments the signal into inspiration and expiration phases by
        analyzing zero-crossings and phase transitions, which is more robust than
        simple peak detection for irregular breathing patterns.

        Args:
            breathing_signal: Array of breathing measurements
            fps: Frame rate

        Returns:
            Dictionary with breath counts, phase information, and validation metrics
        """
        if fps is None:
            fps = self.fps

        # Unified preprocessing (bandpass → clip → savgol → TKEO → normalize)
        filtered_signal, _ = self._preprocess_signal(breathing_signal, fps)
        filtered_signal_clean = filtered_signal  # clip + normalize already applied

        # Zero-mean the signal for phase detection
        signal_zeromean = filtered_signal_clean - np.mean(filtered_signal_clean)

        # Calculate signal range for amplitude thresholds
        signal_range = np.ptp(filtered_signal_clean)

        # Find zero crossings (transitions between inspiration/expiration)
        # Apply threshold to avoid false crossings from noise
        zero_crossings = np.where(np.diff(np.sign(signal_zeromean)))[0]

        # Separate into inspiration and expiration phases
        # Inspiration: signal goes from negative to positive (upward crossing)
        # Expiration: signal goes from positive to negative (downward crossing)
        inspiration_starts = []
        expiration_starts = []

        # Scale threshold by signal range
        scaled_zero_crossing_threshold = self.emv_zero_crossing_threshold * signal_range

        for i in range(len(zero_crossings) - 1):
            cross_idx = zero_crossings[i]

            # Apply zero-crossing threshold to filter noise
            if scaled_zero_crossing_threshold > 0:
                # Check if crossing magnitude is significant enough
                pre_val = abs(signal_zeromean[cross_idx])
                post_val = abs(signal_zeromean[cross_idx + 1])
                if pre_val < scaled_zero_crossing_threshold and post_val < scaled_zero_crossing_threshold:
                    continue  # Skip insignificant crossing

            # Check if this is an upward or downward crossing
            if signal_zeromean[cross_idx] < 0 and signal_zeromean[cross_idx + 1] >= 0:
                # Upward crossing = start of inspiration
                inspiration_starts.append(cross_idx)
            elif signal_zeromean[cross_idx] > 0 and signal_zeromean[cross_idx + 1] <= 0:
                # Downward crossing = start of expiration
                expiration_starts.append(cross_idx)

        inspiration_starts = np.array(inspiration_starts)
        expiration_starts = np.array(expiration_starts)

        # Count complete breath cycles (inspiration + expiration = 1 breath)
        # A complete cycle starts with inspiration and ends before the next inspiration
        breath_cycles = []

        if len(inspiration_starts) > 1:
            for i in range(len(inspiration_starts) - 1):
                cycle_start = inspiration_starts[i]
                cycle_end = inspiration_starts[i + 1]

                # Find expiration in between
                exp_in_cycle = expiration_starts[(expiration_starts > cycle_start) & (expiration_starts < cycle_end)]

                if len(exp_in_cycle) > 0:
                    # Valid cycle with both inspiration and expiration
                    insp_duration = (exp_in_cycle[0] - cycle_start) / fps
                    exp_duration = (cycle_end - exp_in_cycle[0]) / fps
                    total_duration = (cycle_end - cycle_start) / fps

                    # Calculate I:E ratio
                    ie_ratio = insp_duration / exp_duration if exp_duration > 0 else 0.0

                    # Calculate cycle amplitude (peak-to-peak range during this cycle)
                    cycle_signal = signal_zeromean[cycle_start:cycle_end]
                    cycle_amplitude = np.ptp(cycle_signal) if len(cycle_signal) > 0 else 0.0
                    amplitude_ratio = cycle_amplitude / signal_range if signal_range > 0 else 0.0

                    # Apply multiple filters to reject false cycles

                    # 1. Filter by cycle duration (too fast breathing)
                    min_cycle_duration = 60.0 / self.max_breathing_rate  # seconds
                    if total_duration < min_cycle_duration:
                        continue  # Too fast, skip

                    # 2. Filter by phase duration (inspiration and expiration must be long enough)
                    if insp_duration < self.emv_min_phase_duration or exp_duration < self.emv_min_phase_duration:
                        continue  # Phase too short, likely noise

                    # 3. Filter by I:E ratio (must be physiologically realistic)
                    min_ie, max_ie = self.emv_ie_ratio_range
                    if ie_ratio < min_ie or ie_ratio > max_ie:
                        continue  # Unrealistic I:E ratio

                    # 4. Filter by cycle amplitude (must be strong enough)
                    if amplitude_ratio < self.emv_min_cycle_amplitude:
                        continue  # Cycle too shallow, likely noise

                    # All filters passed - accept this cycle
                    breath_cycles.append({
                        'start_frame': int(cycle_start),
                        'end_frame': int(cycle_end),
                        'expiration_frame': int(exp_in_cycle[0]),
                        'inspiration_duration': float(insp_duration),
                        'expiration_duration': float(exp_duration),
                        'total_duration': float(total_duration),
                        'ie_ratio': float(ie_ratio),
                        'amplitude': float(cycle_amplitude),
                        'amplitude_ratio': float(amplitude_ratio)
                    })

        # Calculate breath intervals
        breath_intervals = {}
        if len(breath_cycles) > 1:
            durations = [cycle['total_duration'] for cycle in breath_cycles]
            breath_intervals = {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
            }
        else:
            breath_intervals = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
            }

        # Count breaths in different time windows
        total_duration = len(breathing_signal) / fps
        time_windows = [10, 20, 30, 60]  # seconds

        breath_counts = {}
        window_rates = []

        for window_duration in time_windows:
            if total_duration >= window_duration:
                window_frames = int(window_duration * fps)

                # Count cycles that START within the window
                cycles_in_window = [c for c in breath_cycles if c['start_frame'] < window_frames]
                count = len(cycles_in_window)
                rate_bpm = (count / window_duration) * 60

                breath_counts[f'{window_duration}s'] = {
                    'count': int(count),
                    'rate_bpm': float(rate_bpm)
                }
                window_rates.append(rate_bpm)

        # Full duration
        count_full = len(breath_cycles)
        rate_full = (count_full / total_duration) * 60 if total_duration > 0 else 0.0
        breath_counts['full'] = {
            'count': int(count_full),
            'rate_bpm': float(rate_full),
            'duration_s': float(total_duration)
        }
        window_rates.append(rate_full)

        # Validation: check consistency across windows
        validation = {}
        if len(window_rates) > 1:
            mean_rate = np.mean(window_rates)
            std_rate = np.std(window_rates)
            cv = std_rate / mean_rate if mean_rate > 0 else 0.0
            is_consistent = cv < 0.2  # Less than 20% variation

            validation = {
                'cv': float(cv),
                'is_consistent': bool(is_consistent),
                'mean_rate': float(mean_rate),
                'std_rate': float(std_rate),
            }
        else:
            validation = {
                'cv': 0.0,
                'is_consistent': True,
                'mean_rate': float(rate_full),
                'std_rate': 0.0,
            }

        # Calculate average I:E ratio
        ie_ratios = [c['ie_ratio'] for c in breath_cycles if c['ie_ratio'] > 0]
        avg_ie_ratio = float(np.mean(ie_ratios)) if len(ie_ratios) > 0 else 0.0

        return {
            'method': 'emv_phase',
            'breath_counts': breath_counts,
            'breath_intervals': breath_intervals,
            'breath_cycles': breath_cycles,
            'avg_ie_ratio': avg_ie_ratio,
            'num_zero_crossings': len(zero_crossings),
            'validation': validation,
        }

    def count_breaths(self, breathing_signal: np.ndarray, fps: Optional[float] = None) -> dict:
        """
        Count individual breaths using peak detection across different time windows

        Args:
            breathing_signal: Array of breathing measurements
            fps: Frame rate

        Returns:
            Dictionary with breath counts, intervals, and validation metrics
        """
        if fps is None:
            fps = self.fps

        # Unified preprocessing (bandpass → clip → savgol → TKEO → normalize)
        filtered_signal, _ = self._preprocess_signal(breathing_signal, fps)
        filtered_signal_clean = filtered_signal  # clip + normalize already applied

        # Peak detection parameters (read from config)
        min_distance = int(fps / (self.max_breathing_rate / 60))  # Minimum frames between peaks

        # Auto-calculate prominence using cleaned signal (robust to outliers)
        signal_range = np.ptp(filtered_signal_clean)  # Peak-to-peak
        prominence = signal_range * self.peak_prominence_ratio  # Use config value

        # Find peaks on original filtered signal (not cleaned)
        # We only used cleaned signal to calculate robust thresholds
        peaks, properties = signal.find_peaks(
            filtered_signal,
            distance=min_distance,
            prominence=prominence,
            rel_height=0.5
        )

        # Filter out outlier peaks (peaks that are too high compared to median peak height)
        if len(peaks) > 3:
            peak_heights = filtered_signal[peaks]
            median_height = np.median(peak_heights)
            mad = np.median(np.abs(peak_heights - median_height))  # Median Absolute Deviation

            # Keep peaks within 3 MAD of median (robust outlier detection)
            if mad > 0:
                threshold = median_height + 3 * mad
                valid_peaks_mask = peak_heights <= threshold
                peaks = peaks[valid_peaks_mask]

        # Calculate breath-to-breath intervals
        breath_intervals = {}
        if len(peaks) > 1:
            intervals_frames = np.diff(peaks)
            intervals_seconds = intervals_frames / fps
            breath_intervals = {
                'mean': float(np.mean(intervals_seconds)),
                'std': float(np.std(intervals_seconds)),
                'min': float(np.min(intervals_seconds)),
                'max': float(np.max(intervals_seconds)),
            }
        else:
            breath_intervals = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
            }

        # Count breaths in different time windows
        total_duration = len(breathing_signal) / fps
        time_windows = [10, 20, 30, 60]  # seconds

        breath_counts = {}
        window_rates = []

        for window_duration in time_windows:
            if total_duration >= window_duration:
                window_frames = int(window_duration * fps)
                peaks_in_window = peaks[peaks < window_frames]
                count = len(peaks_in_window)
                rate_bpm = (count / window_duration) * 60
                breath_counts[f'{window_duration}s'] = {
                    'count': int(count),
                    'rate_bpm': float(rate_bpm)
                }
                window_rates.append(rate_bpm)

        # Full duration
        count_full = len(peaks)
        rate_full = (count_full / total_duration) * 60 if total_duration > 0 else 0.0
        breath_counts['full'] = {
            'count': int(count_full),
            'rate_bpm': float(rate_full),
            'duration_s': float(total_duration)
        }
        window_rates.append(rate_full)

        # Validation: check consistency across windows
        validation = {}
        if len(window_rates) > 1:
            mean_rate = np.mean(window_rates)
            std_rate = np.std(window_rates)
            cv = std_rate / mean_rate if mean_rate > 0 else 0.0
            is_consistent = cv < 0.2  # Less than 20% variation

            validation = {
                'cv': float(cv),
                'is_consistent': bool(is_consistent),
                'mean_rate': float(mean_rate),
                'std_rate': float(std_rate),
            }
        else:
            validation = {
                'cv': 0.0,
                'is_consistent': True,
                'mean_rate': float(rate_full),
                'std_rate': 0.0,
            }

        return {
            'breath_counts': breath_counts,
            'breath_intervals': breath_intervals,
            'peak_frames': peaks.tolist(),
            'validation': validation,
        }

    def count_breaths_fft(self, breathing_signal: np.ndarray, fps: float, skip_preprocess=False) -> dict:
        """
        Count breaths using an optimized FFT analysis with a linear ramp bias
        to prioritize high-frequency bird breathing over low-frequency noise.
        """
        # 1. Preprocess signal
        if not skip_preprocess:
            filtered, _ = self._preprocess_signal(breathing_signal, fps)
        else:
            filtered = breathing_signal
        
        # 2. Windowing
        n_samples = len(filtered)
        windowed_signal = filtered * np.hanning(n_samples)
        
        # 3. FFT with Zero-Padding
        n_fft = max(4096, n_samples)
        # Using rfft is more efficient for real-valued signals
        fft_result = np.fft.rfft(windowed_signal, n=n_fft)
        frequencies = np.fft.rfftfreq(n_fft, 1/fps)
        spectrum = np.abs(fft_result)
        
        # 4. Range Masking (Dataset: 95 - 310 BPM)
        # We search up to 350 BPM (approx 5.83Hz) to allow parabolic headroom
        search_max_hz = 350.0 / 60.0
        breathing_mask = (frequencies >= self.low_freq) & (frequencies <= search_max_hz)

        # TEST 02.22>>>>>>>>>>>>>>>>>>>>>>>>>
        #surgical_low_bpm = 140 
        #breathing_mask = (frequencies >= surgical_low_bpm / 60.0) & (frequencies <= 350 / 60.0)
        ##################################################

        breathing_freqs = frequencies[breathing_mask]
        breathing_spectrum = spectrum[breathing_mask]
        
        if len(breathing_spectrum) < 3:
            return {'breathing_rate_bpm': 0.0, 'snr': 0.0, 'confidence': 0.0, 'breath_count': 0}
        
        # 5. BIAS RAMP (The "R2 Fix")
        # Counteracts 1/f noise by linearly boosting higher frequencies for peak selection
        ramp = np.linspace(1.0, 3.5, len(breathing_spectrum))
        spectrum_biased = breathing_spectrum * ramp

        # 6. Find peak on BIASED spectrum, but interpolate on ORIGINAL spectrum
        peak_idx = np.argmax(spectrum_biased)
        
        # 7. Sub-bin Parabolic Interpolation (using original spectrum for accuracy)
        if 0 < peak_idx < len(breathing_spectrum) - 1:
            # Log-magnitude parabolic fit
            y = np.log(breathing_spectrum[peak_idx-1 : peak_idx+2] + 1e-9)
            denom = (2 * y[1] - y[0] - y[2])
            offset = (y[2] - y[0]) / (2 * denom) if denom != 0 else 0.0
            
            df = breathing_freqs[1] - breathing_freqs[0]
            dominant_freq = breathing_freqs[peak_idx] + (offset * df)
        else:
            dominant_freq = breathing_freqs[peak_idx]
        
        # 8. Calculate SNR (Peak power vs Mean noise floor of the original spectrum)
        peak_power = breathing_spectrum[peak_idx]
        mean_power = np.mean(breathing_spectrum)
        snr = peak_power / mean_power if mean_power > 0 else 0.0
        
        # Map SNR to 0.0-1.0 confidence
        confidence = min(snr / 5.0, 1.0)
        
        # 9. Final Metrics
        breathing_rate_bpm = float(dominant_freq * 60.0)
        duration_min = len(breathing_signal) / fps / 60.0
        breath_count = int(round(breathing_rate_bpm * duration_min))
        
        return {
            'breathing_rate_bpm': breathing_rate_bpm,
            'dominant_frequency': float(dominant_freq),
            'snr': float(snr),
            'confidence': float(confidence),
            'breath_count': breath_count,
            'breath_counts': {'total': breath_count, 'accepted': breath_count},
            'validation': {'snr': snr, 'method': 'fft_refined_ramped'},
            'spectrum': breathing_spectrum,
            'frequencies': breathing_freqs,
            'peak_power': float(peak_power),
            'mean_power': float(mean_power)
        }
    
    def count_breaths_fft_windowed(
        self,
        breathing_signal: np.ndarray,
        fps: float,
        window_size: Optional[float] = None,
        overlap: Optional[float] = None
    ) -> dict:
        """
        Count breaths using windowed FFT (STFT) for non-stationary signals.

        Handles breathing frequency changes over time by analyzing
        overlapping windows independently. This is the recommended method
        for bird breathing analysis based on 2018 successful approach.

        Args:
            breathing_signal: Raw breathing signal
            fps: Frames per second
            window_size: Window duration in seconds (default from config)
            overlap: Window overlap fraction 0-1 (default from config)

        Returns:
            dict with breath counting results including per-window estimates
        """
        # Use config defaults if not specified
        # 1. Setup Defaults
        if window_size is None:
            window_size = getattr(self, 'fft_window_size', 8.0) # Changed to 8s
        print("---- window size FFT: ", window_size)
        if overlap is None:
            overlap = getattr(self, 'fft_overlap', 0.5)

        skip_preprocess = True # ALREADY DONE BEFORE CALLING THIS!! we recieve an already clean signal here 21.02
        # Preprocess
        
        if not skip_preprocess:
            filtered, _ = self._preprocess_signal(breathing_signal, fps)
        else:
            filtered = breathing_signal # already clean

        # Window parameters
        window_frames = int(window_size * fps)
        hop_frames = int(window_frames * (1 - overlap))

        ### 02.21 Sanity check:
        #
        print("self.low_freq: ", self.low_freq)
        print("self.acf_min_bpm: ", self.acf_min_bpm)
        print("self.max_freq: ", self.high_freq)
        print("self.acf_max_bpm: ", self.acf_max_bpm)

        # Minimum window size check
        if len(filtered) < window_frames:
            # Fall back to single-window FFT
            return self.count_breaths_fft(breathing_signal, fps)

        # Collect estimates from each window
        window_estimates = []

        start_idx = 0

        # Pre-calculate FFT parameters for resolution NEW 02.21
        # Padding to 4096 gives ~0.4 BPM resolution at 30 FPS
        n_fft = 4096

        while start_idx + window_frames <= len(filtered):
            window_signal = filtered[start_idx:start_idx + window_frames]

            # --- CHANGE 1: LOCAL DETREND --- 02.21
            # This is the "secret sauce." It re-centers the 8s slice to 0
            # so the Hanning window doesn't create a "thump" at the edges.
            from scipy import signal as scipy_signal
            window_signal = scipy_signal.detrend(window_signal, type='linear')

            # --- CHANGE 2: PASS THE FLAG ---
            # You must tell the internal FFT to skip its own preprocessing,
            # otherwise it will try to re-integrate and re-bandpass the slice!
            result = self.count_breaths_fft(window_signal, fps, skip_preprocess=True)

            min_snr = getattr(self, 'fft_min_snr', 3.0)
            if result['confidence'] > (min_snr / 5.0):  # Only use confident estimates
                window_estimates.append({
                    'bpm': result['breathing_rate_bpm'],
                    'confidence': result['confidence'],
                    'snr': result['snr'],
                    'start_frame': start_idx,
                    'end_frame': start_idx + window_frames,
                    'start_time': start_idx / fps,
                    'end_time': (start_idx + window_frames) / fps
                })

            start_idx += hop_frames

        if len(window_estimates) == 0:
            return {
                'breathing_rate_bpm': 0.0,
                'confidence': 0.0,
                'breath_count': 0,
                'breath_counts': {'total': 0, 'accepted': 0},
                'breath_intervals': [],
                'validation': {'error': 'No confident window estimates'},
                'error': 'No confident window estimates'
            }

        # Aggregate estimates (confidence-weighted median)
        estimates = np.array([w['bpm'] for w in window_estimates])
        confidences = np.array([w['confidence'] for w in window_estimates])

        # Weighted median
        sorted_idx = np.argsort(estimates)
        sorted_estimates = estimates[sorted_idx]
        sorted_weights = confidences[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)

        final_bpm = sorted_estimates[median_idx]
        final_confidence = np.mean(confidences)

        # Calculate breath count
        duration_min = len(breathing_signal) / fps / 60.0
        breath_count = int(final_bpm * duration_min)

        # Estimate breath intervals (uniform based on BPM)
        breath_period = 60.0 / final_bpm if final_bpm > 0 else 0.0
        breath_intervals = [breath_period] * (breath_count - 1) if breath_count > 1 else []

        return {
            'breathing_rate_bpm': final_bpm,
            'breath_count': breath_count,
            'breath_counts': {'total': breath_count, 'accepted': breath_count},
            'breath_intervals': breath_intervals,
            'confidence': final_confidence,
            'validation': {
                'method': 'fft_windowed',
                'mean_snr': float(np.mean([w['snr'] for w in window_estimates])),
                'bpm_std': float(np.std(estimates))
            },
            'num_windows': len(window_estimates),
            'window_estimates': window_estimates,
            'bpm_std': np.std(estimates),
            'bpm_range': (float(np.min(estimates)), float(np.max(estimates))),
            'mean_snr': float(np.mean([w['snr'] for w in window_estimates]))
        }

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
                distance=min_lag,
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

        # Aggregate window estimates using confidence-weighted median
        bpms = np.array([w['bpm'] for w in window_estimates])
        confidences = np.array([w['confidence'] for w in window_estimates])

        # Confidence-weighted median
        if len(bpms) > 0:
            sorted_indices = np.argsort(bpms)
            sorted_bpms = bpms[sorted_indices]
            sorted_confidences = confidences[sorted_indices]

            cumulative_confidence = np.cumsum(sorted_confidences)
            total_confidence = cumulative_confidence[-1]

            median_idx = np.searchsorted(cumulative_confidence, total_confidence / 2.0)
            breathing_rate_bpm = sorted_bpms[median_idx]
        else:
            breathing_rate_bpm = 0.0

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
        }

    def _bandpass_filter(self, signal_data: np.ndarray, fps: float) -> np.ndarray:
        """Apply bandpass filter to signal using instance config"""
        return bandpass_filter(signal_data, fps, self.low_freq, self.high_freq, self.order)

    def _remove_outliers(self, signal_data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using MAD method - delegates to utility function"""
        return remove_outliers(signal_data, threshold)

    def _savgol_filter(self, signal_data: np.ndarray, window_size: int, polyorder: int) -> np.ndarray:
        """Apply Savitzky-Golay filter - delegates to utility function"""
        return apply_savgol_filter(signal_data, window_size, polyorder)

    def _normalize(self, signal_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normalize signal - delegates to utility function"""
        return normalize_signal(signal_data, method)

    def _nonlinear_amplification(self, signal_data: np.ndarray, method: str = 'tkeo') -> np.ndarray:
        """Apply nonlinear amplification - delegates to utility function"""
        return nonlinear_amplification(signal_data, method)

    def _preprocess_for_fft(self, raw_signal: np.ndarray, fps: float) -> np.ndarray:
        """
        Surgical preprocessing for Spectral Analysis.
        Prioritizes frequency stability over time-domain shape.
        """

        info = {'raw': raw_signal.copy()}

        # 1. Zero-mean the raw signal immediately
        sig = raw_signal - np.median(raw_signal)

        info['after_zero_mean'] = sig.copy()
        
        # 2. Strong Bandpass (Strictly 1.5Hz to 8.0Hz)
        # Using a higher order (e.g., 4 or 6) here is better for FFT 
        # to kill the 1Hz 'rumble' that causes the 100 BPM outliers.
        from scipy.signal import butter, filtfilt
        nyquist = 0.5 * fps
        b, a = butter(4, [1.5 / nyquist, 8.0 / nyquist], btype='band')
        sig = filtfilt(b, a, sig)

        info['after_bandpass'] = sig.copy()
        
        # 3. LEAKY INTEGRATION (Velocity -> Displacement)
        # Standard np.cumsum creates a 1/f slope that ruins R2.
        # A leaky integrator preserves the oscillation but resets the drift.
        alpha = 0.9  # Adjust between 0.8 and 0.95
        sig_int = np.zeros_like(sig)
        for i in range(1, len(sig)):
            sig_int[i] = alpha * sig_int[i-1] + sig[i]
        
        info['after_integration'] = sig.copy()
        
        # 4. Final Detrend
        from scipy import signal as scipy_signal
        sig_final = scipy_signal.detrend(sig_int, type='linear')

        info['after_detrend'] = sig_final.copy()
        
        return sig_final, info

    def _preprocess_signal(self, raw_signal: np.ndarray, fps: float) -> Tuple[np.ndarray, dict]:
        """
        Updated Pipeline for Optical Flow Divergence:
        1. Bandpass (Clean velocity)
        2. Integrate (Velocity -> Displacement/Size)  <-- FIXED OVERCOUNT
        3. Detrend/High-pass (Remove integration drift)
        4. Outlier clip (Protect scale)
        5. Savitzky-Golay (Smooth)
        6. Normalize (Z-score)
        """
        info = {'raw': raw_signal.copy()}

        # ################## test 02.21 cleaning pipeline
        use_savgol = False
        use_detrend = False



        ###############################################
        

        # 0: Handle nans with linear interpolation!!!
        if np.any(np.isnan(raw_signal)):
            try:
                raw_signal, nan_count = interpolate_nans(raw_signal)
                print(f"  Warning: {nan_count} NaN values in signal ({nan_count/len(raw_signal)*100:.1f}%)")
            except ValueError as e:
                print(f"✗ Error: {e}")
                return None, {}

        # 1. Bandpass — Remove non-biological wiggles first
        processed = self._bandpass_filter(raw_signal, fps)
        info['after_bandpass'] = processed.copy()

        # 2. INTEGRATION: Convert Velocity to Displacement
        # This merges the 'in' and 'out' spikes into a single 'breath' hump
        processed = np.cumsum(processed)
        info['after_integration'] = processed.copy()

        # 2.1. Gentle high-pass to clear integration drift
        processed = self._bandpass_filter(processed, fps)

        if use_detrend:
            # 3. SECONDARY DETREND: Integration causes drift. Must re-center.
            # A simple linear detrend or high-pass handles the "slope" caused by cumsum
            processed = signal.detrend(processed, type='linear')
            info['after_integration_detrend'] = processed.copy()

        # 4. Outlier clip — Cap spikes so they don't dominate
        clip_config = self.preprocess_config.get('outlier_clip', {})
        # Note: 3.0 sigma is safer for integrated signals
        limit = np.std(processed) * clip_config.get('std_threshold', 3.0) 
        processed = np.clip(processed, -limit, limit)
        info['after_clip'] = processed.copy()

        if use_savgol:
            # 5. Savitzky-Golay smoothing
            savgol_config = self.preprocess_config.get('savgol', {})
            if savgol_config.get('enabled', True):
                # For 300 BPM at 30fps, window 5-7 is best for the integrated wave
                window_size = savgol_config.get('window_size', 7)
                polyorder  = savgol_config.get('polyorder', 2)
                processed  = self._savgol_filter(processed, window_size, polyorder)
                info['after_savgol'] = processed.copy()

        # 6. Nonlinear amplification (OPTIONAL - suggest keeping DISABLED)
        # Integration usually makes the signal so clean you don't need TKEO
        nonlinear_config = self.preprocess_config.get('nonlinear_amplification', {})
        if nonlinear_config.get('enabled', False):
            method   = nonlinear_config.get('method', 'tkeo')
            processed = self._nonlinear_amplification(processed, method)
            info['after_nonlinear'] = processed.copy()

        # 7. Normalize
        normalize_config = self.preprocess_config.get('normalize', {})
        if normalize_config.get('enabled', True):
            method   = normalize_config.get('method', 'zscore')
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
