"""
Signal processing for breathing rate estimation
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class SignalProcessor:
    """
    Process breathing signal to estimate breathing rate
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.fps = config.get('fps', 30)

        filter_config = config.get('bandpass_filter', {})
        self.low_freq = filter_config.get('low_freq', 0.5)
        self.high_freq = filter_config.get('high_freq', 4.0)
        self.order = filter_config.get('order', 4)

        # Load preprocessing config
        self.preprocess_config = config.get('preprocessing', {})

        # Load breath counting config
        breath_config = config.get('breath_counting', {})
        self.max_breathing_rate = breath_config.get('max_breathing_rate_bpm', 360)

        print(f"✓ SignalProcessor initialized (fps={self.fps}, "
              f"band={self.low_freq}-{self.high_freq} Hz)")
    
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

        # Apply preprocessing (moving average, detrending, normalization)
        preprocessed_signal, preprocessing_info = self._apply_preprocessing(breathing_signal)

        # Apply bandpass filter
        filtered_signal = self._bandpass_filter(preprocessed_signal, fps)

        # Count breaths using peak detection (use raw signal to preserve peaks)
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

        # Convert to BPM
        breathing_rate_bpm = breathing_freq_hz * 60

        # Confidence score (ratio of peak to total energy)
        total_energy = np.sum(breathing_fft)
        confidence = peak_magnitude / total_energy if total_energy > 0 else 0.0

        info = {
            'frequency_hz': breathing_freq_hz,
            'confidence': confidence,
            'peak_magnitude': peak_magnitude,
            'preprocessed_signal': preprocessed_signal,
            'filtered_signal': filtered_signal,
            'frequencies': breathing_freqs,
            'fft_magnitude': breathing_fft,
            # Add breath counting information
            'breath_counts': breath_count_info['breath_counts'],
            'breath_intervals': breath_count_info['breath_intervals'],
            'peak_frames': breath_count_info['peak_frames'],
            'validation': breath_count_info['validation'],
            # Add preprocessing information
            'preprocessing': preprocessing_info,
        }

        return breathing_rate_bpm, info
    
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

        # Apply bandpass filter
        filtered_signal = self._bandpass_filter(breathing_signal, fps)

        # Remove outliers before peak detection to avoid inflated thresholds
        filtered_signal_clean = self._remove_outliers(filtered_signal)

        # Peak detection parameters (read from config)
        min_distance = int(fps / (self.max_breathing_rate / 60))  # Minimum frames between peaks

        # Auto-calculate prominence using cleaned signal (robust to outliers)
        signal_range = np.ptp(filtered_signal_clean)  # Peak-to-peak
        prominence = signal_range * 0.1  # 10% of range (reduced from 0.2 to match real-time sensitivity)

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

    def _bandpass_filter(self, signal_data: np.ndarray, fps: float) -> np.ndarray:
        """
        Apply bandpass filter to signal
        """
        nyquist = fps / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist

        # Design filter
        b, a = signal.butter(self.order, [low, high], btype='band')

        # Apply filter (forward-backward to avoid phase shift)
        filtered = signal.filtfilt(b, a, signal_data)

        return filtered

    def _remove_outliers(self, signal_data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
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

    def _moving_average(self, signal_data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving average smoothing

        Args:
            signal_data: Input signal
            window_size: Window size in frames

        Returns:
            Smoothed signal
        """
        if window_size <= 1:
            return signal_data

        # Use numpy convolve for efficient moving average
        kernel = np.ones(window_size) / window_size
        # Mode='same' keeps the same length, 'valid' would shorten it
        smoothed = np.convolve(signal_data, kernel, mode='same')

        return smoothed

    def _savgol_filter(self, signal_data: np.ndarray, window_size: int, polyorder: int) -> np.ndarray:
        """
        Apply Savitzky-Golay filter (preserves peaks better than moving average)

        Args:
            signal_data: Input signal
            window_size: Window size (must be odd)
            polyorder: Polynomial order

        Returns:
            Filtered signal
        """
        from scipy.signal import savgol_filter

        # Ensure window size is odd and valid
        if window_size % 2 == 0:
            window_size += 1

        if window_size > len(signal_data):
            window_size = len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1

        if window_size < polyorder + 2:
            return signal_data  # Can't apply filter

        return savgol_filter(signal_data, window_size, polyorder)

    def _detrend(self, signal_data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Remove baseline drift from signal

        Args:
            signal_data: Input signal
            method: 'linear' or 'constant'

        Returns:
            Detrended signal
        """
        from scipy.signal import detrend

        return detrend(signal_data, type=method)

    def _normalize(self, signal_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
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

    def _apply_preprocessing(self, signal_data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply all enabled preprocessing steps

        Args:
            signal_data: Raw input signal

        Returns:
            preprocessed_signal: Processed signal
            preprocessing_info: Dict with intermediate signals for visualization
        """
        preprocessing_info = {
            'raw': signal_data.copy()
        }

        processed = signal_data.copy()

        # 1. Moving average (if enabled)
        ma_config = self.preprocess_config.get('moving_average', {})
        if ma_config.get('enabled', False):
            window_size = ma_config.get('window_size', 5)
            processed = self._moving_average(processed, window_size)
            preprocessing_info['after_moving_average'] = processed.copy()

        # 2. Savitzky-Golay filter (if enabled and moving average not used)
        savgol_config = self.preprocess_config.get('savgol', {})
        if savgol_config.get('enabled', False) and not ma_config.get('enabled', False):
            window_size = savgol_config.get('window_size', 11)
            polyorder = savgol_config.get('polyorder', 3)
            processed = self._savgol_filter(processed, window_size, polyorder)
            preprocessing_info['after_savgol'] = processed.copy()

        # 3. Detrending (if enabled)
        detrend_config = self.preprocess_config.get('detrend', {})
        if detrend_config.get('enabled', False):
            method = detrend_config.get('method', 'linear')
            processed = self._detrend(processed, method)
            preprocessing_info['after_detrend'] = processed.copy()

        # 4. Normalization (if enabled)
        normalize_config = self.preprocess_config.get('normalize', {})
        if normalize_config.get('enabled', False):
            method = normalize_config.get('method', 'zscore')
            processed = self._normalize(processed, method)
            preprocessing_info['after_normalize'] = processed.copy()

        return processed, preprocessing_info

    def plot_analysis(self, breathing_signal: np.ndarray, save_path: Optional[str] = None,
                     show_outliers: bool = True):
        """
        Plot signal analysis with outlier visualization

        Args:
            breathing_signal: Raw breathing signal
            save_path: Optional path to save plot
            show_outliers: Whether to show outlier detection (default: True)
        """
        import matplotlib.pyplot as plt

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
            from scipy.signal import spectrogram as scipy_spectrogram

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
