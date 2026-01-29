"""
Visualize the effect of outlier handling on breathing signal analysis
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.pipeline import BreathingAnalyzer

# Setup paths
config_path = 'configs/default.yaml'
video_path = 'data/videos/bird_sample.mp4'

print("="*70)
print("OUTLIER HANDLING ANALYSIS")
print("="*70)
print("Processing video to analyze outlier effects...\n")

# Process video
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=None)

# Extract data
breathing_signal = np.array(results['breathing_signal'])
fps = results['video_fps']

# Get signal processing module to access outlier removal
from src.signal_processing import SignalProcessor
processor = SignalProcessor({'fps': fps, 'bandpass_filter': {
    'low_freq': 0.5, 'high_freq': 4.0, 'order': 4
}})

# Apply bandpass filter
filtered_signal = processor._bandpass_filter(breathing_signal, fps)

# Remove outliers
cleaned_signal = processor._remove_outliers(filtered_signal)

# Detect which points are outliers
outlier_mask = np.abs(filtered_signal - cleaned_signal) > 0.01

# Calculate statistics - traditional vs robust
print(f"\n{'='*70}")
print("TRADITIONAL STATISTICS (Outlier-Sensitive)")
print(f"{'='*70}")
print(f"Mean:           {np.mean(filtered_signal):.4f}")
print(f"Std Dev:        {np.std(filtered_signal):.4f}")
print(f"Range (max-min): {np.ptp(filtered_signal):.4f}")
print(f"Min:            {np.min(filtered_signal):.4f}")
print(f"Max:            {np.max(filtered_signal):.4f}")

print(f"\n{'='*70}")
print("ROBUST STATISTICS (Outlier-Resistant)")
print(f"{'='*70}")
print(f"Median:         {np.median(filtered_signal):.4f}")
print(f"MAD:            {np.median(np.abs(filtered_signal - np.median(filtered_signal))):.4f}")
print(f"P10-P90 Range:  {np.percentile(filtered_signal, 90) - np.percentile(filtered_signal, 10):.4f}")
print(f"P10:            {np.percentile(filtered_signal, 10):.4f}")
print(f"P90:            {np.percentile(filtered_signal, 90):.4f}")

print(f"\n{'='*70}")
print("OUTLIER DETECTION")
print(f"{'='*70}")
num_outliers = np.sum(outlier_mask)
pct_outliers = 100 * num_outliers / len(filtered_signal)
print(f"Outliers detected: {num_outliers} / {len(filtered_signal)} ({pct_outliers:.1f}%)")
if num_outliers > 0:
    outlier_frames = np.where(outlier_mask)[0]
    print(f"Outlier frames: {outlier_frames[:10].tolist()}" +
          (f"... and {len(outlier_frames)-10} more" if len(outlier_frames) > 10 else ""))

# Peak detection comparison
from scipy import signal as scipy_signal

# Traditional approach (using full range)
trad_range = np.ptp(filtered_signal)
trad_prominence = trad_range * 0.2
trad_peaks, _ = scipy_signal.find_peaks(
    filtered_signal,
    distance=int(fps / 4),
    prominence=trad_prominence
)

# Robust approach (using cleaned signal for threshold)
robust_range = np.ptp(cleaned_signal)
robust_prominence = robust_range * 0.2
robust_peaks, _ = scipy_signal.find_peaks(
    filtered_signal,
    distance=int(fps / 4),
    prominence=robust_prominence
)

# Filter outlier peaks
if len(robust_peaks) > 3:
    peak_heights = filtered_signal[robust_peaks]
    median_height = np.median(peak_heights)
    mad = np.median(np.abs(peak_heights - median_height))
    if mad > 0:
        threshold = median_height + 3 * mad
        robust_peaks = robust_peaks[peak_heights <= threshold]

print(f"\n{'='*70}")
print("PEAK DETECTION COMPARISON")
print(f"{'='*70}")
print(f"Traditional approach:")
print(f"  Range: {trad_range:.4f}")
print(f"  Prominence threshold: {trad_prominence:.4f}")
print(f"  Peaks detected: {len(trad_peaks)}")

print(f"\nRobust approach:")
print(f"  Range (cleaned): {robust_range:.4f}")
print(f"  Prominence threshold: {robust_prominence:.4f}")
print(f"  Peaks detected (before outlier filter): {len(robust_peaks)}")
print(f"  Peaks detected (after outlier filter): {len(robust_peaks)}")

# Calculate breathing rates
duration_s = len(breathing_signal) / fps
trad_rate = (len(trad_peaks) / duration_s) * 60
robust_rate = (len(robust_peaks) / duration_s) * 60

print(f"\nBreathing Rate Estimates:")
print(f"  Traditional: {trad_rate:.1f} BPM ({len(trad_peaks)} breaths in {duration_s:.1f}s)")
print(f"  Robust:      {robust_rate:.1f} BPM ({len(robust_peaks)} breaths in {duration_s:.1f}s)")
print(f"  Difference:  {abs(trad_rate - robust_rate):.1f} BPM")

# Create visualization
fig = GridSpec(5, 2, figure=plt.figure(figsize=(16, 14)), hspace=0.4, wspace=0.3)

# Plot 1: Original signal with outliers highlighted
ax1 = plt.subplot(fig[0, :])
time_axis = np.arange(len(filtered_signal)) / fps
ax1.plot(time_axis, filtered_signal, 'b-', linewidth=1, alpha=0.7, label='Filtered Signal')
if num_outliers > 0:
    outlier_times = time_axis[outlier_mask]
    outlier_values = filtered_signal[outlier_mask]
    ax1.scatter(outlier_times, outlier_values, color='red', s=50, zorder=5,
               marker='x', linewidth=2, label=f'Outliers ({num_outliers})')
ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Signal Magnitude', fontsize=11, fontweight='bold')
ax1.set_title('Filtered Signal with Outliers Highlighted', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cleaned signal overlay
ax2 = plt.subplot(fig[1, :])
ax2.plot(time_axis, filtered_signal, 'b-', linewidth=1, alpha=0.5, label='Original')
ax2.plot(time_axis, cleaned_signal, 'g-', linewidth=1.5, label='Cleaned (outliers removed)')
ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Signal Magnitude', fontsize=11, fontweight='bold')
ax2.set_title('Original vs Cleaned Signal', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Traditional peak detection
ax3 = plt.subplot(fig[2, :])
ax3.plot(time_axis, filtered_signal, 'b-', linewidth=1, alpha=0.6)
if len(trad_peaks) > 0:
    peak_times = time_axis[trad_peaks]
    peak_values = filtered_signal[trad_peaks]
    ax3.scatter(peak_times, peak_values, color='red', s=60, zorder=5,
               marker='v', label=f'Detected Peaks ({len(trad_peaks)})')
ax3.axhline(y=np.mean(filtered_signal) + trad_prominence, color='orange',
           linestyle='--', linewidth=2, alpha=0.7,
           label=f'Threshold (prominence={trad_prominence:.3f})')
ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Signal Magnitude', fontsize=11, fontweight='bold')
ax3.set_title(f'Traditional Peak Detection: {trad_rate:.1f} BPM', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Robust peak detection
ax4 = plt.subplot(fig[3, :])
ax4.plot(time_axis, filtered_signal, 'b-', linewidth=1, alpha=0.6)
if len(robust_peaks) > 0:
    peak_times = time_axis[robust_peaks]
    peak_values = filtered_signal[robust_peaks]
    ax4.scatter(peak_times, peak_values, color='green', s=60, zorder=5,
               marker='v', label=f'Detected Peaks ({len(robust_peaks)})')
ax4.axhline(y=np.median(cleaned_signal) + robust_prominence, color='green',
           linestyle='--', linewidth=2, alpha=0.7,
           label=f'Threshold (prominence={robust_prominence:.3f})')
ax4.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Signal Magnitude', fontsize=11, fontweight='bold')
ax4.set_title(f'Robust Peak Detection: {robust_rate:.1f} BPM', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Histogram comparison
ax5 = plt.subplot(fig[4, 0])
ax5.hist(filtered_signal, bins=50, alpha=0.6, color='blue', edgecolor='black', label='Original')
ax5.axvline(np.mean(filtered_signal), color='blue', linestyle='--', linewidth=2, label='Mean')
ax5.axvline(np.median(filtered_signal), color='red', linestyle='--', linewidth=2, label='Median')
ax5.set_xlabel('Signal Value', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Signal Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Statistics comparison table
ax6 = plt.subplot(fig[4, 1])
ax6.axis('off')

table_data = [
    ['Statistic', 'Traditional', 'Robust', 'Better for Outliers'],
    ['Central Tendency', f"{np.mean(filtered_signal):.4f} (mean)",
     f"{np.median(filtered_signal):.4f} (median)", 'Median ✓'],
    ['Spread', f"{np.std(filtered_signal):.4f} (std)",
     f"{np.median(np.abs(filtered_signal - np.median(filtered_signal))):.4f} (MAD)", 'MAD ✓'],
    ['Range', f"{np.ptp(filtered_signal):.4f} (max-min)",
     f"{np.percentile(filtered_signal, 90) - np.percentile(filtered_signal, 10):.4f} (P90-P10)", 'P90-P10 ✓'],
    ['Outliers Found', '-', f'{num_outliers} ({pct_outliers:.1f}%)', 'Robust ✓'],
    ['Peaks Detected', f'{len(trad_peaks)}', f'{len(robust_peaks)}', 'Depends'],
    ['Breathing Rate', f'{trad_rate:.1f} BPM', f'{robust_rate:.1f} BPM', 'Depends']
]

table = ax6.table(cellText=table_data, cellLoc='left',
                 loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=9)

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

ax6.set_title('Statistics Comparison', fontsize=12, fontweight='bold', pad=15)

plt.suptitle('Outlier Handling Analysis', fontsize=15, fontweight='bold', y=0.995)

# Save plot
output_path = 'data/results/outlier_handling_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"Visualization saved to: {output_path}")
print(f"{'='*70}")

plt.show()
