"""
Visualize breath counting analysis
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

print("Processing video and analyzing breath counts...")
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=None)

# Extract data
breathing_signal = np.array(results['breathing_signal'])
fps = results['video_fps']

# Re-run signal processing to get detailed info
from src.signal_processing import SignalProcessor
processor = SignalProcessor({'fps': fps, 'bandpass_filter': {
    'low_freq': 0.5, 'high_freq': 4.0, 'order': 4
}})

breathing_rate_fft, info = processor.estimate_breathing_rate(breathing_signal, fps)

# Extract breath counting data
filtered_signal = info.get('filtered_signal', breathing_signal)
peak_frames = info.get('peak_frames', [])
breath_counts = info.get('breath_counts', {})
breath_intervals = info.get('breath_intervals', {})
validation = info.get('validation', {})

print(f"\n{'='*70}")
print("BREATH COUNTING ANALYSIS")
print(f"{'='*70}")
print(f"Total peaks detected: {len(peak_frames)}")
print(f"FFT-based rate: {breathing_rate_fft:.1f} BPM")
if breath_counts.get('full'):
    print(f"Count-based rate (full): {breath_counts['full']['rate_bpm']:.1f} BPM")
print(f"{'='*70}\n")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

# Plot 1: Raw signal with detected peaks
ax1 = fig.add_subplot(gs[0, :])
time_axis = np.arange(len(breathing_signal)) / fps
ax1.plot(time_axis, breathing_signal, color='#2E86AB', linewidth=1, alpha=0.6, label='Raw Signal')
ax1.plot(time_axis, filtered_signal, color='#0066CC', linewidth=1.5, label='Filtered Signal')

# Mark detected peaks
if len(peak_frames) > 0:
    peak_times = np.array(peak_frames) / fps
    peak_values = filtered_signal[peak_frames]
    ax1.scatter(peak_times, peak_values, color='red', s=50, zorder=5,
               marker='v', label=f'Detected Breaths (n={len(peak_frames)})')

ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Signal Magnitude', fontsize=11, fontweight='bold')
ax1.set_title('Breathing Signal with Detected Breath Cycles', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Breath counts per time window
ax2 = fig.add_subplot(gs[1, 0])
windows = []
counts = []
rates = []

for window, data in sorted(breath_counts.items(), key=lambda x: (x[0] != 'full', x[0])):
    if window != 'full':
        windows.append(window)
        counts.append(data['count'])
        rates.append(data['rate_bpm'])

if windows:
    x = np.arange(len(windows))
    bars = ax2.bar(x, counts, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows)
    ax2.set_xlabel('Time Window', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Breaths', fontsize=11, fontweight='bold')
    ax2.set_title('Breath Counts per Time Window', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Breathing rates per time window
ax3 = fig.add_subplot(gs[1, 1])
if windows:
    x = np.arange(len(windows))
    bars = ax3.bar(x, rates, color='#F77F00', alpha=0.7, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(windows)
    ax3.set_xlabel('Time Window', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Breathing Rate (BPM)', fontsize=11, fontweight='bold')
    ax3.set_title('Breathing Rate per Time Window', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for FFT rate
    ax3.axhline(y=breathing_rate_fft, color='red', linestyle='--',
               linewidth=2, label=f'FFT Rate: {breathing_rate_fft:.1f} BPM')
    ax3.legend(loc='upper right')

    # Add rate labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Breath-to-breath intervals histogram
ax4 = fig.add_subplot(gs[2, 0])
if len(peak_frames) > 1:
    intervals_frames = np.diff(peak_frames)
    intervals_seconds = intervals_frames / fps

    ax4.hist(intervals_seconds, bins=20, color='#9B59B6', alpha=0.7, edgecolor='black')
    ax4.axvline(breath_intervals['mean'], color='red', linestyle='--',
               linewidth=2, label=f"Mean: {breath_intervals['mean']:.2f}s")
    ax4.set_xlabel('Interval (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Breath-to-Breath Intervals', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add stats text
    stats_text = (f"Mean: {breath_intervals['mean']:.2f}s\n"
                 f"Std: {breath_intervals['std']:.2f}s\n"
                 f"Min: {breath_intervals['min']:.2f}s\n"
                 f"Max: {breath_intervals['max']:.2f}s")
    ax4.text(0.98, 0.97, stats_text,
            transform=ax4.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)

# Plot 5: Breath interval timeline
ax5 = fig.add_subplot(gs[2, 1])
if len(peak_frames) > 1:
    intervals_seconds = np.diff(peak_frames) / fps
    interval_times = np.array(peak_frames[1:]) / fps

    ax5.plot(interval_times, intervals_seconds, marker='o', linestyle='-',
            color='#06A77D', markersize=4, linewidth=1)
    ax5.axhline(breath_intervals['mean'], color='red', linestyle='--',
               linewidth=2, alpha=0.7, label=f"Mean: {breath_intervals['mean']:.2f}s")
    ax5.fill_between([0, max(interval_times)],
                     breath_intervals['mean'] - breath_intervals['std'],
                     breath_intervals['mean'] + breath_intervals['std'],
                     alpha=0.2, color='red', label='±1 SD')

    ax5.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Interval (seconds)', fontsize=11, fontweight='bold')
    ax5.set_title('Breath Intervals Over Time', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

# Plot 6: Comparison table
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

# Create comparison data
table_data = [
    ['Method', 'Rate (BPM)', 'Notes'],
    ['FFT Analysis', f'{breathing_rate_fft:.1f}', 'Frequency domain'],
]

for window, data in sorted(breath_counts.items(), key=lambda x: (x[0] != 'full', x[0])):
    duration = data.get('duration_s', 0)
    duration_str = f"{duration:.1f}s" if window == 'full' else window
    table_data.append([
        f'Peak Count ({duration_str})',
        f"{data['rate_bpm']:.1f}",
        f"{data['count']} breaths detected"
    ])

# Validation row
if validation:
    status = "Consistent" if validation['is_consistent'] else "Inconsistent"
    table_data.append([
        'Validation',
        f"{validation['mean_rate']:.1f} ± {validation['std_rate']:.1f}",
        f"{status} (CV: {validation['cv']:.2%})"
    ])

table = ax6.table(cellText=table_data, cellLoc='left',
                 loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

ax6.set_title('Breathing Rate Comparison', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Breath Counting Analysis', fontsize=15, fontweight='bold', y=0.995)

# Save plot
output_path = 'data/results/breath_counting_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Breath counting plot saved to: {output_path}")

plt.show()
