"""
Visualize tracking status and breathing signal
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from src.pipeline import BreathingAnalyzer

# Setup paths
config_path = 'configs/default.yaml'
video_path = 'data/videos/bird_sample.mp4'

# Run pipeline (without output video for faster processing)
print("Processing video...")
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=None)

# Extract data
tracking_status = np.array(results['tracking_status'])
breathing_signal = np.array(results['breathing_signal'])
total_frames = results['total_frames']

# Find lost frames
lost_frames = np.where(tracking_status == 0)[0]
tracked_frames = np.where(tracking_status == 1)[0]

print(f"\nTracking Statistics:")
print(f"Total frames: {total_frames}")
print(f"Successfully tracked: {len(tracked_frames)} ({len(tracked_frames)/total_frames*100:.1f}%)")
print(f"Lost tracking: {len(lost_frames)} ({len(lost_frames)/total_frames*100:.1f}%)")
print(f"Signal length: {len(breathing_signal)}")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Plot 1: Tracking status
axes[0].plot(tracking_status, linewidth=0.5)
axes[0].fill_between(range(len(tracking_status)), 0, tracking_status,
                      where=(tracking_status==1), alpha=0.3, color='green', label='Tracked')
axes[0].fill_between(range(len(tracking_status)), 0, 1,
                      where=(tracking_status==0), alpha=0.3, color='red', label='Lost')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Tracking Status')
axes[0].set_title('ROI Tracking Status Over Time')
axes[0].set_ylim(-0.1, 1.1)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Breathing signal with lost frames marked
axes[1].plot(breathing_signal, linewidth=1, label='Breathing Signal')
# Mark where tracking was lost (gaps in signal)
for i, status in enumerate(tracking_status):
    if status == 0 and i < len(breathing_signal):
        axes[1].axvline(x=i, color='red', alpha=0.2, linewidth=0.5)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Breathing Magnitude')
axes[1].set_title('Raw Breathing Signal (red lines = lost tracking)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Histogram of lost frame positions
if len(lost_frames) > 0:
    axes[2].hist(lost_frames, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Frame Number')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Lost Frames')
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'No frames lost!',
                ha='center', va='center', fontsize=20, transform=axes[2].transAxes)
    axes[2].set_title('Distribution of Lost Frames')

plt.tight_layout()
plt.savefig('data/results/tracking_status.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: data/results/tracking_status.png")
plt.show()
