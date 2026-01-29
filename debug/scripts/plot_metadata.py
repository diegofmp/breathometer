"""
Visualize breathing signal with environmental metadata
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from src.pipeline import BreathingAnalyzer

# Setup paths
config_path = 'configs/default.yaml'
video_path = 'data/videos/bird_sample.mp4'

print("Processing video and collecting metadata...")
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=None)

# Extract data
breathing_signal = np.array(results['breathing_signal'])
tracking_status = np.array(results['tracking_status'])
metadata = results['metadata']

brightness = np.array(metadata['brightness'])
brightness_change = np.array(metadata['brightness_change'])
motion = np.array(metadata['motion'])

print(f"\nMetadata collected:")
print(f"  Brightness range: [{brightness.min():.1f}, {brightness.max():.1f}]")
print(f"  Brightness change max: {brightness_change.max():.2f}")
print(f"  Motion range: [{motion.min():.2f}, {motion.max():.2f}]")

# Create comprehensive plot
fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)

# Plot 1: Breathing signal
axes[0].plot(breathing_signal, color='#2E86AB', linewidth=1.5, label='Breathing Signal')
axes[0].set_ylabel('Breathing\nSignal', fontsize=11, fontweight='bold')
axes[0].set_title('Breathing Signal with Environmental Metadata', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right')

# Mark lost tracking frames
lost_frames = np.where(tracking_status == 0)[0]
if len(lost_frames) > 0:
    for lf in lost_frames:
        if lf < len(breathing_signal):
            axes[0].axvline(x=lf, color='red', alpha=0.15, linewidth=0.5)

# Plot 2: Brightness
axes[1].plot(brightness, color='#F77F00', linewidth=1, label='Brightness')
axes[1].fill_between(range(len(brightness)), brightness, alpha=0.3, color='#F77F00')
axes[1].set_ylabel('Brightness\n(Mean Intensity)', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')

# Plot 3: Brightness Change
axes[2].plot(brightness_change, color='#EE6C4D', linewidth=1, label='Brightness Change')
axes[2].fill_between(range(len(brightness_change)), brightness_change, alpha=0.3, color='#EE6C4D')
axes[2].set_ylabel('Brightness\nChange', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper right')

# Highlight significant brightness changes
threshold = np.percentile(brightness_change, 95)  # 95th percentile
significant_changes = np.where(brightness_change > threshold)[0]
axes[2].scatter(significant_changes, brightness_change[significant_changes],
               color='red', s=20, zorder=5, alpha=0.5, label=f'> {threshold:.1f}')

# Plot 4: Motion
axes[3].plot(motion, color='#06A77D', linewidth=1, label='Global Motion')
axes[3].fill_between(range(len(motion)), motion, alpha=0.3, color='#06A77D')
axes[3].set_ylabel('Global\nMotion', fontsize=11, fontweight='bold')
axes[3].grid(True, alpha=0.3)
axes[3].legend(loc='upper right')

# Plot 5: Tracking Status
axes[4].fill_between(range(len(tracking_status)), tracking_status,
                     alpha=0.6, color='green', label='Tracked', step='mid')
axes[4].fill_between(range(len(tracking_status)), 0, 1,
                     where=(tracking_status==0), alpha=0.6, color='red',
                     label='Lost', step='mid')
axes[4].set_ylabel('Tracking\nStatus', fontsize=11, fontweight='bold')
axes[4].set_xlabel('Frame Number', fontsize=12, fontweight='bold')
axes[4].set_ylim([-0.1, 1.1])
axes[4].set_yticks([0, 1])
axes[4].set_yticklabels(['Lost', 'Tracked'])
axes[4].grid(True, alpha=0.3, axis='x')
axes[4].legend(loc='upper right')

plt.tight_layout()

# Save plot
output_path = 'data/results/metadata_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nMetadata plot saved to: {output_path}")

# Analyze correlations
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Truncate all to same length
min_len = min(len(breathing_signal), len(brightness), len(motion))
breath_truncated = breathing_signal[:min_len]
brightness_truncated = brightness[:min_len]
brightness_change_truncated = brightness_change[:min_len]
motion_truncated = motion[:min_len]

# Calculate correlations
corr_brightness = np.corrcoef(breath_truncated, brightness_truncated)[0, 1]
corr_brightness_change = np.corrcoef(breath_truncated, brightness_change_truncated)[0, 1]
corr_motion = np.corrcoef(breath_truncated, motion_truncated)[0, 1]

print(f"Breathing vs Brightness:        {corr_brightness:>8.3f}")
print(f"Breathing vs Brightness Change: {corr_brightness_change:>8.3f}")
print(f"Breathing vs Global Motion:     {corr_motion:>8.3f}")
print("="*60)

if abs(corr_brightness) > 0.3:
    print(f"⚠ Warning: High correlation with brightness ({corr_brightness:.3f})")
    print("  Lighting changes may be affecting measurements")

if abs(corr_motion) > 0.3:
    print(f"⚠ Warning: High correlation with motion ({corr_motion:.3f})")
    print("  Camera/hand movement may be affecting measurements")

plt.show()
