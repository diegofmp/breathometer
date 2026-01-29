"""
Enhanced metadata visualization including audio and hand/chest motion tracking
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

print("Processing video and collecting enhanced metadata...")
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=None)

# Extract data
breathing_signal = np.array(results['breathing_signal'])
tracking_status = np.array(results['tracking_status'])
metadata = results['metadata']

brightness = np.array(metadata['brightness'])
brightness_change = np.array(metadata['brightness_change'])
motion = np.array(metadata['motion'])
audio_level = np.array(metadata['audio_level'])
hand_motion = np.array(metadata['hand_motion'])
chest_motion = np.array(metadata['chest_motion'])

has_audio = len(audio_level) > 0 and np.any(audio_level > 0)

print(f"\nMetadata collected:")
print(f"  Frames: {len(breathing_signal)}")
print(f"  Brightness range: [{brightness.min():.1f}, {brightness.max():.1f}]")
print(f"  Motion range: [{motion.min():.2f}, {motion.max():.2f}]")
print(f"  Hand motion range: [{hand_motion.min():.2f}, {hand_motion.max():.2f}]")
print(f"  Chest motion range: [{chest_motion.min():.2f}, {chest_motion.max():.2f}]")
if has_audio:
    print(f"  Audio level range: [{audio_level.min():.4f}, {audio_level.max():.4f}]")
else:
    print(f"  Audio: Not available")

# Create comprehensive plot
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(7, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Breathing signal (spans both columns)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(breathing_signal, color='#2E86AB', linewidth=1.5, label='Breathing Signal')
ax1.set_ylabel('Breathing\nSignal', fontsize=11, fontweight='bold')
ax1.set_title('Breathing Analysis with Environmental & Motion Metadata', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Mark lost tracking frames
lost_frames = np.where(tracking_status == 0)[0]
if len(lost_frames) > 0:
    for lf in lost_frames:
        if lf < len(breathing_signal):
            ax1.axvline(x=lf, color='red', alpha=0.15, linewidth=0.5)

# Plot 2: Hand Motion vs Chest Motion (KEY PLOT)
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(hand_motion, color='#FF6B6B', linewidth=1.5, label='Hand Motion', alpha=0.8)
ax2.plot(chest_motion, color='#4ECDC4', linewidth=1.5, label='Chest ROI Motion', alpha=0.8)
ax2.set_ylabel('Motion\n(pixels)', fontsize=11, fontweight='bold')
ax2.set_title('Hand vs Chest ROI Movement (shows stabilization effect)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Add ratio line if stabilization is working
if np.mean(chest_motion) > 0:
    ratio = np.mean(hand_motion) / np.mean(chest_motion)
    ax2.text(0.02, 0.95, f'Hand/Chest ratio: {ratio:.2f}x',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Brightness
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(brightness, color='#F77F00', linewidth=1)
ax3.fill_between(range(len(brightness)), brightness, alpha=0.3, color='#F77F00')
ax3.set_ylabel('Brightness', fontsize=10, fontweight='bold')
ax3.set_title('Frame Brightness', fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Brightness Change
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(brightness_change, color='#EE6C4D', linewidth=1)
ax4.fill_between(range(len(brightness_change)), brightness_change, alpha=0.3, color='#EE6C4D')
ax4.set_ylabel('Brightness\nChange', fontsize=10, fontweight='bold')
ax4.set_title('Brightness Changes', fontsize=11)
ax4.grid(True, alpha=0.3)

# Plot 5: Global Motion
ax5 = fig.add_subplot(gs[3, 0])
ax5.plot(motion, color='#06A77D', linewidth=1)
ax5.fill_between(range(len(motion)), motion, alpha=0.3, color='#06A77D')
ax5.set_ylabel('Global\nMotion', fontsize=10, fontweight='bold')
ax5.set_title('Global Frame Motion', fontsize=11)
ax5.grid(True, alpha=0.3)

# Plot 6: Audio Level
ax6 = fig.add_subplot(gs[3, 1])
if has_audio:
    ax6.plot(audio_level, color='#9B59B6', linewidth=1, label='Audio RMS')
    ax6.fill_between(range(len(audio_level)), audio_level, alpha=0.3, color='#9B59B6')
    ax6.set_ylabel('Audio\nLevel (RMS)', fontsize=10, fontweight='bold')
    ax6.set_title('Audio Noise Level', fontsize=11)
    ax6.grid(True, alpha=0.3)

    # Highlight loud moments
    if np.max(audio_level) > 0:
        threshold = np.percentile(audio_level[audio_level > 0], 90)
        loud_moments = np.where(audio_level > threshold)[0]
        ax6.scatter(loud_moments, audio_level[loud_moments],
                   color='red', s=20, zorder=5, alpha=0.6, label=f'Loud (>{threshold:.3f})')
        ax6.legend(loc='upper right', fontsize=8)
else:
    ax6.text(0.5, 0.5, 'No audio available',
            ha='center', va='center', fontsize=14, transform=ax6.transAxes)
    ax6.set_title('Audio Noise Level', fontsize=11)

# Plot 7: Hand Motion Detail
ax7 = fig.add_subplot(gs[4, 0])
ax7.plot(hand_motion, color='#FF6B6B', linewidth=1.2)
ax7.fill_between(range(len(hand_motion)), hand_motion, alpha=0.4, color='#FF6B6B')
ax7.set_ylabel('Hand Motion\n(pixels)', fontsize=10, fontweight='bold')
ax7.set_title(f'Hand Movement (mean: {np.mean(hand_motion):.2f} px)', fontsize=11)
ax7.grid(True, alpha=0.3)

# Plot 8: Chest Motion Detail
ax8 = fig.add_subplot(gs[4, 1])
ax8.plot(chest_motion, color='#4ECDC4', linewidth=1.2)
ax8.fill_between(range(len(chest_motion)), chest_motion, alpha=0.4, color='#4ECDC4')
ax8.set_ylabel('Chest Motion\n(pixels)', fontsize=10, fontweight='bold')
ax8.set_title(f'Chest ROI Movement (mean: {np.mean(chest_motion):.2f} px)', fontsize=11)
ax8.grid(True, alpha=0.3)

# Plot 9: Breathing vs Audio correlation
ax9 = fig.add_subplot(gs[5, 0])
if has_audio and len(audio_level) == len(breathing_signal):
    ax9.scatter(audio_level, breathing_signal, alpha=0.3, s=10, c=range(len(audio_level)), cmap='viridis')
    ax9.set_xlabel('Audio Level', fontsize=10)
    ax9.set_ylabel('Breathing Signal', fontsize=10)
    ax9.set_title('Breathing vs Audio Scatter', fontsize=11)
    ax9.grid(True, alpha=0.3)

    # Calculate correlation
    if np.std(audio_level) > 0:
        corr = np.corrcoef(breathing_signal, audio_level)[0, 1]
        ax9.text(0.05, 0.95, f'r = {corr:.3f}',
                transform=ax9.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow' if abs(corr) > 0.3 else 'white', alpha=0.7))
else:
    ax9.text(0.5, 0.5, 'Audio not available',
            ha='center', va='center', fontsize=12, transform=ax9.transAxes)
    ax9.set_title('Breathing vs Audio Scatter', fontsize=11)

# Plot 10: Tracking Status
ax10 = fig.add_subplot(gs[5, 1])
ax10.fill_between(range(len(tracking_status)), tracking_status,
                  alpha=0.6, color='green', step='mid')
ax10.fill_between(range(len(tracking_status)), 0, 1,
                  where=(tracking_status==0), alpha=0.6, color='red', step='mid')
ax10.set_ylabel('Status', fontsize=10, fontweight='bold')
ax10.set_title(f'Tracking Status ({np.mean(tracking_status)*100:.1f}% success)', fontsize=11)
ax10.set_ylim([-0.1, 1.1])
ax10.set_yticks([0, 1])
ax10.set_yticklabels(['Lost', 'OK'])
ax10.grid(True, alpha=0.3, axis='x')

# Plot 11: Correlation matrix (bottom spanning both columns)
ax11 = fig.add_subplot(gs[6, :])

# Prepare data for correlation
min_len = min(len(breathing_signal), len(brightness), len(motion),
             len(hand_motion), len(chest_motion))
data_dict = {
    'Breathing': breathing_signal[:min_len],
    'Brightness': brightness[:min_len],
    'Bright.Change': brightness_change[:min_len],
    'Motion': motion[:min_len],
    'Hand Motion': hand_motion[:min_len],
    'Chest Motion': chest_motion[:min_len],
}

if has_audio and len(audio_level) >= min_len:
    data_dict['Audio'] = audio_level[:min_len]

# Calculate correlation matrix
labels = list(data_dict.keys())
n = len(labels)
corr_matrix = np.zeros((n, n))

for i, key1 in enumerate(labels):
    for j, key2 in enumerate(labels):
        if np.std(data_dict[key1]) > 0 and np.std(data_dict[key2]) > 0:
            corr_matrix[i, j] = np.corrcoef(data_dict[key1], data_dict[key2])[0, 1]

# Plot heatmap
im = ax11.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax11.set_xticks(range(n))
ax11.set_yticks(range(n))
ax11.set_xticklabels(labels, rotation=45, ha='right')
ax11.set_yticklabels(labels)
ax11.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(n):
    for j in range(n):
        text = ax11.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                        fontsize=9)

plt.colorbar(im, ax=ax11, label='Correlation')

# Set common x-label for bottom plots
for ax in [ax9, ax10]:
    ax.set_xlabel('Frame Number', fontsize=10, fontweight='bold')

# Save plot
output_path = 'data/results/metadata_enhanced.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nEnhanced metadata plot saved to: {output_path}")

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print(f"Hand movement (mean):   {np.mean(hand_motion):.2f} pixels")
print(f"Chest movement (mean):  {np.mean(chest_motion):.2f} pixels")
if np.mean(chest_motion) > 0:
    print(f"Hand/Chest ratio:       {np.mean(hand_motion)/np.mean(chest_motion):.2f}x")
    print(f"  → Hand moves {np.mean(hand_motion)/np.mean(chest_motion):.1f}x more than chest")
print("="*70)

plt.show()
