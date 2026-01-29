"""
Analyze the effect of stabilization by measuring motion inside chest ROI
"""

import sys
sys.path.append('.')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from src.pipeline import BreathingAnalyzer

video_path = 'data/videos/bird_sample.mp4'
config_path = 'configs/default.yaml'

print("Analyzing stabilization effect...")

# Load and modify config for WITHOUT stabilization
print("Running pipeline WITHOUT stabilization...")
with open(config_path, 'r') as f:
    config_no_stab = yaml.safe_load(f)

config_no_stab['stabilization']['enabled'] = False

# Save temporary config
temp_config_path_no_stab = 'configs/temp_no_stab.yaml'
with open(temp_config_path_no_stab, 'w') as f:
    yaml.dump(config_no_stab, f)

analyzer_no_stab = BreathingAnalyzer(temp_config_path_no_stab)
results_no_stab = analyzer_no_stab.process_video(video_path, output_path=None)

# Load and modify config for WITH stabilization
print("\nRunning pipeline WITH stabilization...")
with open(config_path, 'r') as f:
    config_with_stab = yaml.safe_load(f)

config_with_stab['stabilization']['enabled'] = True

# Save temporary config
temp_config_path_with_stab = 'configs/temp_with_stab.yaml'
with open(temp_config_path_with_stab, 'w') as f:
    yaml.dump(config_with_stab, f)

analyzer_with_stab = BreathingAnalyzer(temp_config_path_with_stab)
results_with_stab = analyzer_with_stab.process_video(video_path, output_path=None)

# Clean up temp files
import os
os.remove(temp_config_path_no_stab)
os.remove(temp_config_path_with_stab)

# Get signals
signal_no_stab = np.array(results_no_stab['breathing_signal'])
signal_with_stab = np.array(results_with_stab['breathing_signal'])

# Truncate to same length
min_len = min(len(signal_no_stab), len(signal_with_stab))
signal_no_stab = signal_no_stab[:min_len]
signal_with_stab = signal_with_stab[:min_len]

# Calculate statistics
print("\n" + "="*60)
print("STABILIZATION EFFECT ANALYSIS")
print("="*60)
print(f"\nWithout Stabilization:")
print(f"  Signal length: {len(signal_no_stab)}")
print(f"  Signal std: {np.std(signal_no_stab):.4f}")
print(f"  Signal range: [{np.min(signal_no_stab):.4f}, {np.max(signal_no_stab):.4f}]")
print(f"  Breathing rate: {results_no_stab['breathing_rate_bpm']:.1f} BPM")

print(f"\nWith Stabilization:")
print(f"  Signal length: {len(signal_with_stab)}")
print(f"  Signal std: {np.std(signal_with_stab):.4f}")
print(f"  Signal range: [{np.min(signal_with_stab):.4f}, {np.max(signal_with_stab):.4f}]")
print(f"  Breathing rate: {results_with_stab['breathing_rate_bpm']:.1f} BPM")

# Calculate high-frequency noise (differentiate to see jitter)
diff_no_stab = np.diff(signal_no_stab)
diff_with_stab = np.diff(signal_with_stab)

print(f"\nHigh-frequency noise (std of derivative):")
print(f"  Without stabilization: {np.std(diff_no_stab):.6f}")
print(f"  With stabilization: {np.std(diff_with_stab):.6f}")
print(f"  Reduction: {(1 - np.std(diff_with_stab)/np.std(diff_no_stab))*100:.1f}%")

# Create comparison plots
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Raw signals
axes[0].plot(signal_no_stab, label='Without Stabilization', alpha=0.7, linewidth=1)
axes[0].plot(signal_with_stab, label='With Stabilization', alpha=0.7, linewidth=1)
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Breathing Signal')
axes[0].set_title('Raw Breathing Signals Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: High-frequency components (derivative)
axes[1].plot(diff_no_stab, label='Without Stabilization', alpha=0.7, linewidth=0.5)
axes[1].plot(diff_with_stab, label='With Stabilization', alpha=0.7, linewidth=0.5)
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Signal Derivative (Motion/Jitter)')
axes[1].set_title('High-Frequency Noise/Jitter Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Power spectral density
from scipy import signal as scipy_signal

fps = results_no_stab['video_fps']
freqs_no_stab, psd_no_stab = scipy_signal.welch(signal_no_stab, fs=fps, nperseg=min(256, len(signal_no_stab)))
freqs_with_stab, psd_with_stab = scipy_signal.welch(signal_with_stab, fs=fps, nperseg=min(256, len(signal_with_stab)))

axes[2].semilogy(freqs_no_stab * 60, psd_no_stab, label='Without Stabilization', alpha=0.7)
axes[2].semilogy(freqs_with_stab * 60, psd_with_stab, label='With Stabilization', alpha=0.7)
axes[2].set_xlabel('Frequency (BPM)')
axes[2].set_ylabel('Power Spectral Density')
axes[2].set_title('Frequency Domain Comparison')
axes[2].set_xlim([0, 240])  # 0-240 BPM
axes[2].axvspan(30, 240, alpha=0.1, color='green', label='Breathing range')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/results/stabilization_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print(f"Analysis plot saved to: data/results/stabilization_analysis.png")
print(f"{'='*60}")
plt.show()
