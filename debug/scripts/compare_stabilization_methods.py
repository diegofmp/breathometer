"""
Compare all stabilization methods
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from src.pipeline import BreathingAnalyzer

video_path = 'data/videos/bird_sample.mp4'
config_path = 'configs/default.yaml'

# Stabilization methods to compare (plus no stabilization)
stabilization_methods = [
    ('none', False, None),
    ('tracker_position', True, 'tracker_position'),
    ('optical_flow', True, 'optical_flow'),
    ('optical_flow_masked', True, 'optical_flow_masked'),
]

print("="*70)
print("COMPARING STABILIZATION METHODS")
print("="*70)

results_dict = {}

for name, enabled, method in stabilization_methods:
    print(f"\nRunning pipeline with stabilization: {name}")
    print("-" * 70)

    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['stabilization']['enabled'] = enabled
    if method:
        config['stabilization']['method'] = method

    # Save temporary config
    temp_config_path = f'configs/temp_stab_{name}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    try:
        # Run pipeline
        analyzer = BreathingAnalyzer(temp_config_path)
        results = analyzer.process_video(video_path, output_path=None)

        # Store results
        results_dict[name] = {
            'signal': np.array(results['breathing_signal']),
            'breathing_rate': results['breathing_rate_bpm'],
            'confidence': results['confidence'],
            'metadata': results['metadata']
        }

        print(f"✓ {name}: {results['breathing_rate_bpm']:.1f} BPM")
        print(f"  Confidence: {results['confidence']:.2f}")

    except Exception as e:
        print(f"✗ {name} FAILED: {e}")
        results_dict[name] = None

    finally:
        # Clean up temp file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

# Filter out failed methods
successful_methods = {k: v for k, v in results_dict.items() if v is not None}

if not successful_methods:
    print("\nNo methods succeeded!")
    sys.exit(1)

print("\n" + "="*70)
print("GENERATING COMPARISON PLOTS")
print("="*70)

# Create comparison plot
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# Plot 1: All breathing signals overlaid
for name, data in successful_methods.items():
    signal = data['signal']
    axes[0].plot(signal, label=f"{name} ({data['breathing_rate']:.1f} BPM)",
                alpha=0.7, linewidth=1.5)

axes[0].set_ylabel('Breathing Signal', fontsize=12, fontweight='bold')
axes[0].set_title('Breathing Signals - All Stabilization Methods', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Signal derivatives (high-frequency noise)
for name, data in successful_methods.items():
    signal = data['signal']
    diff = np.diff(signal)
    axes[1].plot(diff, label=f"{name} (std={np.std(diff):.4f})",
                alpha=0.6, linewidth=1)

axes[1].set_ylabel('Signal Derivative', fontsize=12, fontweight='bold')
axes[1].set_title('High-Frequency Noise/Jitter Comparison', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

# Plot 3: Hand motion comparison
ax3 = axes[2]
for name, data in successful_methods.items():
    if 'hand_motion' in data['metadata']:
        hand_motion = np.array(data['metadata']['hand_motion'])
        ax3.plot(hand_motion, label=f"{name} (mean={np.mean(hand_motion):.2f}px)",
                alpha=0.7, linewidth=1)

ax3.set_ylabel('Hand Motion (px)', fontsize=12, fontweight='bold')
ax3.set_title('Hand Movement Across Methods', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Chest motion comparison
ax4 = axes[3]
for name, data in successful_methods.items():
    if 'chest_motion' in data['metadata']:
        chest_motion = np.array(data['metadata']['chest_motion'])
        ax4.plot(chest_motion, label=f"{name} (mean={np.mean(chest_motion):.2f}px)",
                alpha=0.7, linewidth=1)

ax4.set_ylabel('Chest Motion (px)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
ax4.set_title('Chest ROI Movement Across Methods', fontsize=13, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = 'data/results/stabilization_methods_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nComparison plot saved to: {output_path}")

# Print summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Method':<20} {'BPM':<8} {'Confidence':<12} {'Signal Std':<12} {'Deriv Std':<12}")
print("-"*70)

for name in successful_methods.keys():
    data = successful_methods[name]
    signal_std = np.std(data['signal'])
    deriv_std = np.std(np.diff(data['signal']))
    print(f"{name:<20} {data['breathing_rate']:<8.1f} {data['confidence']:<12.2f} "
          f"{signal_std:<12.4f} {deriv_std:<12.6f}")

print("="*70)
print("\nLower 'Deriv Std' = Less high-frequency noise = Better stabilization")
print("="*70)

plt.show()
