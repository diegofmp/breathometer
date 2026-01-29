"""
Compare all chest localization methods
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

# All localization methods to compare
localization_methods = [
    'simple',
    'contour',
    'variance',
    'motion',
    'optical_flow'
]

print("="*70)
print("COMPARING ALL CHEST LOCALIZATION METHODS")
print("="*70)

results_dict = {}

for method in localization_methods:
    print(f"\nRunning pipeline with localization method: {method}")
    print("-" * 70)

    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['localization']['method'] = method

    # Save temporary config
    temp_config_path = f'configs/temp_loc_{method}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    try:
        # Run pipeline
        analyzer = BreathingAnalyzer(temp_config_path)
        results = analyzer.process_video(video_path, output_path=None)

        # Store results
        results_dict[method] = {
            'signal': np.array(results['breathing_signal']),
            'breathing_rate': results['breathing_rate_bpm'],
            'confidence': results['confidence'],
            'metadata': results['metadata']
        }

        print(f"✓ {method}: {results['breathing_rate_bpm']:.1f} BPM")
        print(f"  Confidence: {results['confidence']:.2f}")
        print(f"  Signal length: {len(results['breathing_signal'])} frames")

    except Exception as e:
        print(f"✗ {method} FAILED: {e}")
        import traceback
        traceback.print_exc()
        results_dict[method] = None

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
n_methods = len(successful_methods)
fig, axes = plt.subplots(n_methods + 1, 1, figsize=(16, 4 * (n_methods + 1)))

if n_methods == 1:
    axes = [axes]

# Plot 1: Overlay all signals
for method, data in successful_methods.items():
    signal = data['signal']
    axes[0].plot(signal, label=f"{method} ({data['breathing_rate']:.1f} BPM)",
                alpha=0.7, linewidth=1.5)

axes[0].set_ylabel('Breathing Signal', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Frame Number', fontsize=12, fontweight='bold')
axes[0].set_title('Breathing Signals - All Localization Methods', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot individual signals
for idx, (method, data) in enumerate(successful_methods.items(), start=1):
    signal = data['signal']
    axes[idx].plot(signal, color='#0066CC', linewidth=1.5)
    axes[idx].set_ylabel('Signal', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{method.upper()}: {data["breathing_rate"]:.1f} BPM '
                       f'(Confidence: {data["confidence"]:.2f})',
                       fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

    # Add stats
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    stats_text = f"Mean: {signal_mean:.2f}\nStd: {signal_std:.2f}\nSNR: {signal_mean/signal_std:.2f}"
    axes[idx].text(0.98, 0.97, stats_text,
                  transform=axes[idx].transAxes,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                  fontsize=9)

axes[-1].set_xlabel('Frame Number', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save plot
output_path = 'data/results/localization_methods_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nComparison plot saved to: {output_path}")

# Print summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Method':<15} {'BPM':<8} {'Confidence':<12} {'Signal Mean':<12} {'Signal Std':<12} {'SNR':<8}")
print("-"*70)

for method in successful_methods.keys():
    data = successful_methods[method]
    signal = data['signal']
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    snr = signal_mean / signal_std if signal_std > 0 else 0
    print(f"{method:<15} {data['breathing_rate']:<8.1f} {data['confidence']:<12.2f} "
          f"{signal_mean:<12.2f} {signal_std:<12.4f} {snr:<8.2f}")

print("="*70)
print("\nHigher SNR = Better signal quality")
print("Higher Confidence = More reliable breathing rate estimate")
print("="*70)

plt.show()
