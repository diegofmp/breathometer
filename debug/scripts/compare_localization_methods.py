"""
Compare different localization methods and their effect on breathing measurements
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

# Localization methods to compare
localization_methods = ['simple', 'contour', 'motion', 'variance', 'optical_flow']

print("="*70)
print("COMPARING LOCALIZATION METHODS")
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
            'signal_length': results['signal_length'],
            'tracking_status': np.array(results['tracking_status'])
        }

        print(f"✓ {method}: {results['breathing_rate_bpm']:.1f} BPM")
        print(f"  Signal length: {results['signal_length']}")
        print(f"  Confidence: {results['confidence']:.2f}")
        print(f"  Tracking success: {np.mean(results['tracking_status'])*100:.1f}%")

    except Exception as e:
        print(f"✗ {method} FAILED: {e}")
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

# Create comprehensive comparison plot
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot 1: All breathing signals overlaid
ax1 = fig.add_subplot(gs[0:2, :])
for method, data in successful_methods.items():
    signal = data['signal']
    ax1.plot(signal, label=f"{method} ({data['breathing_rate']:.1f} BPM)",
             alpha=0.7, linewidth=1.5)

ax1.set_xlabel('Frame', fontsize=12)
ax1.set_ylabel('Breathing Signal', fontsize=12)
ax1.set_title('Breathing Signals Comparison - All Localization Methods', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Breathing rates comparison (bar chart)
ax2 = fig.add_subplot(gs[2, 0])
methods_names = list(successful_methods.keys())
breathing_rates = [successful_methods[m]['breathing_rate'] for m in methods_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(methods_names)))

bars = ax2.bar(methods_names, breathing_rates, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Breathing Rate (BPM)', fontsize=11)
ax2.set_title('Breathing Rate by Method', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, rate in zip(bars, breathing_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Signal statistics comparison
ax3 = fig.add_subplot(gs[2, 1])
signal_stds = [np.std(successful_methods[m]['signal']) for m in methods_names]
bars = ax3.bar(methods_names, signal_stds, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Signal Std Dev', fontsize=11)
ax3.set_title('Signal Variability by Method', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels
for bar, std in zip(bars, signal_stds):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{std:.3f}',
            ha='center', va='bottom', fontsize=9)

# Plot 4: Tracking success rate
ax4 = fig.add_subplot(gs[3, 0])
tracking_success = [np.mean(successful_methods[m]['tracking_status'])*100 for m in methods_names]
bars = ax4.bar(methods_names, tracking_success, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Tracking Success (%)', fontsize=11)
ax4.set_title('Tracking Success Rate by Method', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.3, axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels
for bar, success in zip(bars, tracking_success):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{success:.1f}%',
            ha='center', va='bottom', fontsize=9)

# Plot 5: Confidence scores
ax5 = fig.add_subplot(gs[3, 1])
confidences = [successful_methods[m]['confidence'] for m in methods_names]
bars = ax5.bar(methods_names, confidences, color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Confidence', fontsize=11)
ax5.set_title('Signal Confidence by Method', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 1.05])
ax5.grid(True, alpha=0.3, axis='y')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels
for bar, conf in zip(bars, confidences):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{conf:.2f}',
            ha='center', va='bottom', fontsize=9)

plt.suptitle('Localization Methods Comparison', fontsize=16, fontweight='bold', y=0.995)

# Save plot
output_path = 'data/results/localization_methods_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nComparison plot saved to: {output_path}")

# Print summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Method':<15} {'BPM':<8} {'Confidence':<12} {'Signal Std':<12} {'Tracking %':<12}")
print("-"*70)
for method in methods_names:
    data = successful_methods[method]
    print(f"{method:<15} {data['breathing_rate']:<8.1f} {data['confidence']:<12.2f} "
          f"{np.std(data['signal']):<12.4f} {np.mean(data['tracking_status'])*100:<12.1f}")
print("="*70)

plt.show()
