"""
Multi-Method Comparison Tool

Compares performance of different breath counting configurations.
Currently supports autocorrelation-based windowed ACF method with different parameters.

This script helps identify:
1. Which method performs best overall
2. Which videos are problematic across all methods (likely ground truth issues)
3. Which videos show method-specific failures (method limitations)
4. Whether an ensemble approach could improve results

Usage:
    python src/tuning/compare_methods.py \
        --cache-dir cache/ \
        --configs configs/acf_10s.yaml configs/acf_20s.yaml configs/acf_30s.yaml \
        --output-dir diagnostics/comparison/

The script will:
- Run diagnostics on each config
- Generate comparison tables and plots
- Identify consensus vs disagreement cases
- Recommend best method or ensemble strategy
"""

import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tuning.signal_cache import SignalCache
from src.signal_processing import SignalProcessor


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_signal_with_config(signal_data: Dict, config: Dict, config_name: str) -> Dict:
    """
    Evaluate a signal with a specific configuration.

    The cached signal always represents 60s. If the config has tracking.max_frames set,
    the signal is truncated to that many frames before processing.

    Returns:
        Dictionary with evaluation results
    """
    raw_signal = signal_data['raw_signal']
    fps = signal_data['fps']
    ground_truth_bpm = signal_data.get('ground_truth_bpm')

    if ground_truth_bpm is None:
        return None

    # Truncate signal to config's max_frames (cache always holds full 60s)
    max_frames = config.get('tracking', {}).get('max_frames')
    if max_frames is not None and max_frames < len(raw_signal):
        raw_signal = raw_signal[:max_frames]

    signal_duration_s = len(raw_signal) / fps

    # Process signal
    processor = SignalProcessor(config.get('signal_processing', {}))
    breathing_rate_bpm, info = processor.estimate_breathing_rate(raw_signal, fps)
    breath_counts = info.get('breath_counts', {})

    # Get 60s window BPM (matches ground truth measurement)
    window_60s = breath_counts.get('60s', {})
    if window_60s:
        detected_bpm = window_60s.get('rate_bpm', 0)
    else:
        # Fallback to full window
        full_window = breath_counts.get('full', {})
        if full_window:
            detected_bpm = full_window.get('rate_bpm', 0)
        else:
            # Fallback to main return value
            detected_bpm = breathing_rate_bpm

    # Calculate error
    absolute_error = abs(detected_bpm - ground_truth_bpm)
    relative_error = absolute_error / ground_truth_bpm if ground_truth_bpm > 0 else 0

    return {
        'config_name': config_name,
        'detected_bpm': detected_bpm,
        'ground_truth_bpm': ground_truth_bpm,
        'absolute_error': absolute_error,
        'relative_error_pct': relative_error * 100,
        'signal_duration_s': signal_duration_s,
    }


def compare_methods(cache_dir: str, config_paths: List[str], output_dir: str):
    """
    Compare multiple breath counting methods/configurations

    Args:
        cache_dir: Directory with cached signals
        config_paths: List of configuration file paths
        output_dir: Directory to save comparison outputs
    """
    print(f"\n{'='*70}")
    print(f"MULTI-METHOD COMPARISON")
    print(f"{'='*70}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configurations
    configs = {}
    for config_path in config_paths:
        config_name = Path(config_path).stem  # e.g., 'tuned_peak', 'tuned_emv'
        print(f"\nLoading config: {config_name} ({config_path})")
        configs[config_name] = load_config(config_path)

    # Load cached signals
    print(f"\nLoading cached signals from: {cache_dir}")
    signal_cache = SignalCache(cache_dir)
    cached_signals = signal_cache.get_signals_with_ground_truth()

    if not cached_signals:
        print("✗ Error: No cached signals with ground truth found")
        return

    print(f"✓ Loaded {len(cached_signals)} signals with ground truth\n")

    # Evaluate each signal with each config
    all_results = []

    for idx, signal_data in enumerate(cached_signals, 1):
        video_name = Path(signal_data['video_path']).name
        print(f"\n[{idx}/{len(cached_signals)}] Evaluating: {video_name}")

        for config_name, config in configs.items():
            result = evaluate_signal_with_config(signal_data, config, config_name)
            if result:
                result['video_name'] = video_name
                all_results.append(result)
                print(f"  {config_name:20s}: {result['detected_bpm']:6.1f} BPM "
                      f"(GT: {result['ground_truth_bpm']:6.1f}, error: {result['relative_error_pct']:5.1f}%, "
                      f"using {result['signal_duration_s']:.0f}s)")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Pivot for easier comparison
    pivot_detected = df.pivot(index='video_name', columns='config_name', values='detected_bpm')
    pivot_error = df.pivot(index='video_name', columns='config_name', values='relative_error_pct')
    pivot_gt = df.groupby('video_name')['ground_truth_bpm'].first()

    # Add ground truth column
    comparison_df = pivot_detected.copy()
    comparison_df.insert(0, 'ground_truth', pivot_gt)

    # Add error columns
    for config_name in configs.keys():
        comparison_df[f'{config_name}_error'] = pivot_error[config_name]

    # Calculate summary statistics
    summary_stats = {}
    for config_name in configs.keys():
        errors = pivot_error[config_name].dropna()
        # Signal duration is the same for all videos under a given config
        durations = df[df['config_name'] == config_name]['signal_duration_s']
        max_frames = configs[config_name].get('tracking', {}).get('max_frames')
        summary_stats[config_name] = {
            'mean_error_pct': errors.mean(),
            'median_error_pct': errors.median(),
            'std_error_pct': errors.std(),
            'min_error_pct': errors.min(),
            'max_error_pct': errors.max(),
            'count': len(errors),
            'signal_duration_s': durations.mean(),
            'max_frames': max_frames,
        }

    # Save detailed comparison
    csv_path = output_path / 'method_comparison.csv'
    comparison_df.to_csv(csv_path)
    print(f"\n✓ Detailed comparison saved to: {csv_path}")

    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats).T
    summary_csv_path = output_path / 'method_summary.csv'
    summary_df.to_csv(summary_csv_path)
    print(f"✓ Summary statistics saved to: {summary_csv_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}\n")

    for config_name, stats in summary_stats.items():
        print(f"{config_name}:")
        print(f"  Signal used:  {stats['signal_duration_s']:.0f}s  (max_frames={stats['max_frames']})")
        print(f"  Mean error:   {stats['mean_error_pct']:6.2f}%")
        print(f"  Median error: {stats['median_error_pct']:6.2f}%")
        print(f"  Std error:    {stats['std_error_pct']:6.2f}%")
        print(f"  Range:        [{stats['min_error_pct']:5.2f}%, {stats['max_error_pct']:5.2f}%]")
        print()

    # Find best method per video
    best_method_per_video = pivot_error.idxmin(axis=1)
    worst_method_per_video = pivot_error.idxmax(axis=1)

    # Count wins per method
    method_wins = best_method_per_video.value_counts()
    print(f"Method Performance (wins):")
    for method, count in method_wins.items():
        print(f"  {method:20s}: {count:3d} videos ({count/len(best_method_per_video)*100:5.1f}%)")
    print()

    # Identify problematic videos (high error across all methods)
    mean_error_per_video = pivot_error.mean(axis=1)
    problematic_videos = mean_error_per_video[mean_error_per_video > 20].sort_values(ascending=False)

    if len(problematic_videos) > 0:
        print(f"Problematic videos (>20% error across all methods):")
        for video, error in problematic_videos.items():
            print(f"  {video:40s}: {error:6.2f}% avg error")
        print(f"\n→ These {len(problematic_videos)} videos likely have ground truth or quality issues\n")

    # Identify videos with high method disagreement
    std_error_per_video = pivot_error.std(axis=1)
    disagreement_videos = std_error_per_video.sort_values(ascending=False).head(10)

    print(f"Videos with highest method disagreement (top 10):")
    for video, std in disagreement_videos.items():
        print(f"  {video:40s}: {std:6.2f}% std deviation")
        methods_str = "  Methods: "
        for config_name in configs.keys():
            error = pivot_error.loc[video, config_name]
            methods_str += f"{config_name}={error:.1f}% "
        print(methods_str)
    print()

    # Create visualization plots
    create_comparison_plots(pivot_error, pivot_detected, pivot_gt, configs, output_path)

    # Save analysis JSON
    analysis_data = {
        'summary_stats': summary_stats,
        'method_wins': method_wins.to_dict(),
        'problematic_videos': problematic_videos.to_dict(),
        'best_method_per_video': best_method_per_video.to_dict(),
    }

    json_path = output_path / 'comparison_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Analysis data saved to: {json_path}")

    # Recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}\n")

    best_overall = min(summary_stats.items(), key=lambda x: x[1]['mean_error_pct'])
    print(f"1. Best overall method: {best_overall[0]}")
    print(f"   Mean error: {best_overall[1]['mean_error_pct']:.2f}%\n")

    # Check if ensemble would help
    ensemble_potential = (std_error_per_video.mean() > 5)
    if ensemble_potential:
        print(f"2. Ensemble approach recommended:")
        print(f"   High disagreement between methods (avg std: {std_error_per_video.mean():.2f}%)")
        print(f"   Consider averaging predictions or selecting best method per video\n")
    else:
        print(f"2. Ensemble approach NOT recommended:")
        print(f"   Low disagreement between methods (avg std: {std_error_per_video.mean():.2f}%)")
        print(f"   Methods make similar predictions\n")

    if len(problematic_videos) > 0:
        print(f"3. Ground truth validation needed:")
        print(f"   {len(problematic_videos)} videos have >20% error across all methods")
        print(f"   Review these videos manually\n")

    print(f"{'='*70}\n")


def create_comparison_plots(pivot_error: pd.DataFrame, pivot_detected: pd.DataFrame,
                            pivot_gt: pd.Series, configs: Dict, output_path: Path):
    """
    Create comprehensive comparison plots
    """
    print(f"\nGenerating comparison plots...")

    config_names = list(configs.keys())
    n_methods = len(config_names)

    # 1. Error distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Box plot of errors
    pivot_error.boxplot(ax=axes[0, 0])
    axes[0, 0].set_title('Error Distribution by Method', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Relative Error (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Histogram comparison
    colors = ['#2E86AB', '#A23E48', '#06A77D', '#F77F00', '#6A4C93']
    for i, config_name in enumerate(config_names):
        axes[0, 1].hist(pivot_error[config_name].dropna(), bins=20, alpha=0.5,
                       label=config_name, color=colors[i % len(colors)])
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Error Histograms', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Method agreement scatter (first two methods)
    if n_methods >= 2:
        method1, method2 = config_names[0], config_names[1]
        axes[1, 0].scatter(pivot_error[method1], pivot_error[method2], alpha=0.6, s=100)
        axes[1, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.5)  # Perfect agreement line
        axes[1, 0].set_xlabel(f'{method1} Error (%)')
        axes[1, 0].set_ylabel(f'{method2} Error (%)')
        axes[1, 0].set_title(f'Method Agreement: {method1} vs {method2}', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Calculate correlation
        corr = pivot_error[method1].corr(pivot_error[method2])
        axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[1, 0].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Mean error per method (bar chart)
    mean_errors = pivot_error.mean()
    axes[1, 1].bar(range(len(mean_errors)), mean_errors.values,
                   color=[colors[i % len(colors)] for i in range(len(mean_errors))])
    axes[1, 1].set_xticks(range(len(mean_errors)))
    axes[1, 1].set_xticklabels(mean_errors.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Mean Error (%)')
    axes[1, 1].set_title('Mean Error by Method', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add error bars (std)
    std_errors = pivot_error.std()
    axes[1, 1].errorbar(range(len(mean_errors)), mean_errors.values,
                       yerr=std_errors.values, fmt='none', color='black',
                       capsize=5, capthick=2)

    plt.tight_layout()
    plot_path = output_path / 'method_comparison_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison plots saved to: {plot_path}")

    # 2. Per-video comparison (heatmap)
    fig, ax = plt.subplots(figsize=(max(12, n_methods * 3), min(20, len(pivot_error) * 0.3)))

    # Create heatmap
    im = ax.imshow(pivot_error.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=30)

    # Set ticks
    ax.set_xticks(np.arange(len(config_names)))
    ax.set_yticks(np.arange(len(pivot_error)))
    ax.set_xticklabels(config_names)
    ax.set_yticklabels(pivot_error.index)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Error (%)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(pivot_error)):
        for j in range(len(config_names)):
            text = ax.text(j, i, f'{pivot_error.iloc[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Per-Video Error Heatmap (All Methods)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    heatmap_path = output_path / 'method_comparison_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Error heatmap saved to: {heatmap_path}")

    # 3. Predicted vs Ground Truth (scatter plot for each method)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, config_name in enumerate(config_names):
        ax = axes[i]

        # Scatter plot
        ax.scatter(pivot_gt, pivot_detected[config_name], alpha=0.6, s=100)

        # Perfect prediction line
        min_val = min(pivot_gt.min(), pivot_detected[config_name].min())
        max_val = max(pivot_gt.max(), pivot_detected[config_name].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5)

        ax.set_xlabel('Ground Truth BPM')
        ax.set_ylabel('Detected BPM')
        ax.set_title(f'{config_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add R² score
        from scipy.stats import pearsonr
        if len(pivot_gt) > 1:
            r, _ = pearsonr(pivot_gt, pivot_detected[config_name])
            r_squared = r ** 2
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    scatter_path = output_path / 'method_comparison_scatter.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Scatter plots saved to: {scatter_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple breath counting methods/configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare different ACF window configurations (10s, 20s, 30s)
    python src/tuning/compare_methods.py \\
        --cache-dir cache/ \\
        --configs configs/acf_10s.yaml configs/acf_20s.yaml configs/acf_30s.yaml \\
        --output-dir diagnostics/comparison/

    # Compare only two configurations
    python src/tuning/compare_methods.py \\
        --cache-dir cache/ \\
        --configs configs/acf_10s.yaml configs/acf_20s.yaml \\
        --output-dir diagnostics/comparison/

Output Files:
    - method_comparison.csv: Detailed per-video comparison
    - method_summary.csv: Summary statistics per method
    - comparison_analysis.json: Structured analysis data
    - method_comparison_plots.png: Error distributions and agreement
    - method_comparison_heatmap.png: Per-video error heatmap
    - method_comparison_scatter.png: Predicted vs ground truth
        """
    )

    parser.add_argument('--cache-dir', type=str, required=True,
                       help='Directory with cached signals')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                       help='List of configuration files to compare')
    parser.add_argument('--output-dir', type=str, default='diagnostics/comparison',
                       help='Output directory for comparison results (default: diagnostics/comparison/)')

    args = parser.parse_args()

    if len(args.configs) < 2:
        print("Error: Please provide at least 2 configuration files to compare")
        return

    compare_methods(
        cache_dir=args.cache_dir,
        config_paths=args.configs,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
