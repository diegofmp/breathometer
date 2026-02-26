#!/usr/bin/env python3
"""
Test script for pipeline - Process a video with the BreathingAnalyzer
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import BreathingAnalyzer


def plot_results(results, output_path=None):
    """Plot breathing signal and analysis results (matches Streamlit UI)"""
    from matplotlib.gridspec import GridSpec

    # Signal data
    signal = np.array(results['breathing_signal'])
    fps = results['video_fps']
    time_signal = np.arange(len(signal)) / fps
    signal_duration = len(signal) / fps

    # Check if window estimates are available
    window_estimates = results.get('window_estimates', [])
    has_windows = len(window_estimates) > 0

    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 10), dpi=100)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[30, 1], wspace=0.05, hspace=0.35)

    # Create axes - main plots use first column, colorbar space in second column
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0])
    ]
    cbar_ax = fig.add_subplot(gs[1, 1])  # Colorbar axis for middle plot

    # 1. Breathing signal with window boundaries
    axes[0].plot(time_signal, signal, linewidth=0.8, alpha=0.7, color='steelblue', label='Raw signal')

    if has_windows:
        # Mark window boundaries
        for i, w in enumerate(window_estimates):
            color = 'red' if i == 0 else 'gray'
            alpha = 0.5 if i == 0 else 0.2
            axes[0].axvline(w['start_time'], color=color, linestyle='--', alpha=alpha, linewidth=1)
            if i == 0:
                axes[0].axvline(w['end_time'], color=color, linestyle='--', alpha=alpha, linewidth=1, label='Window boundaries')

        overlap = results.get('acf_overlap', 0)
        axes[0].set_title(f'Breathing Signal with Window Boundaries (overlap={overlap*100:.0f}%)', fontweight='bold')
    else:
        axes[0].set_title(f'Breathing Signal - Estimated Rate: {results["breathing_rate_bpm"]:.1f} BPM', fontweight='bold')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, signal_duration])

    # 2. BPM estimates per window (if available) or tracking status
    if has_windows:
        window_times = [(w['start_time'] + w['end_time'])/2 for w in window_estimates]
        window_bpms = [w['bpm'] for w in window_estimates]
        window_confidences = [w['confidence'] for w in window_estimates]

        scatter = axes[1].scatter(window_times, window_bpms, s=150, c=window_confidences,
                                  cmap='RdYlGn', vmin=0, vmax=1, edgecolors='black',
                                  linewidth=1.5, zorder=5)
        axes[1].plot(window_times, window_bpms, '--', linewidth=1, alpha=0.4, color='gray')

        # Add colorbar
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Confidence', rotation=270, labelpad=20)

        # Final estimate line
        final_bpm = results['breathing_rate_bpm']
        axes[1].axhline(final_bpm, color='blue', linestyle='-', linewidth=2.5,
                        label=f"Final: {final_bpm:.1f} BPM", zorder=3)

        # Mean and std bands
        mean_bpm = np.mean(window_bpms)
        std_bpm = np.std(window_bpms)
        axes[1].axhline(mean_bpm, color='orange', linestyle=':', linewidth=1.5,
                        alpha=0.7, label=f"Mean: {mean_bpm:.1f} BPM")
        axes[1].fill_between([0, signal_duration], mean_bpm - std_bpm, mean_bpm + std_bpm,
                              alpha=0.2, color='orange', label=f'±1 std ({std_bpm:.1f})')

        axes[1].set_title('BPM Estimates per Window (Color = Confidence)', fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('BPM')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, signal_duration])
    else:
        # Fallback to tracking status
        tracking = np.array(results['tracking_status'])
        time_tracking = np.arange(len(tracking)) / fps
        axes[1].plot(time_tracking, tracking, color='green', linewidth=1)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Tracking Success')
        success_rate = np.mean(tracking) * 100
        axes[1].set_title(f'Tracking Status (Success Rate: {success_rate:.1f}%)', fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].grid(True, alpha=0.3)
        cbar_ax.axis('off')

    # 3. Confidence per window (if available) or motion/brightness
    if has_windows:
        window_times = [(w['start_time'] + w['end_time'])/2 for w in window_estimates]
        window_confidences = [w['confidence'] for w in window_estimates]

        # Get window width for bar chart
        acf_window_size = results.get('acf_window_size', 10)
        acf_overlap = results.get('acf_overlap', 0.5)
        window_width = acf_window_size * (1 - acf_overlap)

        bars = axes[2].bar(window_times, window_confidences, width=window_width * 0.9,
                           alpha=0.7, color='forestgreen', edgecolor='black')

        # Color bars by confidence
        for bar, conf in zip(bars, window_confidences):
            if conf < 0.3:
                bar.set_color('red')
            elif conf < 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('forestgreen')

        mean_conf = np.mean(window_confidences)
        axes[2].axhline(mean_conf, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_conf:.3f}')

        acf_min_confidence = results.get('acf_min_confidence', 0)
        if acf_min_confidence > 0:
            axes[2].axhline(acf_min_confidence, color='purple', linestyle=':', linewidth=2,
                            alpha=0.7, label=f'Min threshold: {acf_min_confidence}')

        axes[2].set_title('Confidence per Window', fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Confidence')
        axes[2].set_ylim([0, 1.05])
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_xlim([0, signal_duration])
    else:
        # Fallback to motion/brightness
        metadata = results['metadata']
        if len(metadata['motion']) > 0:
            motion = np.array(metadata['motion'])
            brightness = np.array(metadata['brightness'])
            time_metadata = np.arange(len(motion)) / fps

            ax3a = axes[2]
            ax3a.plot(time_metadata, motion, '-', label='Motion', alpha=0.7, color='red')
            ax3a.set_xlabel('Time (s)')
            ax3a.set_ylabel('Motion', color='red')
            ax3a.tick_params(axis='y', labelcolor='red')
            ax3a.grid(True, alpha=0.3)

            ax3b = ax3a.twinx()
            ax3b.plot(time_metadata, brightness, '-', label='Brightness', alpha=0.7, color='blue')
            ax3b.set_ylabel('Brightness', color='blue')
            ax3b.tick_params(axis='y', labelcolor='blue')

            axes[2].set_title('Motion and Brightness', fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results plot saved to: {output_path}")
    #plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test pipeline process_video method')
    parser.add_argument('--video', type=str,
                       default='data/videos/H5_F_2.mp4',
                       help='Path to input video file')
    parser.add_argument('--config', type=str,
                       default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--plot', action='store_true',
                       help='Plot results after processing')

    args = parser.parse_args()

    # Check if video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {args.video}")
        print("\nAvailable videos:")
        for vid in Path('.').rglob('*.mp4'):
            print(f"  - {vid}")
        return 1

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        print("\nAvailable configs:")
        for cfg in Path('configs').glob('*.yaml'):
            print(f"  - {cfg}")
        return 1

    print("="*60)
    print("BREATHOMETER PIPELINE TEST")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Config: {args.config}")
    print("="*60)
    print()

    try:
        # Initialize analyzer
        print("Initializing BreathingAnalyzer...")
        analyzer = BreathingAnalyzer(config_path=str(args.config))

        # Process video
        print("\nProcessing video...")
        results = analyzer.process_video(
            video_path=str(args.video)
        )

        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Breathing Rate: {results['breathing_rate_bpm']:.1f} BPM")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"Frequency: {results['frequency_hz']:.3f} Hz")
        print(f"Signal Length: {results['signal_length']} frames")
        print(f"Duration: {results['signal_length']/results['video_fps']:.1f} seconds")
        print(f"Tracking Success Rate: {np.mean(results['tracking_status'])*100:.1f}%")

        # Validation info
        if 'validation' in results and results['validation']:
            val = results['validation']
            if 'is_consistent' in val:
                status = "✓ Consistent" if val['is_consistent'] else "⚠ Inconsistent"
                print(f"\nValidation: {status}")
                if 'cv' in val:
                    print(f"Coefficient of Variation: {val['cv']:.2%}")
                if 'mean_rate' in val:
                    print(f"Mean Rate: {val['mean_rate']:.1f} BPM")

        print("="*60)

        # Plot if requested
        if args.plot:
            print("\nGenerating plots...")
            # Extract video name from path
            video_name = Path(args.video).stem  # Gets filename without extension

            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)

            plot_path = output_dir / f"{video_name}_pipeline_results.png"
            plot_results(results, str(plot_path))

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
