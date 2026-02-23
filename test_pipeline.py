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
    """Plot breathing signal and analysis results"""

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # 1. Breathing signal
    signal = np.array(results['breathing_signal'])
    fps = results['video_fps']
    time = np.arange(len(signal)) / fps

    axes[0].plot(time, signal, 'b-', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Breathing Signal')
    axes[0].set_title(f'Breathing Signal - Estimated Rate: {results["breathing_rate_bpm"]:.1f} BPM')
    axes[0].grid(True, alpha=0.3)

    # 2. Tracking status
    tracking = np.array(results['tracking_status'])
    axes[1].plot(time, tracking, 'g-', linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Tracking Success')
    axes[1].set_title(f'Tracking Status (Success Rate: {np.mean(tracking)*100:.1f}%)')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)

    # 3. Metadata (motion, brightness)
    metadata = results['metadata']
    if len(metadata['motion']) > 0:
        ax3a = axes[2]
        ax3a.plot(time, metadata['motion'], 'r-', label='Motion', alpha=0.7)
        ax3a.set_xlabel('Time (s)')
        ax3a.set_ylabel('Motion', color='r')
        ax3a.tick_params(axis='y', labelcolor='r')
        ax3a.grid(True, alpha=0.3)

        ax3b = ax3a.twinx()
        brightness = np.array(metadata['brightness'])
        ax3b.plot(time, brightness, 'b-', label='Brightness', alpha=0.7)
        ax3b.set_ylabel('Brightness', color='b')
        ax3b.tick_params(axis='y', labelcolor='b')

        axes[2].set_title('Motion and Brightness')

    plt.tight_layout()
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
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Path to output video file (optional)')
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
    if args.output:
        print(f"Output: {args.output}")
    print("="*60)
    print()

    try:
        # Initialize analyzer
        print("Initializing BreathingAnalyzer...")
        analyzer = BreathingAnalyzer(config_path=str(args.config))

        # Process video
        print("\nProcessing video...")
        results = analyzer.process_video(
            video_path=str(args.video),
            output_path=args.output
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

        # Breath counts
        if 'breath_counts' in results and results['breath_counts']:
            print("\nBreath Counts:")
            for window, data in results['breath_counts'].items():
                if isinstance(data, dict):
                    print(f"  {window}: {data.get('count', 0)} breaths → {data.get('rate_bpm', 0):.1f} BPM")
                else:
                    print(f"  {window}: {data} breaths")

        print("="*60)

        # Plot if requested
        if args.plot:
            print("\nGenerating plots...")
            # Extract video name from path
            video_name = Path(args.video).stem  # Gets filename without extension

            # Determine output path for plot
            if args.output:
                output_path = Path(args.output)
                # Check if it's a directory or file path
                if output_path.is_dir() or str(args.output).endswith('/'):
                    output_dir = output_path
                else:
                    output_dir = output_path.parent

                # Create directory if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_path = output_dir / f"{video_name}_pipeline_results.png"
            else:
                plot_path = Path(f"{video_name}_pipeline_results.png")

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
