#!/usr/bin/env python3
"""
Batch processing script for multiple videos - Process all videos in a directory
and save results to CSV
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import BreathingAnalyzer


def process_directory(directory, config_path, output_csv,
                     video_extensions=None, recursive=False):
    """
    Process all videos in a directory and save results to CSV

    Args:
        directory: Path to directory containing videos
        config_path: Path to config file
        output_csv: Path to output CSV file (or directory, will create results.csv inside)
        video_extensions: List of video file extensions to process
        recursive: Whether to search subdirectories recursively

    Returns:
        DataFrame with all results
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']

    # Handle case where output_csv is a directory
    output_csv_path = Path(output_csv)
    if output_csv_path.is_dir() or (not output_csv_path.suffix and not output_csv_path.exists()):
        # If it's a directory or has no extension, create a CSV file inside it
        output_csv_path = output_csv_path / 'results.csv'
        output_csv = str(output_csv_path)
        print(f"Note: Output path is a directory, saving to: {output_csv}")

    # Find all video files
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if recursive:
        video_files = []
        for ext in video_extensions:
            video_files.extend(directory.rglob(f'*{ext}'))
    else:
        video_files = []
        for ext in video_extensions:
            video_files.extend(directory.glob(f'*{ext}'))

    video_files = sorted(video_files)

    if not video_files:
        print(f"⚠ No video files found in {directory}")
        print(f"   Searched extensions: {', '.join(video_extensions)}")
        return None

    print("="*70)
    print("BATCH VIDEO PROCESSING")
    print("="*70)
    print(f"Directory: {directory}")
    print(f"Config: {config_path}")
    print(f"Found {len(video_files)} video(s)")
    print(f"Output CSV: {output_csv}")
    print("="*70)
    print()

    # Create output directories
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer once
    print("Initializing BreathingAnalyzer...")
    analyzer = BreathingAnalyzer(config_path=str(config_path))
    print("✓ Analyzer initialized\n")

    # Results storage
    results_list = []

    # Process each video
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 70)

        try:
            # Process video
            results = analyzer.process_video(
                video_path=str(video_path)
            )

            # Extract key metrics
            row = {
                'video_name': video_path.name,
                'video_path': str(video_path),
                'breathing_rate_bpm': results['breathing_rate_bpm'],
                'confidence': results['confidence'],
                'frequency_hz': results['frequency_hz'],
                'signal_length_frames': results['signal_length'],
                'video_fps': results['video_fps'],
                'duration_seconds': results['signal_length'] / results['video_fps'],
                'tracking_success_rate': np.mean(results['tracking_status']),
                'processing_status': 'success',
                'error_message': None
            }

            # Add validation metrics if available
            if 'validation' in results and results['validation']:
                val = results['validation']
                row['validation_consistent'] = val.get('is_consistent', None)
                row['validation_cv'] = val.get('cv', None)
                row['validation_mean_rate'] = val.get('mean_rate', None)

            # Add breath counts if available
            if 'breath_counts' in results and results['breath_counts']:
                for window, data in results['breath_counts'].items():
                    if isinstance(data, dict):
                        row[f'breath_count_{window}'] = data.get('count', None)
                        row[f'breath_rate_{window}_bpm'] = data.get('rate_bpm', None)
                    else:
                        row[f'breath_count_{window}'] = data

            # Add metadata statistics if available
            if 'metadata' in results:
                metadata = results['metadata']
                if 'motion' in metadata and len(metadata['motion']) > 0:
                    row['motion_mean'] = np.mean(metadata['motion'])
                    row['motion_std'] = np.std(metadata['motion'])
                    row['motion_max'] = np.max(metadata['motion'])

                if 'brightness' in metadata and len(metadata['brightness']) > 0:
                    row['brightness_mean'] = np.mean(metadata['brightness'])
                    row['brightness_std'] = np.std(metadata['brightness'])

            # Add quality metrics if available
            if 'quality' in results:
                quality = results['quality']
                for key, value in quality.items():
                    row[f'quality_{key}'] = value

            results_list.append(row)

            # Print summary
            print(f"✓ Breathing Rate: {results['breathing_rate_bpm']:.1f} BPM")
            print(f"✓ Confidence: {results['confidence']:.2f}")
            print(f"✓ Tracking Success: {row['tracking_success_rate']*100:.1f}%")

        except KeyboardInterrupt:
            print("\n\n⚠ Processing interrupted by user")
            # Save partial results
            if results_list:
                df = pd.DataFrame(results_list)
                partial_csv = output_csv.replace('.csv', '_partial.csv')
                df.to_csv(partial_csv, index=False)
                print(f"✓ Partial results saved to: {partial_csv}")
            raise

        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            traceback.print_exc()

            # Record error
            row = {
                'video_name': video_path.name,
                'video_path': str(video_path),
                'breathing_rate_bpm': None,
                'confidence': None,
                'processing_status': 'error',
                'error_message': str(e)
            }
            results_list.append(row)

    # Create DataFrame and save to CSV
    if results_list:
        df = pd.DataFrame(results_list)
        df['processed_at'] = datetime.now().isoformat()
        df.to_csv(output_csv, index=False)

        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"Total videos: {len(video_files)}")
        print(f"Successful: {sum(df['processing_status'] == 'success')}")
        print(f"Failed: {sum(df['processing_status'] == 'error')}")
        print(f"\nResults saved to: {output_csv}")

        # Print summary statistics
        successful = df[df['processing_status'] == 'success']
        if len(successful) > 0:
            print("\nSummary Statistics (successful videos):")
            print(f"  Mean breathing rate: {successful['breathing_rate_bpm'].mean():.1f} ± {successful['breathing_rate_bpm'].std():.1f} BPM")
            print(f"  Mean confidence: {successful['confidence'].mean():.2f}")
            print(f"  Mean tracking success: {successful['tracking_success_rate'].mean()*100:.1f}%")

        print("="*70)

        return df
    else:
        print("\n⚠ No results to save")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all videos in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in a directory (saves to output/results.csv by default)
  python batch_process_videos.py --directory data/videos

  # Process with custom config and output path
  python batch_process_videos.py --directory data/videos --config configs/default.yaml \\
      --output output/my_results.csv

  # Process recursively (include subdirectories)
  python batch_process_videos.py --directory data/ --recursive
        """
    )

    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--config', type=str,
                       default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--output', type=str, default='output/results.csv',
                       help='Path to output CSV file (default: output/results.csv)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories recursively')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV'],
                       help='Video file extensions to process (default: .mp4 .avi .mov .mkv)')

    args = parser.parse_args()

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        print("\nAvailable configs:")
        for cfg in Path('configs').glob('*.yaml'):
            print(f"  - {cfg}")
        return 1

    try:
        df = process_directory(
            directory=args.directory,
            config_path=str(config_path),
            output_csv=args.output,
            video_extensions=args.extensions,
            recursive=args.recursive
        )

        return 0 if df is not None else 1

    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
