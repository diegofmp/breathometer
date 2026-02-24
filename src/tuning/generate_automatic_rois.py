"""
Automatic ROI Generation Tool

Generates automatic bird chest ROI predictions for all videos in an existing ROI manifest.
Uses the pipeline's _locate_bird_roi() method to detect ROIs automatically instead of
relying on manual annotations.

This allows you to:
1. Compare automatic detection vs manual ROIs
2. Test pipeline performance with automatic detection
3. Generate ROIs for new videos without manual annotation

Usage:
    python src/tuning/generate_automatic_rois.py \
        --input-manifest rois/roi_manifest.json \
        --output-manifest rois/roi_manifest_auto.json \
        --config configs/default.yaml
"""

import sys
from pathlib import Path
import cv2
import argparse
import yaml
from typing import Dict, Optional
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tuning.signal_cache import ROIManager
from src.pipeline import BreathingAnalyzer


def generate_automatic_roi(
    video_path: str,
    config: Dict,
    frame_number: int = 300
) -> Optional[Dict]:
    """
    Generate automatic ROI for a video using the pipeline's detection method

    Args:
        video_path: Path to video file
        config: Pipeline configuration dictionary
        frame_number: Frame to start buffering from (default: 300)

    Returns:
        Dictionary with ROI and metadata, or None if detection failed
    """
    print(f"\nProcessing: {Path(video_path).name}")

    # Check if video exists
    if not Path(video_path).exists():
        print(f"✗ Error: Video not found: {video_path}")
        return None

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Error: Could not open video: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {fps:.1f} fps, {total_frames} frames")

    # Initialize pipeline (this loads all detectors)
    try:
        # Create temporary config file (BreathingAnalyzer requires a config path)
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_config_path = tmp.name

        # Initialize analyzer with temp config
        analyzer = BreathingAnalyzer(config_path=tmp_config_path)

        # Clean up temp file
        import os
        os.remove(tmp_config_path)

        print(f"  Pipeline initialized (buffer size: {analyzer.buffer_frames_size})")

    except Exception as e:
        print(f"✗ Error: Could not initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        cap.release()
        return None

    # Seek to frame where we'll start buffering
    buffer_start = max(0, frame_number - analyzer.buffer_frames_size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)

    print(f"  Buffering frames {buffer_start} to {frame_number}...")

    # Fill buffer with frames
    from collections import deque
    analyzer.buffer_frames = deque(maxlen=analyzer.buffer_frames_size)

    for i in range(analyzer.buffer_frames_size):
        ret, frame = cap.read()
        if not ret:
            print(f"✗ Error: Could not read frame {buffer_start + i}")
            cap.release()
            return None
        analyzer.buffer_frames.append(frame)

    cap.release()

    print(f"  Buffer filled with {len(analyzer.buffer_frames)} frames")
    print(f"  Running automatic ROI detection...")

    # Run automatic ROI detection
    try:
        roi = analyzer._locate_bird_roi()
        x, y, w, h = roi

        print(f"  ✓ ROI detected: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})")

        return {
            'roi': [int(v) for v in roi],
            'frame_number': frame_number,
            'metadata': {
                'fps': fps,
                'total_frames': total_frames,
                'detection_method': 'automatic',
                'config': {
                    'detection': config['detection']['mode'],
                    'segmentation': config['segmentation']['mode'],
                    'localization': config['localization']['method'],
                    'hard_constrain_mode': config['localization']['custom_localizer']['hard_constrain_mode']
                }
            }
        }

    except Exception as e:
        print(f"✗ Error: ROI detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_automatic_rois(
    input_manifest_path: str,
    output_manifest_path: str,
    config_path: str,
    resume: bool = True,
    limit: Optional[int] = None
):
    """
    Generate automatic ROIs for all videos in an existing manifest

    Args:
        input_manifest_path: Path to input ROI manifest (with manual ROIs and ground truth)
        output_manifest_path: Path to output manifest for automatic ROIs
        config_path: Path to pipeline configuration file
        resume: Skip videos that already have automatic ROIs
        limit: Maximum number of videos to process (for testing)
    """
    print(f"\n{'='*70}")
    print(f"AUTOMATIC ROI GENERATION")
    print(f"{'='*70}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config: {config_path}")
    print(f"Detection: {config['detection']['mode']}")
    print(f"Segmentation: {config['segmentation']['mode']}")
    print(f"Localization: {config['localization']['method']}")

    # Load input manifest (manual ROIs with ground truth)
    input_roi_manager = ROIManager(Path(input_manifest_path).parent)
    input_rois = input_roi_manager.load_all_rois()

    if not input_rois:
        print(f"✗ Error: No ROIs found in input manifest: {input_manifest_path}")
        return

    print(f"Input manifest: {input_manifest_path}")
    print(f"  Videos with manual ROIs: {len(input_rois)}")

    # Load or create output manifest
    output_dir = Path(output_manifest_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if Path(output_manifest_path).exists() and resume:
        with open(output_manifest_path, 'r') as f:
            import json
            output_manifest = json.load(f)
        print(f"Output manifest: {output_manifest_path} (existing)")
        print(f"  Videos with automatic ROIs: {len(output_manifest)}")
    else:
        output_manifest = {}
        print(f"Output manifest: {output_manifest_path} (new)")

    # Apply limit if specified
    video_list = list(input_rois.items())
    if limit is not None and limit > 0:
        video_list = video_list[:limit]
        print(f"⚠ Limit applied: Processing only first {limit} videos")

    print(f"{'='*70}\n")

    # Process each video
    success_count = 0
    skip_count = 0
    error_count = 0

    for video_idx, (video_path, roi_data) in enumerate(video_list, 1):
        print(f"\n{'='*70}")
        print(f"Video {video_idx}/{len(video_list)}")
        print(f"{'='*70}")

        # Check if already processed (resume mode)
        if resume and video_path in output_manifest:
            print(f"✓ Already processed: {Path(video_path).name}")
            print(f"  Existing ROI: {output_manifest[video_path]['roi']}")
            skip_count += 1
            continue

        # Get ground truth from input manifest
        ground_truth_bpm = roi_data.get('ground_truth_bpm')
        frame_number = roi_data.get('frame_number', 300)

        if ground_truth_bpm:
            print(f"Ground Truth: {ground_truth_bpm:.1f} BPM")

        # Generate automatic ROI
        result = generate_automatic_roi(
            video_path=video_path,
            config=config,
            frame_number=frame_number
        )

        if result is None:
            print(f"✗ Failed to generate automatic ROI")
            error_count += 1
            continue

        # Add ground truth and timestamp to result
        result['ground_truth_bpm'] = ground_truth_bpm
        result['timestamp'] = datetime.now().isoformat()
        result['roi_source'] = 'automatic'

        # Compare with manual ROI if available
        manual_roi = roi_data.get('roi')
        if manual_roi:
            auto_roi = result['roi']
            print(f"  Manual ROI:    ({manual_roi[0]:.0f}, {manual_roi[1]:.0f}, {manual_roi[2]:.0f}, {manual_roi[3]:.0f})")
            print(f"  Automatic ROI: ({auto_roi[0]:.0f}, {auto_roi[1]:.0f}, {auto_roi[2]:.0f}, {auto_roi[3]:.0f})")

            # Calculate IoU (Intersection over Union)
            mx, my, mw, mh = manual_roi
            ax, ay, aw, ah = auto_roi

            # Calculate intersection
            x1 = max(mx, ax)
            y1 = max(my, ay)
            x2 = min(mx + mw, ax + aw)
            y2 = min(my + mh, ay + ah)

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                union = (mw * mh) + (aw * ah) - intersection
                iou = intersection / union if union > 0 else 0
                print(f"  IoU with manual ROI: {iou:.3f}")
                result['iou_with_manual'] = float(iou)

        # Save to output manifest
        output_manifest[video_path] = result

        # Save manifest after each successful detection (in case of crashes)
        with open(output_manifest_path, 'w') as f:
            import json
            json.dump(output_manifest, f, indent=2)

        print(f"✓ Automatic ROI saved to manifest")
        success_count += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"AUTOMATIC ROI GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully generated: {success_count}")
    print(f"Skipped (already processed): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total automatic ROIs: {len(output_manifest)}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate automatic ROIs for videos using pipeline detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate automatic ROIs for all videos in manifest
    python src/tuning/generate_automatic_rois.py \\
        --input-manifest rois/roi_manifest.json \\
        --output-manifest rois/roi_manifest_auto.json \\
        --config configs/default.yaml

    # Test on first 5 videos
    python src/tuning/generate_automatic_rois.py \\
        --input-manifest rois/roi_manifest.json \\
        --output-manifest rois/roi_manifest_auto.json \\
        --config configs/default.yaml \\
        --limit 5

    # Force re-generate all ROIs (ignore existing)
    python src/tuning/generate_automatic_rois.py \\
        --input-manifest rois/roi_manifest.json \\
        --output-manifest rois/roi_manifest_auto.json \\
        --config configs/default.yaml \\
        --no-resume
        """
    )

    parser.add_argument('--input-manifest', type=str, required=True,
                       help='Path to input ROI manifest (with manual ROIs and ground truth)')
    parser.add_argument('--output-manifest', type=str, required=True,
                       help='Path to output manifest for automatic ROIs')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to pipeline configuration file (default: configs/default.yaml)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Re-generate all ROIs (ignore existing)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos to process (for testing)')

    args = parser.parse_args()

    generate_automatic_rois(
        input_manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        config_path=args.config,
        resume=not args.no_resume,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
