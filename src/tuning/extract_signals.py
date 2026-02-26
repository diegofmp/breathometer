"""
Batch signal extraction tool

Extracts raw breathing signals from multiple videos using stored ROI coordinates.
This skips the slow detection/segmentation steps and only performs tracking and
measurement, making it much faster than the full pipeline.

Usage:
    python src/tuning/extract_signals.py \
        --roi-file rois/roi_manifest.json \
        --config configs/default.yaml \
        --output cache/v2/

The extracted signals are cached for fast parameter tuning experiments.
"""

import sys
from pathlib import Path
import cv2
import argparse
import numpy as np
from typing import Optional, Tuple, Dict
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tuning.signal_cache import ROIManager, SignalCache
from src.measurements import get_measurement


class SignalExtractor:
    """Extract breathing signals from videos using fixed ROI coordinates"""

    def __init__(self, config: Dict):
        """
        Initialize signal extractor

        Args:
            config: Configuration dictionary
        """
        # Set default measurement config if not provided
        default_measurement = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'patch_rows': 3,
            'patch_cols': 3
        }

        # Merge user config with defaults
        self.config = config.copy()
        if 'measurement' not in self.config:
            self.config['measurement'] = default_measurement
        else:
            # Fill in any missing measurement params with defaults
            for key, value in default_measurement.items():
                if key not in self.config['measurement']:
                    self.config['measurement'][key] = value

        self.measurement_method = 'optical_flow_divergence'  # Currently the only method
        self.tracker_type = config.get('tracking', {}).get('chest_tracker', 'KCF')
        self.start_frame = config.get('tracking', {}).get('start_frame', 0)
        self.max_frames = config.get('tracking', {}).get('max_frames', None)

    def _create_tracker(self) -> cv2.Tracker:
        """Create OpenCV tracker"""
        tracker_types = {
            'KCF': cv2.TrackerKCF_create,
            'CSRT': cv2.TrackerCSRT_create,
            'MIL': cv2.TrackerMIL_create,
            'MedianFlow': cv2.legacy.TrackerMedianFlow_create,
        }

        if self.tracker_type not in tracker_types:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")

        return tracker_types[self.tracker_type]()

    def extract_signal(
        self,
        video_path: str,
        roi: Tuple[float, float, float, float],
        fps: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Extract raw breathing signal from video using fixed ROI

        Args:
            video_path: Path to video file
            roi: (x, y, w, h) chest ROI coordinates
            fps: Video frame rate (if None, auto-detect)

        Returns:
            Dictionary with extraction results or None if failed
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ Error: Could not open video: {video_path}")
            return None

        # Get video properties
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range
        start_frame = self.start_frame
        if self.max_frames:
            end_frame = min(start_frame + self.max_frames, total_frames)
        else:
            end_frame = total_frames

        frames_to_process = end_frame - start_frame

        print(f"Video: {fps:.1f} fps, {total_frames} frames")
        print(f"Processing: frames {start_frame}-{end_frame} ({frames_to_process} frames, {frames_to_process/fps:.1f}s)")
        print(f"ROI: ({roi[0]:.0f}, {roi[1]:.0f}, {roi[2]:.0f}, {roi[3]:.0f})")

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read first frame and initialize tracker
        ret, frame = cap.read()
        if not ret:
            print(f"✗ Error: Could not read frame {start_frame}")
            cap.release()
            return None

        # Initialize tracker with ROI
        # Convert ROI to integers (OpenCV requires int tuple)
        roi_int = tuple(int(v) for v in roi)

        # Validate ROI dimensions
        x, y, w, h = roi_int
        if w <= 0 or h <= 0:
            print(f"✗ Error: Invalid ROI dimensions ({w}x{h})")
            cap.release()
            return None

        # Store original ROI dimensions (to preserve size during tracking)
        original_w, original_h = w, h

        tracker = self._create_tracker()
        tracker.init(frame, roi_int)

        # Create measurement object
        measurement = get_measurement(self.config['measurement'])

        # Storage for signal and ROI history
        raw_signal = []
        roi_history = []
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process frames
        frame_count = 0
        tracking_failures = 0

        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Update tracker
            success, tracked_roi_raw = tracker.update(frame)

            if not success:
                tracking_failures += 1
                # Use last known ROI
                if roi_history:
                    tracked_roi = roi_history[-1]
                else:
                    tracked_roi = roi_int

                # Validate ROI before re-initializing tracker
                x, y, w, h = [int(v) for v in tracked_roi]
                if w <= 0 or h <= 0:
                    print(f"✗ Warning: Invalid ROI dimensions ({w}x{h}) at frame {start_frame + frame_count}")
                    raw_signal.append(np.nan)
                    roi_history.append(tracked_roi)
                    prev_frame = gray_frame
                    frame_count += 1
                    continue

                # Re-initialize tracker with the last known ROI
                tracker = self._create_tracker()
                # Ensure ROI is int tuple for OpenCV
                tracked_roi_int = tuple(int(v) for v in tracked_roi)
                tracker.init(frame, tracked_roi_int)
            else:
                # Success: Extract center from tracker output and preserve original dimensions
                tx, ty, tw, th = tracked_roi_raw
                # Calculate center of tracked ROI
                center_x = tx + tw / 2
                center_y = ty + th / 2
                # Build new ROI with original dimensions centered at tracked position
                new_x = center_x - original_w / 2
                new_y = center_y - original_h / 2
                tracked_roi = (new_x, new_y, original_w, original_h)

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract chest region
            x, y, w, h = [int(v) for v in tracked_roi]
            chest_roi = gray_frame[y:y+h, x:x+w]
            prev_chest_roi = prev_frame[y:y+h, x:x+w]

            if chest_roi.size == 0:
                print(f"✗ Warning: Empty ROI at frame {start_frame + frame_count}")
                raw_signal.append(np.nan)
                roi_history.append(tracked_roi)
                prev_frame = gray_frame
                frame_count += 1
                continue

            # Measure breathing signal
            signal_value = measurement.measure(chest_roi)

            # Store results
            raw_signal.append(signal_value)
            roi_history.append(tracked_roi)

            # Update previous frame
            prev_frame = gray_frame

            # Progress indicator
            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / frames_to_process) * 100
                print(f"  Progress: {frame_count}/{frames_to_process} frames ({progress:.1f}%)")

        cap.release()


        # Validate extraction
        if not raw_signal:
            print(f"✗ Error: No signal extracted")
            return None
        
        

        # Convert to numpy array
        raw_signal = np.array(raw_signal)

        # Count NaN and zero values BEFORE interpolation
        nan_count_original = int(np.sum(np.isnan(raw_signal)))
        zero_count_original = int(np.sum(raw_signal == 0.0))

        # Handle NaN values (fill with interpolation)
        if np.any(np.isnan(raw_signal)):
            nan_count = np.sum(np.isnan(raw_signal))
            print(f"  Warning: {nan_count} NaN values in signal ({nan_count/len(raw_signal)*100:.1f}%)")

            # Simple linear interpolation
            mask = ~np.isnan(raw_signal)
            if np.any(mask):
                indices = np.arange(len(raw_signal))
                raw_signal = np.interp(indices, indices[mask], raw_signal[mask])
            else:
                print(f"✗ Error: All signal values are NaN")
                return None

        # Summary
        print(f"✓ Extraction complete:")
        print(f"  Signal length: {len(raw_signal)} frames ({len(raw_signal)/fps:.1f}s)")
        print(f"  Signal range: [{np.min(raw_signal):.3f}, {np.max(raw_signal):.3f}]")
        print(f"  Tracking failures: {tracking_failures}")
        if nan_count_original > 0:
            print(f"  NaN values: {nan_count_original} ({nan_count_original/len(raw_signal)*100:.1f}%)")
        if zero_count_original > 0:
            print(f"  Zero values: {zero_count_original} ({zero_count_original/len(raw_signal)*100:.1f}%)")

        return {
            'raw_signal': raw_signal,
            'roi': roi,
            'fps': fps,
            'metadata': {
                'measurement_method': self.measurement_method,
                'tracker_type': self.tracker_type,
                'start_frame': start_frame,
                'end_frame': start_frame + frame_count,
                'frames_processed': frame_count,
                'tracking_failures': tracking_failures,
                'signal_mean': float(np.mean(raw_signal)),
                'signal_std': float(np.std(raw_signal)),
                'nan_count': nan_count_original,
                'zero_count': zero_count_original,
            }
        }


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_signals(
    roi_file: str,
    config_path: str,
    cache_dir: str,
    resume: bool = True,
    limit: Optional[int] = None
):
    """
    Batch extract signals from videos using stored ROIs

    Args:
        roi_file: Path to ROI manifest file
        config_path: Path to configuration file
        cache_dir: Directory to save cached signals
        resume: Skip videos that already have cached signals
        limit: Maximum number of videos to process (default: None = all)
    """
    # Load configuration
    config = load_config(config_path)

    # Load ROI manifest
    roi_manager = ROIManager(Path(roi_file).parent)
    rois = roi_manager.load_all_rois()

    if not rois:
        print("Error: No ROIs found in manifest")
        return

    # Apply limit if specified
    if limit is not None and limit > 0:
        # Convert dict to list of items, take first N, convert back to dict
        rois = dict(list(rois.items())[:limit])
        print(f"⚠ Limit applied: Processing only first {limit} videos")

    print(f"\n{'='*70}")
    print(f"SIGNAL EXTRACTION")
    print(f"{'='*70}")
    print(f"Videos to process: {len(rois)}")
    print(f"Config: {config_path}")
    print(f"Measurement: optical_flow_divergence")
    print(f"Output: {cache_dir}")
    print(f"{'='*70}\n")

    # Initialize signal cache
    signal_cache = SignalCache(cache_dir)

    if resume:
        print(f"Resume mode: Skipping videos with existing signals")
        print(f"Current cached signals: {signal_cache.count()}\n")

    # Initialize extractor
    extractor = SignalExtractor(config)

    # Process each video
    success_count = 0
    skip_count = 0
    error_count = 0

    for video_idx, (video_path, roi_data) in enumerate(rois.items(), 1):
        print(f"\n{'='*70}")
        print(f"SIGNAL EXTRACTION - Video {video_idx}/{len(rois)}")
        print(f"{'='*70}")
        print(f"Video: {Path(video_path).name}")

        ground_truth = roi_data.get('ground_truth_bpm')
        if ground_truth:
            print(f"Ground Truth: {ground_truth:.1f} BPM")

        # Check if signal already cached
        if resume and signal_cache.signal_exists(video_path):
            cached_signal = signal_cache.load_signal(video_path)
            print(f"✓ Signal already cached: {len(cached_signal['raw_signal'])} frames")
            print(f"  Skipping... (use --no-resume to re-extract)")
            skip_count += 1
            continue

        # Extract ROI
        roi = tuple(roi_data['roi'])

        # Extract signal
        result = extractor.extract_signal(video_path, roi)

        if result is None:
            print(f"✗ Extraction failed")
            error_count += 1
            continue

        # Save to cache
        signal_cache.save_signal(
            video_path=video_path,
            raw_signal=result['raw_signal'],
            roi=result['roi'],
            fps=result['fps'],
            ground_truth_bpm=ground_truth,
            metadata=result['metadata']
        )

        print(f"✓ Signal cached")
        success_count += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SIGNAL EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully extracted: {success_count}")
    print(f"Skipped (already cached): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total cached signals: {signal_cache.count()}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract breathing signals from videos using stored ROIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract signals using stored ROIs
    python src/tuning/extract_signals.py --roi-file rois/roi_manifest.json

    # Use custom config
    python src/tuning/extract_signals.py --roi-file rois/roi_manifest.json --config configs/custom.yaml

    # Force re-extract all signals
    python src/tuning/extract_signals.py --roi-file rois/roi_manifest.json --no-resume

    # Process only first 5 videos (for quick testing)
    python src/tuning/extract_signals.py --roi-file rois/roi_manifest.json --limit 5
        """
    )

    parser.add_argument('--roi-file', type=str, required=True,
                       help='Path to ROI manifest file (from collect_rois.py)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file (default: configs/default.yaml)')
    parser.add_argument('--output', type=str, default='cache',
                       help='Output directory for cached signals (default: cache/)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Re-extract all signals (ignore existing cache)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos to process (for quick testing)')

    args = parser.parse_args()

    extract_signals(
        roi_file=args.roi_file,
        config_path=args.config,
        cache_dir=args.output,
        resume=not args.no_resume,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
