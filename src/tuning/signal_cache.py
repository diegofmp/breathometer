"""
Signal caching utilities for parameter optimization

Manages storage and retrieval of:
1. ROI coordinates (from manual selection)
2. Raw breathing signals (extracted from videos)
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import numpy as np


class ROIManager:
    """
    Manages storage and retrieval of ROI coordinates

    ROIs are collected once per video and reused for all signal extraction
    and parameter tuning experiments.
    """

    def __init__(self, roi_dir: str = 'rois'):
        """
        Initialize ROI manager

        Args:
            roi_dir: Directory to store ROI manifest
        """
        self.roi_dir = Path(roi_dir)
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.roi_dir / 'roi_manifest.json'
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load existing ROI manifest or create new one"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Save ROI manifest to disk"""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def save_roi(self,
                 video_path: str,
                 roi: Tuple[float, float, float, float],
                 ground_truth_bpm: Optional[float] = None,
                 frame_number: int = 300,
                 metadata: Optional[Dict] = None):
        """
        Save ROI coordinates for a video

        Args:
            video_path: Path to video file
            roi: (x, y, w, h) ROI coordinates
            ground_truth_bpm: Optional ground truth BPM for this video
            frame_number: Frame where ROI was selected
            metadata: Optional additional metadata
        """
        video_path = str(Path(video_path).resolve())

        roi_data = {
            'roi': list(roi),  # Convert tuple to list for JSON
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
        }

        if ground_truth_bpm is not None:
            roi_data['ground_truth_bpm'] = float(ground_truth_bpm)

        if metadata:
            roi_data['metadata'] = metadata

        self.manifest[video_path] = roi_data
        self._save_manifest()

    def load_roi(self, video_path: str) -> Optional[Dict]:
        """
        Load ROI data for a video

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with ROI data or None if not found
        """
        video_path = str(Path(video_path).resolve())
        return self.manifest.get(video_path)

    def load_all_rois(self) -> Dict:
        """
        Load all ROIs

        Returns:
            Dictionary mapping video paths to ROI data
        """
        return self.manifest.copy()

    def roi_exists(self, video_path: str) -> bool:
        """
        Check if ROI exists for a video

        Args:
            video_path: Path to video file

        Returns:
            True if ROI exists, False otherwise
        """
        video_path = str(Path(video_path).resolve())
        return video_path in self.manifest

    def get_videos_with_rois(self) -> List[str]:
        """
        Get list of all video paths with stored ROIs

        Returns:
            List of video paths
        """
        return list(self.manifest.keys())

    def count(self) -> int:
        """
        Get number of stored ROIs

        Returns:
            Number of ROIs in manifest
        """
        return len(self.manifest)


class SignalCache:
    """
    Manages storage and retrieval of extracted breathing signals

    Signals are extracted once per video and cached for fast parameter tuning.
    """

    def __init__(self, cache_dir: str = 'cache'):
        """
        Initialize signal cache

        Args:
            cache_dir: Directory to store cached signals
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / 'signal_manifest.json'
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load existing signal manifest or create new one"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Save signal manifest to disk"""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _get_cache_filename(self, video_path: str) -> Path:
        """
        Get cache filename for a video

        Args:
            video_path: Path to video file

        Returns:
            Path to cache file
        """
        video_name = Path(video_path).stem
        return self.cache_dir / f'{video_name}_signal.pkl'

    def save_signal(self,
                   video_path: str,
                   raw_signal: np.ndarray,
                   roi: Tuple[float, float, float, float],
                   fps: float,
                   ground_truth_bpm: Optional[float] = None,
                   metadata: Optional[Dict] = None):
        """
        Save extracted signal to cache

        Args:
            video_path: Path to video file
            raw_signal: Raw breathing signal (unfiltered)
            roi: (x, y, w, h) ROI coordinates used for extraction
            fps: Video frame rate
            ground_truth_bpm: Optional ground truth BPM
            metadata: Optional additional metadata (measurement method, etc.)
        """
        video_path = str(Path(video_path).resolve())
        cache_file = self._get_cache_filename(video_path)

        # Prepare cache data
        cache_data = {
            'video_path': video_path,
            'raw_signal': raw_signal,
            'roi': roi,
            'fps': fps,
            'timestamp': datetime.now().isoformat(),
            'signal_length': len(raw_signal),
            'duration_s': len(raw_signal) / fps,
        }

        if ground_truth_bpm is not None:
            cache_data['ground_truth_bpm'] = float(ground_truth_bpm)

        if metadata:
            cache_data['metadata'] = metadata

        # Save signal to pickle file
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update manifest
        manifest_entry = {
            'cache_file': str(cache_file),
            'timestamp': cache_data['timestamp'],
            'signal_length': cache_data['signal_length'],
            'duration_s': cache_data['duration_s'],
            'fps': fps,
        }

        if ground_truth_bpm is not None:
            manifest_entry['ground_truth_bpm'] = float(ground_truth_bpm)

        # Add signal quality metrics from metadata if available
        if metadata:
            if 'nan_count' in metadata:
                manifest_entry['nan_count'] = metadata['nan_count']
            if 'zero_count' in metadata:
                manifest_entry['zero_count'] = metadata['zero_count']
            if 'tracking_failures' in metadata:
                manifest_entry['tracking_failures'] = metadata['tracking_failures']

        self.manifest[video_path] = manifest_entry
        self._save_manifest()

    def load_signal(self, video_path: str) -> Optional[Dict]:
        """
        Load cached signal for a video

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with signal data or None if not found
        """
        video_path = str(Path(video_path).resolve())

        if video_path not in self.manifest:
            return None

        cache_file = self.manifest[video_path]['cache_file']

        if not Path(cache_file).exists():
            print(f"Warning: Cache file {cache_file} not found in manifest but missing from disk")
            return None

        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def load_all_signals(self) -> List[Dict]:
        """
        Load all cached signals

        Returns:
            List of signal data dictionaries
        """
        signals = []
        for video_path in self.manifest.keys():
            signal_data = self.load_signal(video_path)
            if signal_data is not None:
                signals.append(signal_data)
        return signals

    def signal_exists(self, video_path: str) -> bool:
        """
        Check if signal is cached for a video

        Args:
            video_path: Path to video file

        Returns:
            True if signal exists, False otherwise
        """
        video_path = str(Path(video_path).resolve())
        if video_path not in self.manifest:
            return False

        cache_file = self.manifest[video_path]['cache_file']
        return Path(cache_file).exists()

    def get_videos_with_signals(self) -> List[str]:
        """
        Get list of all video paths with cached signals

        Returns:
            List of video paths
        """
        return list(self.manifest.keys())

    def count(self) -> int:
        """
        Get number of cached signals

        Returns:
            Number of signals in cache
        """
        return len(self.manifest)

    def get_signals_with_ground_truth(self) -> List[Dict]:
        """
        Get all cached signals that have ground truth labels

        Returns:
            List of signal data dictionaries with ground truth
        """
        signals = []
        for video_path, manifest_entry in self.manifest.items():
            if 'ground_truth_bpm' in manifest_entry:
                signal_data = self.load_signal(video_path)
                if signal_data is not None:
                    signals.append(signal_data)
        return signals
