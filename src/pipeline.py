"""
Main breathing analysis pipeline
"""

import matplotlib.pyplot as plt

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from src.detectors import get_detector
from src.segmenters import get_segmenter, convert_mask_to_frame_coords
from src.localizers import get_localizer
from src.measurements import get_measurement
from src.signal_processing import SignalProcessor
from src.stabilizers import get_stabilizer
from src.utils import get_inner_hand_bbox


class BreathingAnalyzer:
    """
    Complete breathing analysis pipeline
    
    Phases:
    1. Initialization: Detect hand, segment bird, locate chest
    2. Tracking: Track hand and chest ROIs
    3. Stabilization (optional): Remove hand motion
    4. Measurement: Extract breathing signal
    5. Signal processing: Estimate breathing rate
    """
    
    def __init__(self, config_path: str = 'configs/default.yaml'):
        """
        Initialize analyzer with configuration

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("="*60)
        print("BIRD BREATHING ANALYZER")
        print("="*60)

        # Initialize components
        self.detector = get_detector(self.config['detection'])

        # Check if segmentation is enabled
        self.segmentation_enabled = self.config['segmentation'].get('enabled', True)
        if self.segmentation_enabled:
            self.segmenter = get_segmenter(self.config['segmentation'])
        else:
            self.segmenter = None
            print("⚠ Bird segmentation DISABLED - using hand mask for chest localization")

        self.localizer = get_localizer(self.config['localization'])
        self.measurement = get_measurement(self.config['measurement'])
        self.signal_processor = SignalProcessor(self.config['signal_processing'])
        self.stabilizer = get_stabilizer(self.config['stabilization'])

        # Tracking
        self.hand_tracker = None
        self.chest_tracker = None
        self.redetect_interval = self.config['tracking']['redetect_interval']
        self.start_frame = self.config['tracking'].get('start_frame', 0)
        self.max_frames = self.config['tracking'].get('max_frames', None)

        # Stabilization
        self.stabilization_enabled = self.config['stabilization']['enabled']

        # Breath counting parameters
        breath_config = self.config['signal_processing'].get('breath_counting', {})
        self.min_signal_length = breath_config.get('min_signal_length', 30)
        self.peak_prominence_ratio = breath_config.get('peak_prominence_ratio', 0.2)
        self.max_breathing_rate = breath_config.get('max_breathing_rate_bpm', 240)
        self.peak_threshold_ratio = breath_config.get('peak_height_threshold_ratio', 0.3)

        # State
        self.prev_frame = None
        self.prev_hand_bbox = None
        self.prev_chest_stabilized = None

        # Results
        self.breathing_signal = []
        self.tracking_status = []  # Track success/failure per frame
        self.metadata = {
            'brightness': [],  # Frame brightness (mean intensity)
            'brightness_change': [],  # Frame-to-frame brightness change
            'motion': [],  # Global motion estimate
            'audio_level': [],  # Audio noise level
            'hand_motion': [],  # Hand position change
            'chest_motion': [],  # Chest ROI position change
        }
        self.prev_frame_gray = None
        self.prev_hand_center = None
        self.prev_chest_center = None
        
        print("="*60)
        print("✓ All components initialized")
        print("="*60)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process video and estimate breathing rate
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save visualization video
        
        Returns:
            results: Dictionary with breathing rate and metadata
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Extract audio using ffmpeg-python
        audio_samples = self._extract_audio(video_path, fps)
        
        print(f"\nVideo: {video_path}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        if self.start_frame > 0:
            print(f"  Start frame: {self.start_frame} (skipping {self.start_frame/fps:.1f}s)")
        if self.max_frames is not None:
            print(f"  Max frames to process: {self.max_frames} ({self.max_frames/fps:.1f}s)")
        print()

        # Update signal processor with actual FPS
        self.signal_processor.fps = fps

        # Skip to start frame
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # Calculate frames to process
        frames_available = total_frames - self.start_frame
        if self.max_frames is not None:
            frames_to_process = min(self.max_frames, frames_available)
        else:
            frames_to_process = frames_available

        # Setup output video if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames
        frame_idx = self.start_frame
        frames_processed = 0
        self.breathing_signal = []
        self.tracking_status = []
        self.metadata = {
            'brightness': [],
            'brightness_change': [],
            'motion': [],
            'audio_level': [],
            'hand_motion': [],
            'chest_motion': [],
        }
        self.prev_frame_gray = None
        self.prev_hand_center = None
        self.prev_chest_center = None

        # Breath counting state for real-time display
        self.breath_count = 0
        self.last_peak_frame = -1
        self.peak_threshold = None
        self.current_rate_bpm = 0.0
        self.breath_flash_frames = 0  # Frames remaining to show breath flash
        self.beep_frames = []  # Track which frames have beeps for audio generation

        with tqdm(total=frames_to_process, desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if we've reached max frames limit
                if self.max_frames is not None and frames_processed >= self.max_frames:
                    break

                # Process frame
                result = self.process_frame(frame, frame_idx)

                # Get audio level for this frame
                audio_level = 0.0
                if len(audio_samples) > frames_processed:
                    audio_level = audio_samples[frames_processed]

                # Collect metadata (after processing to get hand/chest positions)
                hand_bbox = result.get('hand_bbox') if result else None
                chest_roi = result.get('chest_roi') if result else None
                self._collect_metadata(frame, audio_level, hand_bbox, chest_roi)

                # Record tracking status
                self.tracking_status.append(1 if result is not None else 0)

                # Real-time breath counting (simple peak detection)
                if len(self.breathing_signal) > 10:
                    self._update_breath_count(frames_processed, fps)

                # Visualization
                if output_path and result:
                    # Decrement flash counter
                    if self.breath_flash_frames > 0:
                        self.breath_flash_frames -= 1

                    vis_frame = self._visualize_frame(frame, result, frame_idx,
                                                     breath_count=self.breath_count,
                                                     current_rate=self.current_rate_bpm,
                                                     breath_detected=(self.breath_flash_frames > 0))
                    out.write(vis_frame)

                frame_idx += 1
                frames_processed += 1
                pbar.update(1)
        
        cap.release()
        if out:
            out.release()

        # Add audio beeps to output video
        if output_path:
            if len(self.beep_frames) > 0:
                print(f"\nAdding audio beeps ({len(self.beep_frames)} breaths detected)...")
                self._add_beeps_to_video(output_path, video_path, fps, frames_to_process)
            else:
                print(f"\n⚠ No breaths detected - no beeps to add")

        # Estimate breathing rate
        print("\nEstimating breathing rate...")
        
        if len(self.breathing_signal) < 30:
            print("⚠ Signal too short for analysis")
            breathing_rate = 0.0
            info = {}
        else:
            breathing_rate, info = self.signal_processor.estimate_breathing_rate(
                np.array(self.breathing_signal), fps
            )
        
        results = {
            'breathing_rate_bpm': breathing_rate,
            'confidence': info.get('confidence', 0.0),
            'frequency_hz': info.get('frequency_hz', 0.0),
            'signal_length': len(self.breathing_signal),
            'video_fps': fps,
            'total_frames': frame_idx,
            'breathing_signal': self.breathing_signal,
            'tracking_status': self.tracking_status,
            'metadata': self.metadata
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Breathing Rate (FFT): {breathing_rate:.1f} BPM")
        print(f"Confidence: {info.get('confidence', 0.0):.2f}")
        print(f"Frequency: {info.get('frequency_hz', 0.0):.2f} Hz")

        # Display breath counts
        if 'breath_counts' in info:
            print(f"\nBREATH COUNTS (Peak Detection):")
            breath_counts = info['breath_counts']
            for window, data in breath_counts.items():
                if window == 'full':
                    print(f"  Full duration ({data.get('duration_s', 0):.1f}s): "
                          f"{data['count']} breaths → {data['rate_bpm']:.1f} BPM")
                else:
                    print(f"  {window}: {data['count']} breaths → {data['rate_bpm']:.1f} BPM")

            # Validation
            if 'validation' in info:
                val = info['validation']
                status = "✓ Consistent" if val['is_consistent'] else "⚠ Inconsistent"
                print(f"\nValidation: {status} (CV: {val['cv']:.2%})")
                print(f"Mean rate across windows: {val['mean_rate']:.1f} BPM")

        print(f"{'='*60}\n")
        
        return results
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[Dict]:
        """
        Process single frame
        
        Args:
            frame: Input frame
            frame_idx: Frame index
        
        Returns:
            result: Dictionary with detection/tracking results
        """
        # PHASE 1: Initialization (first frame, periodic re-detection, or trackers not initialized)
        if (frame_idx == self.start_frame or
            frame_idx % self.redetect_interval == 0 or
            self.chest_tracker is None):
            return self._initialize_frame(frame)

        # PHASE 2: Tracking + Measurement
        return self._track_and_measure(frame)
    
    def _initialize_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Initialize: detect hand, segment bird (optional), locate chest
        """
        # 1.1: Detect hand
        hand_bbox, confidence, hand_mask = self.detector.detect(frame)

        if hand_bbox is None:
            return None

        # 1.1b: Extract inner hand bbox (focuses on bird region, avoiding fingers/edges)
        use_inner_bbox = self.config['detection'].get('use_inner_bbox', False)
        analysis_bbox = hand_bbox  # Default to full hand bbox

        if use_inner_bbox and hand_mask is not None:
            inner_method = self.config['detection'].get('inner_bbox_method', 'percentile')
            inner_margin = self.config['detection'].get('inner_bbox_margin', 0.15)
            inner_erosion = self.config['detection'].get('inner_bbox_erosion', 3)

            analysis_bbox = get_inner_hand_bbox(
                hand_mask,
                hand_bbox,
                method=inner_method,
                margin_ratio=inner_margin,
                erosion_iterations=inner_erosion
            )

        # 1.2: Segment bird (optional) - use analysis_bbox (inner or full)
        x, y, w, h = [int(v) for v in analysis_bbox]

        bird_mask = None
        if self.segmentation_enabled and self.segmenter is not None:
            # Bird segmentation enabled - use traditional pipeline
            analysis_region = frame[y:y+h, x:x+w]

            # Extract hand mask in local coordinates if available
            hand_mask_local = None
            if hand_mask is not None:
                hand_mask_local = hand_mask[y:y+h, x:x+w]

            bird_mask_local = self.segmenter.segment(analysis_region, hand_mask_local=hand_mask_local)
            bird_mask_local = self.segmenter.get_largest_component(bird_mask_local)

            if np.sum(bird_mask_local) < 100:
                return None

            bird_mask = convert_mask_to_frame_coords(bird_mask_local, analysis_bbox, frame.shape)

        # 1.3: Locate chest (pass frame and fps for advanced localizers)
        # Use bird_mask if available, otherwise create mask from analysis_bbox
        if bird_mask is not None:
            analysis_mask = bird_mask

            
        elif hand_mask is not None:
            # If using inner bbox, create a mask from the inner region
            if use_inner_bbox:
                print("USE INNER BBOX-> 366")
                # Create mask for inner bbox region (more constrained than full hand)
                analysis_mask = np.zeros_like(hand_mask)
                ix, iy, iw, ih = [int(v) for v in analysis_bbox]
                # Use the hand mask within the inner bbox region
                analysis_mask[iy:iy+ih, ix:ix+iw] = hand_mask[iy:iy+ih, ix:ix+iw]

                plt.imshow(hand_mask)
                plt.title("Hand Mask")
                plt.show()
                plt.close()

                plt.imshow(analysis_mask)
                plt.title("Analysis Mask")
                plt.show()
                plt.close()
            else:
                print("DO NOT USE INNER BBOX-> 383")
                # Use full hand mask
                analysis_mask = hand_mask
        else:
            return None

        if analysis_mask is None or np.sum(analysis_mask) < 100:
            return None

        chest_roi = self.localizer.locate(
            analysis_mask,
            hand_mask=hand_mask,
            frame=frame,
            fps=self.signal_processor.fps
        )

        if chest_roi is None:
            return None
        
        debug_mode = False
        
        if debug_mode:

            # DEBUG: Visualize chest ROI region
            cx, cy, cw, ch = [int(v) for v in chest_roi]
            chest_vis = frame.copy()

            # Draw all three boxes
            if hand_bbox is not None:
                hx, hy, hw, hh = [int(v) for v in hand_bbox]
                cv2.rectangle(chest_vis, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 2)  # Blue = hand

            if use_inner_bbox:
                ix, iy, iw, ih = [int(v) for v in analysis_bbox]
                cv2.rectangle(chest_vis, (ix, iy), (ix+iw, iy+ih), (0, 255, 255), 2)  # Cyan = inner

            cv2.rectangle(chest_vis, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 3)  # Red = chest

            # Highlight the chest region with color overlay
            overlay = chest_vis.copy()
            overlay[cy:cy+ch, cx:cx+cw] = overlay[cy:cy+ch, cx:cx+cw] * 0.5 + np.array([0, 255, 0]) * 0.5
            chest_vis = cv2.addWeighted(chest_vis, 0.7, overlay, 0.3, 0)

            # Show with matplotlib
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(chest_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Chest ROI Localization\nBlue=Hand, Cyan=Inner, Red+Green=Chest\nChest: {cw}x{ch} at ({cx},{cy})")
            plt.axis('off')
            plt.show()
            plt.close()
        
        # 1.4: Initialize trackers
        tracker_type = self.config['tracking']['hand_tracker']
        self.hand_tracker = self._create_tracker(tracker_type)
        hand_bbox_tuple = tuple(int(v) for v in hand_bbox)
        self.hand_tracker.init(frame, hand_bbox_tuple)

        tracker_type = self.config['tracking']['chest_tracker']
        self.chest_tracker = self._create_tracker(tracker_type)
        chest_roi_tuple = tuple(int(v) for v in chest_roi)
        self.chest_tracker.init(frame, chest_roi_tuple)
        
        # Store state
        self.prev_frame = frame.copy()
        self.prev_hand_bbox = hand_bbox
        self.measurement.reset()

        # Append zero breathing for initialization frames
        self.breathing_signal.append(0.0)

        return {
            'hand_bbox': hand_bbox,
            'hand_confidence': confidence,
            'inner_bbox': analysis_bbox if use_inner_bbox else None,  # NEW: for visualization
            'bird_mask': bird_mask,
            'chest_roi': chest_roi,
            'breathing': 0.0,
            'mode': 'detected'
        }
    
    def _track_and_measure(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Track ROIs and measure breathing
        """
        # 2.1: Track chest
        success_chest, chest_roi = self.chest_tracker.update(frame)

        if not success_chest:
            # Tracking failed - re-initialize
            print("  [Tracking lost - re-initializing]")
            return self._initialize_frame(frame)

        # 2.2: Track hand (needed for stabilization or visualization)
        # Only track if stabilization is enabled or visualization shows hand bbox
        hand_bbox = None
        if self.stabilizer is not None:
            success_hand, hand_bbox = self.hand_tracker.update(frame)
            if not success_hand:
                # Hand tracking failed - re-initialize
                print("  [Hand tracking lost - re-initializing]")
                return self._initialize_frame(frame)

        # 2.3: Stabilization (optional)
        cx, cy, cw, ch = [int(v) for v in chest_roi]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chest_region = gray[cy:cy+ch, cx:cx+cw]

        if self.stabilizer is not None:
            chest_stabilized = self.stabilizer.stabilize(
                chest_region,
                frame=frame,
                hand_bbox=hand_bbox,
                prev_hand_bbox=self.prev_hand_bbox,
                prev_frame=self.prev_frame
            )
        else:
            chest_stabilized = chest_region
        
        # 2.4: Measure breathing
        breathing = self.measurement.measure(chest_stabilized)
        
        self.breathing_signal.append(breathing)
        
        # Update state
        self.prev_frame = frame.copy()
        self.prev_hand_bbox = hand_bbox
        
        return {
            'hand_bbox': hand_bbox,
            'chest_roi': chest_roi,
            'breathing': breathing,
            'mode': 'tracked'
        }
    
    def _create_tracker(self, tracker_type: str):
        """Create OpenCV tracker"""
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        else:
            return cv2.TrackerKCF_create()
    
    def _visualize_frame(self, frame: np.ndarray, result: Dict, frame_idx: int = 0,
                        breath_count: int = 0, current_rate: float = 0.0,
                        breath_detected: bool = False) -> np.ndarray:
        """
        Visualize detection/tracking results with breath count

        Args:
            breath_detected: If True, highlights chest ROI to show breath was detected
        """
        vis = frame.copy()

        # Text positioning - left side with proper spacing
        left_x = 10
        y_pos = 30
        line_spacing = 35

        # Line 1: Breath count
        breath_text = f"Breaths: {breath_count}"
        cv2.putText(vis, breath_text, (left_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += line_spacing

        # Line 2: Current breathing rate
        if current_rate > 0:
            rate_text = f"Rate: {current_rate:.1f} BPM"
            cv2.putText(vis, rate_text, (left_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += line_spacing

        # Line 3: Current signal value
        if 'breathing' in result:
            signal_text = f"Signal: {result['breathing']:.2f}"
            cv2.putText(vis, signal_text, (left_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            y_pos += line_spacing

        # Line 4: Mode
        if 'mode' in result:
            mode_text = f"Mode: {result['mode']}"
            cv2.putText(vis, mode_text, (left_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Frame number in top-right corner
        frame_text = f"Frame: {frame_idx}"
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = vis.shape[1] - text_size[0] - 10
        cv2.putText(vis, frame_text, (text_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw hand bbox (blue)
        if 'hand_bbox' in result and result['hand_bbox'] is not None:
            x, y, w, h = [int(v) for v in result['hand_bbox']]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)

            label = "Hand"
            if 'hand_confidence' in result:
                label += f" {result['hand_confidence']:.2f}"

            cv2.putText(vis, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw inner bbox (yellow/cyan) - shows the reduced search region
        if 'inner_bbox' in result and result['inner_bbox'] is not None:
            ix, iy, iw, ih = [int(v) for v in result['inner_bbox']]
            cv2.rectangle(vis, (ix, iy), (ix+iw, iy+ih), (0, 255, 255), 2)
            cv2.putText(vis, "Inner", (ix, iy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw chest ROI with breath detection visual feedback
        if 'chest_roi' in result:
            cx, cy, cw, ch = [int(v) for v in result['chest_roi']]

            # Change color and thickness when breath is detected
            if breath_detected:
                # Yellow (BGR: 0, 255, 255) and thicker when breath detected
                roi_color = (0, 255, 255)
                roi_thickness = 4
            else:
                # Red (BGR: 0, 0, 255) and normal thickness otherwise
                roi_color = (0, 0, 255)
                roi_thickness = 2

            cv2.rectangle(vis, (cx, cy), (cx+cw, cy+ch), roi_color, roi_thickness)

            # Draw "Chest" label above the box
            cv2.putText(vis, "Chest", (cx, cy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

            # Draw breath count inside the box (top-left corner)
            count_text = f"#{breath_count}"
            # Position it inside the box with some padding
            count_x = cx + 5
            count_y = cy + 25
            cv2.putText(vis, count_text, (count_x, count_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

        return vis

    def _update_breath_count(self, current_frame: int, fps: float):
        """
        Update breath count in real-time using simple peak detection

        Uses configurable thresholds from config:
        - min_signal_length: Minimum frames needed before detection
        - peak_threshold_ratio: Height threshold (ratio of signal range above mean)
        - max_breathing_rate: Maximum BPM (sets minimum distance between peaks)
        """
        # Need minimum signal length for meaningful detection
        if len(self.breathing_signal) < self.min_signal_length:
            return

        # Initialize threshold on first call (recalculate periodically)
        # Use ROBUST statistics (median, percentiles) to avoid outlier influence
        if self.peak_threshold is None or len(self.breathing_signal) % 30 == 0:
            signal_array = np.array(self.breathing_signal)

            # Use median instead of mean (robust to outliers)
            signal_median = np.median(signal_array)

            # Use percentile range instead of full range (robust to outliers)
            # 10th to 90th percentile covers normal breathing range
            p10 = np.percentile(signal_array, 10)
            p90 = np.percentile(signal_array, 90)
            robust_range = p90 - p10

            # Peak must be above: median + (threshold_ratio * robust_range)
            self.peak_threshold = signal_median + robust_range * self.peak_threshold_ratio

        # Check if current value is a peak
        current_value = self.breathing_signal[-1]

        # Simple peak detection: current value is high and higher than neighbors
        if len(self.breathing_signal) >= 3:
            prev_value = self.breathing_signal[-2]
            prev_prev_value = self.breathing_signal[-3]

            # Peak conditions:
            # 1. Current value above threshold (amplitude requirement)
            # 2. Current value is local maximum (shape requirement)
            # 3. Enough frames since last peak (temporal requirement - prevent double counting)
            min_distance = int(fps / (self.max_breathing_rate / 60))

            is_above_threshold = current_value > self.peak_threshold
            is_local_max = current_value > prev_value and prev_value >= prev_prev_value
            enough_distance = (current_frame - self.last_peak_frame) >= min_distance

            if is_above_threshold and is_local_max and enough_distance:
                self.breath_count += 1
                self.last_peak_frame = current_frame

                # Trigger visual flash for 8 frames (~0.25s at 30fps)
                self.breath_flash_frames = 8

                # Mark this frame for audio beep (current_frame is already relative to processing start)
                self.beep_frames.append(current_frame)

                # Update breathing rate estimate (running average)
                if self.breath_count >= 2:
                    elapsed_time = (current_frame - self.start_frame) / fps
                    self.current_rate_bpm = (self.breath_count / elapsed_time) * 60

    def _collect_metadata(self, frame: np.ndarray, audio_level: float = 0.0,
                         hand_bbox: Optional[tuple] = None, chest_roi: Optional[tuple] = None):
        """
        Collect frame metadata (brightness, motion, audio, hand/chest motion)
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Brightness (mean intensity)
        brightness = np.mean(gray)
        self.metadata['brightness'].append(brightness)

        # 2. Brightness change (from previous frame)
        if self.prev_frame_gray is not None:
            brightness_prev = np.mean(self.prev_frame_gray)
            brightness_change = abs(brightness - brightness_prev)
            self.metadata['brightness_change'].append(brightness_change)
        else:
            self.metadata['brightness_change'].append(0.0)

        # 3. Global motion estimate (using frame difference)
        if self.prev_frame_gray is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame_gray)
            motion = np.mean(frame_diff)
            self.metadata['motion'].append(motion)
        else:
            self.metadata['motion'].append(0.0)

        # 4. Audio level
        self.metadata['audio_level'].append(audio_level)

        # 5. Hand motion (distance moved from previous frame)
        if hand_bbox is not None:
            hand_center = np.array([
                hand_bbox[0] + hand_bbox[2]/2,
                hand_bbox[1] + hand_bbox[3]/2
            ])
            if self.prev_hand_center is not None:
                hand_motion = np.linalg.norm(hand_center - self.prev_hand_center)
                self.metadata['hand_motion'].append(hand_motion)
            else:
                self.metadata['hand_motion'].append(0.0)
            self.prev_hand_center = hand_center.copy()
        else:
            self.metadata['hand_motion'].append(0.0)

        # 6. Chest motion (distance moved from previous frame)
        if chest_roi is not None:
            chest_center = np.array([
                chest_roi[0] + chest_roi[2]/2,
                chest_roi[1] + chest_roi[3]/2
            ])
            if self.prev_chest_center is not None:
                chest_motion = np.linalg.norm(chest_center - self.prev_chest_center)
                self.metadata['chest_motion'].append(chest_motion)
            else:
                self.metadata['chest_motion'].append(0.0)
            self.prev_chest_center = chest_center.copy()
        else:
            self.metadata['chest_motion'].append(0.0)

        # Store current frame for next iteration
        self.prev_frame_gray = gray.copy()

    def _extract_audio(self, video_path: str, fps: float) -> np.ndarray:
        """
        Extract audio from video and compute RMS energy per frame
        """
        try:
            import subprocess
            import tempfile
            import wave
            import struct

            # Extract audio to temporary WAV file using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_audio_path = tmp.name

            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '1', '-y', tmp_audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  Warning: Could not extract audio (ffmpeg not available or no audio track)")
                return np.array([])

            # Read WAV file
            with wave.open(tmp_audio_path, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)

                # Convert to numpy array
                samples = np.array(struct.unpack(f'{n_frames}h', audio_data), dtype=np.float32)
                samples = samples / 32768.0  # Normalize to [-1, 1]

            # Clean up temp file
            import os
            os.remove(tmp_audio_path)

            # Compute RMS energy per video frame
            samples_per_frame = int(sample_rate / fps)
            n_video_frames = int(len(samples) / samples_per_frame)

            audio_levels = []
            for i in range(n_video_frames):
                start_idx = i * samples_per_frame
                end_idx = start_idx + samples_per_frame
                frame_samples = samples[start_idx:end_idx]
                rms = np.sqrt(np.mean(frame_samples ** 2))
                audio_levels.append(rms)

            print(f"  Audio extracted: {len(audio_levels)} frames")
            return np.array(audio_levels)

        except Exception as e:
            print(f"  Warning: Audio extraction failed: {e}")
            return np.array([])

    def _add_beeps_to_video(self, video_path: str, original_video_path: str,
                            fps: float, total_frames: int):
        """
        Add audio beeps to the output video at frames where breaths were detected

        Args:
            video_path: Path to the video file (will be replaced with version containing beeps)
            original_video_path: Path to original video (for extracting original audio)
            fps: Video frame rate
            total_frames: Total number of frames processed
        """
        try:
            import subprocess
            import tempfile
            import wave
            import os

            print(f"  Generating beep audio track...")
            print(f"  Beep frames: {self.beep_frames[:5]}{'...' if len(self.beep_frames) > 5 else ''}")

            # Generate beep audio track
            sample_rate = 44100
            beep_freq = 800  # Hz (A5 note - pleasant beep sound)
            beep_duration = 0.15  # seconds
            beep_samples = int(sample_rate * beep_duration)

            # Create beep waveform (sine wave with envelope)
            t = np.linspace(0, beep_duration, beep_samples)
            beep = np.sin(2 * np.pi * beep_freq * t)

            # Apply envelope (fade in/out) to avoid clicks
            envelope = np.hanning(beep_samples)
            beep = beep * envelope * 0.5  # 0.5 = volume

            # Create full audio track
            samples_per_frame = int(sample_rate / fps)
            total_samples = total_frames * samples_per_frame
            audio_track = np.zeros(total_samples, dtype=np.float32)

            print(f"  Audio track: {total_samples} samples ({total_samples/sample_rate:.2f}s)")

            # Add beeps at specified frames
            for frame_idx in self.beep_frames:
                # Skip negative frame indices (shouldn't happen but safety check)
                if frame_idx < 0:
                    continue

                start_sample = frame_idx * samples_per_frame
                end_sample = start_sample + beep_samples

                if end_sample <= total_samples:
                    audio_track[start_sample:end_sample] += beep

            # Clip to [-1, 1] range
            audio_track = np.clip(audio_track, -1.0, 1.0)

            # Convert to 16-bit PCM
            audio_pcm = (audio_track * 32767).astype(np.int16)

            # Write beep audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                beep_audio_path = tmp.name

            with wave.open(beep_audio_path, 'wb') as wav:
                wav.setnchannels(1)  # Mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_pcm.tobytes())

            # Create temporary output file
            temp_output = video_path.replace('.mp4', '_temp.mp4')

            # Try to extract original audio and mix with beeps
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                original_audio_path = tmp.name

            # Extract original audio (accounting for start_frame offset)
            start_time = self.start_frame / fps
            duration = total_frames / fps

            extract_cmd = [
                'ffmpeg', '-i', original_video_path,
                '-ss', str(start_time), '-t', str(duration),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '1', '-y', original_audio_path
            ]

            result = subprocess.run(extract_cmd, capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(original_audio_path):
                # Mix original audio with beeps
                mix_cmd = [
                    'ffmpeg', '-i', video_path, '-i', original_audio_path,
                    '-i', beep_audio_path,
                    '-filter_complex', '[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=0[aout]',
                    '-map', '0:v', '-map', '[aout]',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-y', temp_output
                ]
            else:
                # No original audio - just add beeps
                mix_cmd = [
                    'ffmpeg', '-i', video_path, '-i', beep_audio_path,
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-shortest', '-y', temp_output
                ]

            print(f"  Running ffmpeg to mix audio...")
            result = subprocess.run(mix_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original file with version containing beeps
                os.replace(temp_output, video_path)
                print(f"  ✓ Audio beeps added successfully")
            else:
                print(f"  ⚠ Warning: Could not add beeps (ffmpeg error)")
                print(f"  Error output: {result.stderr[:500]}")  # First 500 chars
                if os.path.exists(temp_output):
                    os.remove(temp_output)

            # Cleanup temp files
            if os.path.exists(beep_audio_path):
                os.remove(beep_audio_path)
            if os.path.exists(original_audio_path):
                os.remove(original_audio_path)

        except Exception as e:
            import traceback
            print(f"  ⚠ Warning: Could not add audio beeps: {e}")
            print(f"  Traceback: {traceback.format_exc()[:500]}")
