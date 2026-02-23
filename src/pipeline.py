"""
Main breathing analysis pipeline
"""

from collections import deque
import matplotlib.pyplot as plt

import cv2
import numpy as np
import yaml
from typing import Optional, Dict
from tqdm import tqdm

from src.detectors import get_detector
from src.localizers import get_localizer
from src.measurements import get_measurement
from src.signal_processing import SignalProcessor
from src.utils.data_utils import Segment, extract_bird_mask, verify_hand_segmentation



class BreathingAnalyzer:
    """
    Complete breathing analysis pipeline RELYING ON THE BIRD MASKS!
    
    Phases:
    1. Initialization: Locate ROI (detect the bird and hand masks, locate chest by energy content)
    2. Tracking: Track hand and chest ROIs
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
        self.detector = get_detector(self.config['detection']) # un Auto mode: HAND detector. Manual: acts directly as CHEST "detector"
        self.bird_detector = get_detector(self.config['segmentation']) # TODO double check naming on config file
        self.rely_segmentator = self.config['segmentation'].get('rely_segmentator', True)

        # Check detection mode
        self.detection_mode = self.config['detection'].get('mode', 'auto')
        self.manual_mode = (self.detection_mode == 'manual')

        #buffers config
        self.buffer_frames_size = self.config["localization"]['custom_localizer'].get('buffer_frames', 30)
        self.buffer_frames_for_masks = self.config["localization"]['custom_localizer'].get('hand_mask_buffer_frames', 20)
        self.debug_plots=False

        # Skip segmentation and localization in manual mode
        if self.manual_mode:
            self.segmentation_enabled = False
            self.segmenter = None
            self.localizer = None
            print("⚙ MANUAL MODE: User will select chest ROI directly")
            print("⚠ Skipping: hand detection, bird segmentation, chest localization")
        else:
            # Check if segmentation is enabled
            self.segmentation_enabled = self.config['segmentation'].get('enabled', True)
            print("SELF__SEGMENTATION ENABE? ", self.segmentation_enabled)

            self.localizer = get_localizer(self.config['localization'])
        self.measurement = get_measurement(self.config['measurement'])
        self.signal_processor = SignalProcessor(self.config['signal_processing'])

        # buffers
        self.buffer_frames = deque(maxlen=self.buffer_frames_size) 

        # Tracking
        self.hand_tracker = None
        self.chest_tracker = None
        self.redetect_interval = self.config['tracking'].get('redetect_interval', 0)
        self.start_frame = self.config['tracking'].get('start_frame', 0)
        self.max_frames = self.config['tracking'].get('max_frames', None)

        # State
        self.prev_frame = None
        self.prev_hand_bbox = None

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

    def _initialize_tracker(self, frame, roi):
        tracker_type = self.config['tracking']['chest_tracker']
        self.chest_tracker = self._create_tracker(tracker_type)

        roi_tuple = tuple(int(v) for v in roi)
        self.chest_tracker.init(frame, roi_tuple)
        #self.tracking_status.append(1)

    def _generate_distance_mask(self, segment: Segment) -> np.ndarray:
        """
        Generate a distance gradient from center
        """

        h, w = segment.mask.shape
        bx, by, bw, bh = segment.bbox
        bird_center_x = bx + bw / 2
        bird_center_y = by + bh / 2

        # 1. Localized Distance Gradient
        y, x = np.ogrid[:h, :w]
        dist_x = (x - bird_center_x) / (bw / 2)
        dist_y = (y - bird_center_y) / (bh / 2)
        dist_norm = np.sqrt(dist_x**2 + dist_y**2)

        return dist_norm

    def _aggregate_masks(
        self,
        bird_masks: list[Segment],
        hand_masks: list[Segment],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fuse per-frame bird and hand masks into a single clean bird silhouette.

        Takes the list of per-frame ``Segment`` objects produced by
        ``_get_masks`` and combines them using a multi-step consensus strategy
        designed to handle mask noise, hand-bird overlap, and detection
        inconsistencies across frames.

        Steps:
            1. **Distance gradient** — A normalised distance map ``dist_norm``
               is computed for every pixel relative to the center of the most
               recent bird bounding box. Pixels at the center have distance 0;
               pixels at the bbox edge have distance ≈ 1.

            2. **Dynamic hand threshold ("Sanctuary Guard")** — The threshold
               required for a pixel to be classified as hand varies with
               ``dist_norm`` following a Gaussian drop-off:

               .. code-block:: text

                   thresh = 0.1 + 0.85 * exp(-1.5 * dist_norm²)

               This makes it nearly impossible to label bird-center pixels as
               hand (threshold ≈ 0.95), while pixels far from the bird bbox
               center are held to a strict standard (threshold ≈ 0.05 at
               corners). The resulting binary hand map is morphologically
               dilated by a 7×7 kernel to cover borderline regions.

            3. **Hand persistence consensus** — The hand masks from all frames
               are stacked and averaged. Only pixels whose mean hand activation
               exceeds the dynamic threshold (step 2) are kept as hand pixels.

            4. **Bird persistence with sanctuary bias** — The bird masks are
               similarly stacked and averaged. The acceptance threshold for bird
               pixels also depends on ``dist_norm``, but in the opposite
               direction: central pixels need only 20% persistence while
               peripheral pixels require up to 60%, biasing the mask toward a
               tight, reliable core region.

            5. **Combine and solidify** — The final bird mask is the set of
               pixels that pass the bird consensus (step 4) *and* are not
               covered by the dilated hand mask (step 3). The largest connected
               contour of that intersection is replaced by its convex hull to
               produce a filled, hole-free silhouette.

        Args:
            bird_masks (list[Segment]): Per-frame bird detections. Each
                ``Segment.mask`` is a uint8 binary image (0/255) in full-frame
                coordinates. The bbox from the *last* element is used as the
                reference for the distance gradient.
            hand_masks (list[Segment]): Per-frame hand detections, index-aligned
                with ``bird_masks``. Each ``Segment.mask`` is a uint8 binary
                image (0/255) in full-frame coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair
            ``(final_bird_mask, super_hand_mask)`` where:

                - ``final_bird_mask``: uint8 image (0/255), same shape as the
                  input masks, containing the convex-hull-filled bird silhouette
                  with hand regions removed.
                - ``super_hand_mask``: uint8 image (0/255), the dilated
                  consensus hand mask used to subtract hand pixels from the
                  bird result (useful for debugging or downstream filtering).
        """

        if len(bird_masks) == 0:
            raise Exception("No bird masks to aggregate!!")

        if len(bird_masks) == 0 or len(hand_masks) == 0:
            raise Exception("No masks to aggregate!!")
        
        # If we had not get a valid hand bbox, directly return a dummy shrinked BIRD bbox
        if len(hand_masks)==1 and hand_masks[0].source is None:
            # dummy hand mask => just a fully 0 empty mask
            h, w = self.buffer_frames[-1].shape[:2]
            super_hand = np.zeros((h, w), dtype=np.uint8)

            # final_mask => apply distance gradient to last bird mask, removing corners

            source_frame = bird_masks[-1].source
            # bx, by, bw, bh = bird_masks[-1].bbox
            # bird_center_x = bx + bw / 2
            # bird_center_y = by + bh / 2
            # y, x = np.ogrid[:h, :w]
            # dist_x = (x - bird_center_x) / (bw / 2)
            # dist_y = (y - bird_center_y) / (bh / 2)
            # dist_norm = np.sqrt(dist_x**2 + dist_y**2)
            # TODO: double check this option. do i need to set a h,w based on buffer_frameS? (as commented out)
            dist_norm = self._generate_distance_mask(bird_masks[-1])
            radial_mask = (dist_norm <= 1.0).astype(np.uint8)
            final_mask = (radial_mask * 255).astype(np.uint8)

            
            #plot_matrices([(source_frame, "Source frame"), (super_hand, "Dummy super hand"), (final_mask, "Dummy final bird mask")], suptitle="Dummy masks")

            # Returns: final_bird and hand masks
            return final_mask, super_hand

        # plot_matrices(
        #     [(bird_masks[0].mask, "Bird first mask"), (hand_masks[0].mask, "Hand first mask")],
        #     suptitle="Bird and Hand Masks (frist) - from agg masks")

        dist_norm = self._generate_distance_mask(bird_masks[-1])

        # 2. THE DYNAMIC THRESHOLD (The "Sanctuary Guard")
        # We use a steep drop-off. 
        # Center (0.0) -> 0.95 (Almost impossible to ban bird as hand)
        # Edge (1.0)   -> 0.2  (Standard skepticism)
        # Corners (>1.5) -> 0.05 (Extremely strict, cleans all shards)
        hand_dynamic_thresh = 0.1 + 0.85 * np.exp(-1.5 * dist_norm**2)

        # 3. Hand Persistence Consensus
        hand_stack = [h.mask.astype(np.uint8) for h in hand_masks]
        hand_persistence = np.mean(hand_stack, axis=0) / 255.0
        # Pixel is only a hand if it's persistent AND outside the sanctuary guard
        super_hand_bin = (hand_persistence > hand_dynamic_thresh).astype(np.uint8)
        super_hand = cv2.dilate(super_hand_bin * 255, np.ones((7, 7), np.uint8))

        # 4. Bird Persistence with Sanctuary Bias
        # In the center, we accept bird pixels even if they have lower persistence (0.2)
        bird_persistence = np.mean([b.mask for b in bird_masks], axis=0) / 255.0
        bird_dynamic_thresh = 0.2 + 0.4 * (1 - np.exp(-dist_norm**2))
        bird_consensus = (bird_persistence > bird_dynamic_thresh)

        if not self.rely_segmentator:
            bird_mask_consensus = bird_consensus & (super_hand == 0) # remove hand 
        else:
            bird_mask_consensus = bird_consensus # rely on bird masks only

        # 5. Combine and Solidify (Convex Hull)
        consensus = (bird_mask_consensus).astype(np.uint8) * 255

        contours, _ = cv2.findContours(consensus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(best_cnt)
            final_mask = np.zeros_like(consensus)
            cv2.drawContours(final_mask, [hull], -1, 255, -1)
        else:
            final_mask = consensus
        # Returns: final_bird and hand masks
        return final_mask, super_hand

    def _get_masks(self) -> tuple[list[Segment] | None, list[Segment] | None]:
        """
        Run hand and bird detection over the most recent buffered frames using batch inference.

        Uses batch inference for both hand and bird detectors (RF-DETR) to process
        multiple frames in a single forward pass, which is significantly faster than
        loop-based detection (2-5x speedup expected).

        Workflow:
            1. **Batch hand detection** — ``self.detector.detect_batch(frames)``
               processes all frames at once, returning bounding boxes, confidence
               scores, and binary segmentation masks.
            2. **Batch bird detection** — ``self.bird_detector.detect_batch(frames)``
               processes all frames at once for bird detection.
            3. **Quality validation** — Each hand detection is validated via
               ``verify_hand_segmentation``.
            4. **Pairing** — Valid hand/bird pairs are stored as ``Segment`` objects.

        Both results for a given frame are appended together, so the two
        returned lists always have the same length and are index-aligned
        (``detected_hands[i]`` and ``bird_results[i]`` correspond to the same
        source frame).

        Args:
            None. Reads from ``self.buffer_frames``, ``self.detector``,
            ``self.bird_detector``, and ``self.buffer_frames_for_masks``.

        Returns:
            tuple[list[Segment], list[Segment]]: A pair
            ``(detected_hands, bird_results)`` where each list contains
            ``Segment`` objects with ``bbox``, ``confidence``, ``mask``, and
            ``source`` fields.

            Returns ``(None, None)`` if no valid hand detection was found in
            any of the buffered frames.
        """

        hand_mask_limit = self.buffer_frames_for_masks
        frames_to_process = list(self.buffer_frames)[-hand_mask_limit:]

        detected_hands = []   # hand detections
        bird_results = []     # bird detections

        # Check if detectors support batch inference
        has_batch_support = hasattr(self.detector, 'detect_batch') and hasattr(self.bird_detector, 'detect_batch')

        if has_batch_support:
            # --- BATCH INFERENCE (FAST PATH) ---
            print(f"Running batch inference on {len(frames_to_process)} frames...")

            # Batch hand detection (single forward pass)
            hand_detections = self.detector.detect_batch(frames_to_process)

            # Batch bird detection (single forward pass)
            bird_detections = self.bird_detector.detect_batch(frames_to_process)

            # Process results for each frame
            for frame, (hand_bbox, confidence, hand_mask), (bird_bbox, _, pred_bird_mask) in \
                    zip(frames_to_process, hand_detections, bird_detections):

                if hand_bbox is None or hand_mask is None:
                    # Check if bird was found without hand
                    if bird_bbox is not None:
                        bird_results.append(
                            Segment(bbox=bird_bbox, confidence=None,
                                    mask=pred_bird_mask, source=frame))
                    continue

                # VERIFY QUALITY
                valid_hand_mask = verify_hand_segmentation(bbox=hand_bbox, mask=hand_mask, confidence=confidence)

                if valid_hand_mask:  # Valid hand
                    # Store it
                    detected_hands.append(
                        Segment(bbox=hand_bbox, confidence=confidence, mask=hand_mask, source=frame)
                    )

                    if bird_bbox is not None:  # If bird bbox available, get its mask
                        if pred_bird_mask is not None:
                            bird_mask = pred_bird_mask
                        else:
                            bird_mask = extract_bird_mask(frame, hand_mask, bird_bbox)

                        bird_results.append(
                            Segment(bbox=bird_bbox, confidence=None, mask=bird_mask, source=frame)
                        )
                else:  # Invalid hand
                    if bird_bbox is not None:  # If bird bbox, store it without mask
                        bird_results.append(
                            Segment(bbox=bird_bbox, confidence=None, mask=pred_bird_mask, source=frame))

        else:
            # --- FALLBACK: LOOP-BASED DETECTION (SLOW PATH) ---
            print(f"⚠ Batch inference not available, falling back to loop-based detection")

            for frame in frames_to_process:
                # --- Hand detection (RF-DETR) ---
                hand_bbox, confidence, hand_mask = self.detector.detect(frame)

                # --- Bird detection (fine-tuned YOLO) ---
                bird_bbox, _, pred_bird_mask = self.bird_detector.detect(frame)

                if hand_bbox is None or hand_mask is None:
                    # Check if bird was found without hand
                    if bird_bbox is not None:
                        bird_results.append(
                            Segment(bbox=bird_bbox, confidence=None,
                                    mask=pred_bird_mask, source=frame))
                    else:
                        print("No hand NOR bird found!!!!")

                    continue

                # VERIFY QUALITY
                valid_hand_mask = verify_hand_segmentation(bbox=hand_bbox, mask=hand_mask, confidence=confidence)

                if valid_hand_mask:  # Valid hand
                    # Store it
                    detected_hands.append(
                        Segment(bbox=hand_bbox, confidence=confidence, mask=hand_mask, source=frame)
                    )

                    if bird_bbox is not None:  # If bird bbox available, get its mask
                        if pred_bird_mask is not None:
                            bird_mask = pred_bird_mask
                        else:
                            bird_mask = extract_bird_mask(frame, hand_mask, bird_bbox)

                        bird_results.append(
                            Segment(bbox=bird_bbox, confidence=None, mask=bird_mask, source=frame)
                        )

                    else:  # No bird bbox available. Just continue
                        continue

                else:  # Invalid hand
                    if bird_bbox is not None:  # If bird bbox, store it without mask
                        bird_results.append(Segment(bbox=bird_bbox, confidence=None, mask=pred_bird_mask, source=frame))
                    else:  # No hand nor bird available. just warn
                        continue

        if self.debug_plots:
            # --- Visualize masks ---
            if bird_results:
                for idx, bird in enumerate(bird_results):
                    #bird = bird_results[-1]
                    hand = detected_hands[idx]
                    vis = bird.source.copy()
                    bx, by, bw, bh = bird.bbox
                    hx, hy, hw, hh = hand.bbox

                    hx = int(hx)
                    hy = int(hy)
                    hw = int(hw)
                    hh = int(hh)

                    # Overlay hand mask as semi-transparent green if available
                    if hand.mask is not None:
                        mask_bool = hand.mask > 127
                        overlay = vis.copy()
                        overlay[mask_bool] = (200, 0, 0)  # blue
                        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                    
                    # Draw hand bbox on top
                    cv2.rectangle(vis, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 2)
                    cv2.putText(vis, f"bird {hand.confidence:.2f}", (hx, hy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

                    # Overlay bird mask as semi-transparent green if available
                    if bird.mask is not None:
                        mask_bool = bird.mask > 127
                        overlay = vis.copy()
                        overlay[mask_bool] = (0, 200, 0)  # green
                        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

                    # Draw bird bbox on top
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(vis, f"bird {bird.confidence:.2f}", (bx, by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                    plt.title(f"Frame {idx} - Bird Detection (YOLO FT) - conf: B: {bird.confidence:.2f} H: {hand.confidence:.2f}")
                    plt.axis('off')

        if not detected_hands:
            if not bird_results:
                print('\n⚠ WARNING: No hand NOR bird detected in any frame')
            else:
                # Consistency check: reject if bbox area varies wildly (likely FP)
                areas = [b.bbox[2] * b.bbox[3] for b in bird_results if b.bbox is not None]
                if areas:
                    min_area, max_area = min(areas), max(areas)
                    if min_area > 0 and (max_area / min_area) > 1.5:
                        print(f'\n⚠ WARNING: Bird bbox area varies too much ({max_area/min_area:.2f}x) - likely false positives')
                        print('  Discarding bird results and adding dummy hand')
                        bird_results = []
                
                # Set a dummy hand mask for aggregation
                detected_hands.append(
                    Segment(bbox=None, confidence=None, mask=None, source=None))
                print(f'\n✓ No valid hands, but {len(bird_results)} birds - using dummy hand')


        # Print statistics (TODO> debug flag?)
        #avg_hand_conf = sum(s.confidence for s in detected_hands) / len(detected_hands)
        #avg_bird_conf = sum(s.confidence for s in bird_results) / len(bird_results) if bird_results else 0.0
        #print(f"Hand detections : {len(detected_hands)}/{hand_mask_limit} | avg conf: {avg_hand_conf:.2f}")
        #print(f"Bird detections : {len(bird_results)}/{len(detected_hands)}  | avg YOLO conf: {avg_bird_conf:.2f}")

        return detected_hands, bird_results
   
    def _locate_bird_roi(self) -> tuple[int, int, int, int]:
        """
        Estimate the bird chest ROI from the recent frame buffer.

        Runs the full detection-aggregation-localization pipeline over the
        last `buffer_frames_size` buffered frames to produce a stable ROI
        that the tracker can be initialized on.

        Steps:
            1. Validate that the frame buffer contains at least
               `buffer_frames_size` frames.
            2. Run hand (RF-DETR) and bird (YOLO) detection over the buffer
               via `_get_masks`, returning per-frame `Segment` lists.
            3. Aggregate the per-frame masks via `_aggregate_masks`:
               - Applies a distance-based dynamic threshold to separate hand
                 pixels from bird pixels.
               - Builds consensus masks using persistence across frames.
               - Fills the final bird mask with a convex hull.
            4. Pass the aggregated masks to `self.localizer.locate` to find
               the chest ROI within the clean bird silhouette.

        Raises:
            ValueError: If `self.buffer_frames` has fewer than
                `self.buffer_frames_size` frames.
            Exception: If no hand or bird was detected in any of the
                buffered frames.

        Returns:
            tuple[int, int, int, int]: The chest ROI as ``(x, y, w, h)``
                in frame coordinates, ready to initialize a tracker.
        """


        min_elements = self.buffer_frames_size
        if len(self.buffer_frames) < min_elements:
            raise ValueError(f"Need more frames to buffer; got {len(self.buffer_frames)}.")
        
        buffer_masks_hands, buffer_masks_birds = self._get_masks()

        if buffer_masks_birds is None and buffer_masks_hands is None:
            # TODO: how to handle this? should it just skip this frame?? set breathing=0?
            raise Exception(f"Could not locate any bird in the last {min_elements} frames!")

        # validate that we have at least one segmentator result (bird.mask)
        found_any_mask = any(obj.mask is not None for obj in buffer_masks_birds)
        if not found_any_mask:
            raise Exception(f"Could not locate any bird in the last {min_elements} frames!. Can not rely on bird segmentator. Disable it")
        
        print(f"Detection results. BIRD masks: {len(buffer_masks_birds)}, HAND masks: {len(buffer_masks_hands)}")
        agg_mask_bird, agg_mask_hand = self._aggregate_masks(bird_masks=buffer_masks_birds, hand_masks=buffer_masks_hands)

        # Locate ROI based on the movement within the clean bird mask
        #roi= self.localizer.locate_w_bird_mask(frame_buffer=self.buffer_frames, hand_mask=agg_mask_hand, bird_mask=agg_mask_bird)
        roi= self.localizer.locate(frame_buffer=self.buffer_frames, hand_mask=agg_mask_hand, bird_mask=agg_mask_bird)

        return roi

    def _relocate_roi(self, frame) -> Dict:
        """
        Locate the bird chest ROI from the frame buffer and reinitialize the tracker.

        Called on the very first processed frame, whenever periodic re-detection
        fires (`redetect_interval`), and whenever `_track_and_measure` reports a
        tracking failure. Delegates ROI estimation to `_locate_bird_roi`, stores
        the ROI dimensions for shape-preserving tracking, then reinitializes
        `self.chest_tracker` on the current frame.

        Side effects:
            - Updates ``self.init_roi_w`` and ``self.init_roi_h`` to the detected
              ROI dimensions (used by ``_track_and_measure`` to keep the box size
              constant across tracked frames).
            - Reinitializes ``self.chest_tracker`` via ``_initialize_tracker``.
            - Appends ``0.0`` to ``self.breathing_signal`` (initialization frame
              produces no real measurement).
            - Resets ``self.measurement`` state.
            - Stores a copy of the current frame in ``self.prev_frame``.

        Args:
            frame (np.ndarray): Current BGR video frame used to initialize the
                tracker.

        Returns:
            dict: A result dict with keys:
                - ``'chest_roi'`` (tuple[int,int,int,int]): ``(x, y, w, h)`` of
                  the detected ROI.
                - ``'breathing'`` (float): Always ``0.0`` for initialization frames.
                - ``'mode'`` (str): Always ``'detected'``.
        """

        try:
            bird_roi = self._locate_bird_roi()
            bx, by, bw, bh = bird_roi

            self.init_roi_w=bw
            self.init_roi_h=bh
            

            # (re)init tracker
            self._initialize_tracker(frame, bird_roi)

            # Append zero breathing for initialization frames
            
            # DO NOT SAVE IT. not real signal!
            #self.breathing_signal.append(0.0)

            self.prev_frame = frame.copy()


            ###########test#####
            cx, cy, cw, ch = [int(v) for v in bird_roi]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            chest_region = gray[cy:cy+ch, cx:cx+cw]
            
            # RESET updated: now it sets the argm as prev_chest, so we dont loose it
            self.measurement.reset(chest_region)
            ###################

            #self.measurement.reset()

            return {
                'chest_roi': bird_roi,
                'breathing': 0.0,
                'mode': 'detected'
            }
        
        except:
            # In case it couldnt find a suitable ROI: if tracker already exists, keep it. If not, throw error (failed to initialize very first ROI)
            if self.chest_tracker is not None:
                return {
                    'chest_roi': None,
                    'breathing': 0.0,
                    'mode': 'failed_to_update'
                }
            else:
                raise Exception("Could not initialize tracker. No valid ROI found. Increase buffer size or look for a more stable start_frame")
    
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
            actual_start_frame = self.start_frame - self.buffer_frames_size
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)

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
        if self.start_frame > 0:
            frame_idx = self.start_frame - self.buffer_frames_size
        else:
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

                # Still just buffering
                self.buffer_frames.append(frame)
                if frame_idx<self.start_frame:
                    frame_idx+=1                    
                    continue

                if self.manual_mode:
                    # If manual mode, prompt manual ROI only once to init tracker
                    if frame_idx == self.start_frame:
                        manual_chest_roi, confidence, chest_mask = self.detector.detect(frame)
                        print("MANUAL CHEST ROI:  ", manual_chest_roi)
                        bx, by, bw, bh = manual_chest_roi

                        self.init_roi_w=bw
                        self.init_roi_h=bh

                        if manual_chest_roi is None:
                            raise Exception("No manual chest ROI provided! Try again making sure to select a valid ROI")

                        self._initialize_tracker(frame, manual_chest_roi)                    
                else:
                    # (RE)Initialization (first frame, periodic re-detection, or trackers not initialized)
                    should_locate_roi = frame_idx == self.start_frame or (self.redetect_interval > 0 and frame_idx % self.redetect_interval == 0) or self.chest_tracker is None

                    if should_locate_roi:
                        self._relocate_roi(frame)
                        frame_idx+=1
                        continue
                
                # Tracking + Measurement
                result = self._track_and_measure(frame)
                
                # Get audio level for this frame
                audio_level = 0.0
                if len(audio_samples) > frames_processed:
                    audio_level = audio_samples[frames_processed]

                # Collect metadata (after processing to get hand/chest positions)
                hand_bbox = result.get('hand_bbox') if result else None
                chest_roi = result.get('chest_roi') if result else None
                tracker_status= result.get('mode') if result else None

                if tracker_status is not None and tracker_status== 'detected':
                    print(f"Tracked failed to track ROI at frame {frame_idx} and was REINITIALIZED")

                self._collect_metadata(frame, audio_level, hand_bbox, chest_roi)

                # Record tracking status
                self.tracking_status.append(1 if result is not None else 0)

                frame_idx += 1
                frames_processed += 1
                pbar.update(1)
        
        cap.release()
        if out:
            out.release()

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
            'metadata': self.metadata,
            'breath_counts': info.get('breath_counts', {}),
            'validation': info.get('validation', {})
        }
        
        return results
    
    def _track_and_measure(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Advance the chest tracker by one frame and extract a breathing signal sample.

        On each regular (non-initialization) frame, this method:
            1. Updates ``self.chest_tracker`` with the new frame. If tracking
               fails, falls back to ``_relocate_roi`` to re-detect and reinitialize.
            2. Recenters the tracked bounding box: uses the tracker-reported
               center but keeps the original ROI dimensions (``self.init_roi_w``,
               ``self.init_roi_h``) to prevent size drift over time.
            3. Extracts the chest region from the grayscale frame and passes it
               to ``self.measurement.measure`` to compute a single breathing
               signal value.
            4. Appends the measurement to ``self.breathing_signal`` and updates
               ``self.prev_frame``.

        Args:
            frame (np.ndarray): Current BGR video frame.

        Returns:
            dict or None: On success, a dict with keys:
                - ``'chest_roi'`` (tuple[float,float,float,float]): ``(x, y, w, h)``
                  of the shape-preserved chest ROI used for measurement.
                - ``'breathing'`` (float): The breathing signal value for this
                  frame (units depend on the configured measurement method).
                - ``'mode'`` (str): ``'tracked'``.

            If tracking fails, returns whatever ``_relocate_roi`` returns
            (i.e. a ``'detected'`` result dict).
        """
        # 2.1: Track chest
        success_chest, tracked_roi = self.chest_tracker.update(frame)

        if not success_chest:
            print("TRACKER WAS NOT SUCCESFUL!!! > gotta try to relocate_roi!")
            return self._relocate_roi(frame)        
        
        # keep the ROI shape consistent, gonna use the received chest_roi CENTER
        # Success: Extract center from tracker output and preserve original dimensions
        tx, ty, tw, th = tracked_roi
        # Calculate center of tracked ROI
        center_x = tx + tw / 2
        center_y = ty + th / 2
        # Build new ROI with original dimensions centered at tracked position
        new_x = center_x - self.init_roi_w / 2
        new_y = center_y - self.init_roi_h / 2
        chest_roi = (new_x, new_y, self.init_roi_w, self.init_roi_h)


    
        # 2.4: Measure breathing

        cx, cy, cw, ch = [int(v) for v in chest_roi]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chest_region = gray[cy:cy+ch, cx:cx+cw]


        breathing = self.measurement.measure(chest_region)
        
        self.breathing_signal.append(breathing)
        
        # Update state
        self.prev_frame = frame.copy()
        #self.prev_hand_bbox = hand_bbox
        
        return {
            #'hand_bbox': hand_bbox,
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
    
    def _collect_metadata(self, frame: np.ndarray, audio_level: float = 0.0,
                         hand_bbox: Optional[tuple] = None, chest_roi: Optional[tuple] = None):
        """
        Compute and accumulate per-frame quality/diagnostic metadata.

        Called once per processed frame (after tracking/measurement) to build
        parallel arrays in ``self.metadata`` that can later be used for signal
        quality analysis, outlier filtering, or debugging.

        The following metrics are appended on every call:
            - ``'brightness'``: Mean pixel intensity of the grayscale frame.
            - ``'brightness_change'``: Absolute mean-intensity difference from
              the previous frame (``0.0`` on the first call).
            - ``'motion'``: Mean per-pixel absolute difference between consecutive
              grayscale frames — a proxy for global camera/scene motion
              (``0.0`` on the first call).
            - ``'audio_level'``: Pre-computed RMS energy value for this frame,
              passed in from the caller (``0.0`` if audio is unavailable).
            - ``'hand_motion'``: Euclidean distance (pixels) the hand bbox
              center moved since the last frame. ``0.0`` if ``hand_bbox`` is
              ``None`` or on the first frame with a hand.
            - ``'chest_motion'``: Euclidean distance (pixels) the chest ROI
              center moved since the last frame. ``0.0`` if ``chest_roi`` is
              ``None`` or on the first frame with a chest ROI.

        Side effects:
            - Appends one value to each of the six lists in ``self.metadata``.
            - Updates ``self.prev_frame_gray``, ``self.prev_hand_center``, and
              ``self.prev_chest_center`` for the next frame's delta computations.

        Args:
            frame (np.ndarray): Current BGR video frame.
            audio_level (float): Pre-computed RMS audio energy for this frame.
                Defaults to ``0.0``.
            hand_bbox (tuple[int,int,int,int] or None): Hand bounding box as
                ``(x, y, w, h)``. Pass ``None`` when not available.
            chest_roi (tuple[int,int,int,int] or None): Chest ROI as
                ``(x, y, w, h)``. Pass ``None`` when not available.
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
        Extract the audio track from a video file and compute per-frame RMS energy.

        Uses ``ffmpeg`` (via subprocess) to decode the audio into a temporary
        mono 16-bit PCM WAV file at 44 100 Hz. The PCM samples are then
        normalized to ``[-1, 1]`` and chunked into segments that correspond to
        individual video frames, with the root-mean-square (RMS) energy computed
        for each chunk.

        The resulting array can be used as a proxy for ambient noise or handling
        events (e.g. the bird being disturbed) and is stored per frame in
        ``self.metadata['audio_level']`` via ``_collect_metadata``.

        Steps:
            1. Spawn ``ffmpeg`` to demux and decode the audio stream into a
               temporary ``.wav`` file (mono, 16-bit, 44 100 Hz).
            2. Read the WAV samples with the standard-library ``wave`` module
               and unpack them with ``struct``.
            3. Normalize samples to ``[-1.0, 1.0]``.
            4. Slice into per-frame windows of ``sample_rate / fps`` samples and
               compute the RMS of each window.
            5. Delete the temporary WAV file.

        Args:
            video_path (str): Absolute or relative path to the source video file.
            fps (float): Video frame rate, used to determine the number of audio
                samples per video frame.

        Returns:
            np.ndarray: 1-D float32 array of length ≈ ``total_video_frames``,
                where each element is the RMS audio energy in ``[0, 1]`` for the
                corresponding video frame. Returns an empty array
                (``np.array([])``) if ``ffmpeg`` is unavailable, the video has
                no audio track, or any other exception occurs.
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