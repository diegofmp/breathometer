"""
YOLO-based hand detector
"""

import cv2
import torch
import numpy as np
from typing import Tuple, Optional, List
from ultralytics import YOLO

from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """
    Hand detection using YOLOv8
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Determine device
        device = config.get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        model_name = config.get('model', 'yolov8n.pt')
        print(f"Loading YOLO model: {model_name} on {self.device}")
        
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        print(f"✓ YOLODetector initialized (device: {self.device})")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]:
        """
        Detect hand using YOLO (single frame)

        Args:
            frame: Input frame

        Returns:
            hand_bbox: (x, y, w, h) in corner format, or None
            confidence: Detection confidence
            hand_mask: Binary mask of hand pixels, or None if segmentation not available
        """
        # Run detection
        results = self.model(frame, verbose=False)

        if len(results) == 0:
            return None, 0.0, None

        # Process the single result using shared logic
        return self._process_single_result(results[0], frame)

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]]:
        """
        Detect objects in multiple frames using batch inference.

        YOLO's batch inference is extremely efficient - just pass a list of frames
        and it processes them all in a single forward pass.

        Args:
            frames: List of input frames (BGR uint8 numpy arrays)

        Returns:
            List of tuples, each containing:
                bbox: (x, y, w, h) corner format or None
                confidence: float in [0, 1]
                mask: Binary uint8 mask (same H x W as frame) or None
        """
        if not frames:
            return []

        # YOLO batch inference - single forward pass for all frames
        try:
            # YOLO automatically handles batch processing when given a list
            results_list = self.model(frames, verbose=False)
            print(f"YOLO batch inference completed for {len(frames)} frames")
        except Exception as e:
            print(f"Warning: YOLO batch inference failed: {e}")
            # Fallback to individual detection
            return [self.detect(frame) for frame in frames]

        # Process each result
        output = []
        for frame, results in zip(frames, results_list):
            bbox, confidence, mask = self._process_single_result(results, frame)
            output.append((bbox, confidence, mask))

        # Clean up GPU memory after batch
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.empty_cache()

        return output

    def _process_single_result(
        self, results, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]:
        """
        Process a single YOLO result object (extracted from batch or single inference).

        Args:
            results: YOLO prediction results for one image
            frame: Original BGR frame (for post-processing and mask sizing)

        Returns:
            bbox: (x, y, w, h) corner format or None
            confidence: float in [0, 1]
            mask: Binary uint8 mask or None
        """
        segmented_subject = results.names[0]

        if len(results.boxes) == 0:
            return None, 0.0, None

        # Get boxes and confidences
        boxes = results.boxes
        confidences = boxes.conf.cpu().numpy()

        # Filter by confidence
        valid_indices = confidences >= self.confidence_threshold

        if not np.any(valid_indices):
            return None, 0.0, None

        # Get highest confidence detection
        best_idx = np.argmax(confidences)
        confidence = float(confidences[best_idx])

        # Get bbox in xywh format (center)
        box_xywh = boxes.xywh[best_idx].cpu().numpy()
        cx, cy, w, h = box_xywh

        # Convert to corner format (x, y, w, h)
        x = cx - w / 2
        y = cy - h / 2

        hand_bbox = (float(x), float(y), float(w), float(h))

        # Extract segmentation mask if available
        hand_mask = None
        if hasattr(results, 'masks') and results.masks is not None:
            try:
                # Get the mask for the best detection
                mask_data = results.masks.data[best_idx].cpu().numpy()

                # Resize to frame size if needed
                h_frame, w_frame = frame.shape[:2]
                if mask_data.shape != (h_frame, w_frame):
                    hand_mask = cv2.resize(mask_data, (w_frame, h_frame), interpolation=cv2.INTER_LINEAR)
                else:
                    hand_mask = mask_data

                # Binarize
                hand_mask = (hand_mask > 0.5).astype(np.uint8) * 255
            except Exception as e:
                print(f"Warning: Could not extract hand mask: {e}")
                hand_mask = None

        # Post-process (HANDS ONLY!!!)
        if segmented_subject == 'person' and hand_mask is not None:
            hand_mask = self.post_process(frame, hand_bbox, hand_mask)

        return hand_bbox, confidence, hand_mask

    def post_process(self, frame, hand_bbox, hand_mask):
        hsv_hue_min= 0
        hsv_hue_max= 20
        hsv_sat_min= 20
        hsv_val_min= 70
        ycrcb_cr_min= 133
        ycrcb_cr_max= 173
        ycrcb_cb_min= 77
        ycrcb_cb_max= 127

        # Fusion: Combine skin detection with YOLO
        mask_hsv = self._hsv_filter(frame, hsv_hue_min, hsv_hue_max, hsv_sat_min, hsv_val_min)
        mask_ycrcb = self._ycrcb_filter(frame, ycrcb_cr_min, ycrcb_cr_max, ycrcb_cb_min, ycrcb_cb_max)

        #skin_combined = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        skin_combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Constrain by YOLO
        hand_mask_tuned = cv2.bitwise_and(skin_combined, hand_mask)
        return hand_mask_tuned
    
    def _hsv_filter(self, image, hsv_hue_min, hsv_hue_max, hsv_sat_min, hsv_val_min):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        roi_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        lower_hsv = np.array([hsv_hue_min, hsv_sat_min, hsv_val_min], dtype=np.uint8)
        upper_hsv = np.array([hsv_hue_max, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)
        return mask_hsv

    def _ycrcb_filter(self, image, cr_min, cr_max, cb_min, cb_max):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        roi_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        lower_ycrcb = np.array([0, cr_min, cb_min], dtype=np.uint8)
        upper_ycrcb = np.array([255, cr_max, cb_max], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(roi_ycrcb, lower_ycrcb, upper_ycrcb)
        return mask_ycrcb
    
    