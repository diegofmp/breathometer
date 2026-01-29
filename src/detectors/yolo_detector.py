"""
YOLO-based hand detector
"""

import cv2
import torch
import numpy as np
from typing import Tuple, Optional
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
        Detect hand using YOLO

        Args:
            frame: Input frame

        Returns:
            hand_bbox: (x, y, w, h) in corner format, or None
            confidence: Detection confidence
            hand_mask: Binary mask of hand pixels, or None if segmentation not available
        """
        # Run detection
        results = self.model(frame, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0, None

        # Get boxes and confidences
        boxes = results[0].boxes
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
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            try:
                # Get the mask for the best detection
                mask_data = results[0].masks.data[best_idx].cpu().numpy()

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

        return hand_bbox, confidence, hand_mask
