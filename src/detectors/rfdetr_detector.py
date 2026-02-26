"""
RF-DETR-based hand detector (segmentation variant)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image



try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge
    _RFDETR_AVAILABLE = True
except ImportError:
    _RFDETR_AVAILABLE = False

from .base_detector import BaseDetector
from rfdetr import RFDETRBase

class RFDETRDetector(BaseDetector):
    """
    Universal RF-DETR detector (Real-time Detection Transformer).

    Uses the rfdetr package segmentation model. Results are in bbox corner
    (x, y, w, h) format, binary uint8 mask, and a confidence score in [0, 1].

    Supports both pre-trained models and custom fine-tuned models.
    Can detect different classes based on configuration.

    Activate with:
        detection:
          mode: 'rfdetr'
          model: 'rfdetr-base.pt'  # or path to custom fine-tuned model
          rfdetr_variant: 'medium'  # 'nano', 'small', 'medium', 'large', 'xlarge'
          target_class: 'hand'      # or 'bird' for fine-tuned bird models
          confidence_threshold: 0.5
          enable_postprocessing: true  # skin-tone filtering (only for 'hand')
          device: 'auto'
    """

    def __init__(self, config: dict):
        if not _RFDETR_AVAILABLE:
            raise ImportError(
                "The 'rfdetr' package is not installed. "
                "Install it with: pip install rfdetr"
            )

        super().__init__(config)

        # Device selection
        device = config.get('device', 'auto')
        if device == 'auto':
            if _TORCH_AVAILABLE:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Target class to detect (e.g., 'hand', 'bird')
        self.target_class = config.get('target_class', 'hand').lower()

        # Enable post-processing (skin-tone filtering for hands)
        self.enable_postprocessing = config.get('enable_postprocessing', True)
        # Disable post-processing for non-hand classes
        if self.target_class != 'hand':
            self.enable_postprocessing = False

        # Model variant: 'nano', 'small', 'medium', 'large', 'xlarge'
        model_variant = config.get('rfdetr_variant', 'medium')
        model_path = config.get('model_path', None)

        print(f"Loading RF-DETR model (variant={model_variant}, target_class={self.target_class}) on {self.device}")

        _SEG_VARIANTS = {
            'nano': RFDETRSegNano,
            'small': RFDETRSegSmall,
            'medium': RFDETRSegMedium,
            'large': RFDETRSegLarge,
            'xlarge': RFDETRSegXLarge,
        }
        ModelClass = _SEG_VARIANTS.get(model_variant, RFDETRSegMedium)

        # Load model with custom weights if provided
        if model_path:
            print(f"  Loading custom model from: {model_path}")
            self.model = RFDETRSegSmall(pretrain_weights=model_path)
            #self.model = ModelClass(pretrain_weights=model_path)
        else:
            print(f"  Loading pre-trained {model_variant} model")
            self.model = ModelClass()

        # Don't optimize for inference to allow flexible batch sizes
        # If you want single-image optimization, call: model.optimize_for_inference(batch_size=1)
        # For batch inference, don't optimize or use: model.optimize_for_inference(batch_size=N)
        # self.model.optimize_for_inference()

        # Confidence threshold - use class-specific defaults
        default_threshold = 0.3 if self.target_class == 'hand' else 0.2
        self.confidence_threshold = config.get('confidence_threshold', default_threshold)

        print(f"✓ RFDETRDetector initialized (device: {self.device}, "
              f"target: {self.target_class}, threshold: {self.confidence_threshold}, "
              f"postprocessing: {self.enable_postprocessing})")

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]:
        """
        Detect target object using RF-DETR (single frame).

        Args:
            frame: Input frame (BGR uint8 numpy array)

        Returns:
            bbox: (x, y, w, h) corner format or None
            confidence: float in [0, 1]
            mask: Binary uint8 mask (same H x W as frame) or None
        """
        # RF-DETR expects a PIL Image in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            results = self.model.predict(pil_image, threshold=self.confidence_threshold)
        except Exception as e:
            print(f"Warning: RF-DETR inference failed: {e}")
            return None, 0.0, None
        
        

        # Process the single result using shared logic
        return self._process_single_result(results, frame)

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]]:
        """
        Detect target objects in multiple frames using batch inference.

        This is significantly faster than calling detect() in a loop because:
        - Single forward pass for all frames (vs N forward passes)
        - Reduced CUDA kernel launch overhead
        - Better GPU utilization through parallelism

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

        # Convert all frames to PIL Images (RGB)
        pil_images = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(frame_rgb))

        # Batch inference - single forward pass for all images
        try:
            results_list = self.model.predict(pil_images, threshold=self.confidence_threshold)
            #print(f"Batch inference completed for {len(frames)} frames")
        except Exception as e:
            print(f"Warning: RF-DETR batch inference failed: {e}")
            # Fallback to individual detection
            return [self.detect(frame) for frame in frames]

        # Process each result
        output = []
        for idx, (frame, results) in enumerate(zip(frames, results_list)):
            bbox, confidence, mask = self._process_single_result(results, frame)
            output.append((bbox, confidence, mask))

        # Clean up GPU memory after batch
        if _TORCH_AVAILABLE and self.device == 'cuda':
            torch.cuda.empty_cache()

        return output

    def _process_single_result(
        self, results, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]:
        """
        Process a single RF-DETR result object (extracted from batch or single inference).

        Args:
            results: RF-DETR prediction results for one image
            frame: Original BGR frame (for post-processing and mask sizing)

        Returns:
            bbox: (x, y, w, h) corner format or None
            confidence: float in [0, 1]
            mask: Binary uint8 mask or None
        """
        # Extract and validate confidence scores
        scores = self._to_numpy_1d(results.confidence)
        if scores is None or len(scores) == 0:
            return None, 0.0, None

        # Filter detections by target class name
        class_names = self._extract_class_names(results)

        if class_names is not None and len(class_names) > 0:
            # Find indices matching target class
            target_indices = [i for i, name in enumerate(class_names)
                            if name.lower() == self.target_class]

            if len(target_indices) == 0:
                return None, 0.0, None

            # Filter scores to only target class
            target_scores = scores[target_indices]

            # Find best detection among target class
            best_target_idx = int(np.argmax(target_scores))
            best_idx = target_indices[best_target_idx]
            confidence = float(target_scores[best_target_idx])

            # Additional confidence check
            if confidence < self.confidence_threshold:
                return None, 0.0, None
        else:
            # Fallback: if class names not available, use highest confidence detection
            valid = scores >= self.confidence_threshold
            if not np.any(valid):
                return None, 0.0, None

            best_idx = int(np.argmax(scores))
            confidence = float(scores[best_idx])

        # Extract bbox — RF-DETR returns xyxy format in results.xyxy
        boxes = self._to_numpy_2d(results.xyxy)
        if boxes is None or len(boxes) == 0:
            return None, 0.0, None

        x1, y1, x2, y2 = boxes[best_idx]

        # Convert xyxy → corner (x, y, w, h)
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        bbox = (x, y, w, h)

        # Extract segmentation mask if available
        mask = None
        if hasattr(results, 'mask') and results.mask is not None:
            try:
                mask_data = self._extract_mask(results.mask, best_idx)
                if mask_data is not None:
                    h_frame, w_frame = frame.shape[:2]
                    if mask_data.shape != (h_frame, w_frame):
                        mask = cv2.resize(
                            mask_data.astype(np.float32),
                            (w_frame, h_frame),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        mask = mask_data.astype(np.float32)

                    mask = (mask > 0.5).astype(np.uint8) * 255

            except Exception as e:
                print(f"Warning: Could not extract RF-DETR {self.target_class} mask: {e}")
                mask = None

        # Apply post-processing only if enabled (e.g., for hand detection)
        if mask is not None and self.enable_postprocessing:
            mask = self.post_process(frame, bbox, mask)

        return bbox, confidence, mask

    def post_process(self, frame, hand_bbox, hand_mask):
        hsv_hue_min = 0
        hsv_hue_max = 20
        hsv_sat_min = 20
        hsv_val_min = 70
        ycrcb_cr_min = 133
        ycrcb_cr_max = 173
        ycrcb_cb_min = 77
        ycrcb_cb_max = 127

        mask_hsv = self._hsv_filter(frame, hsv_hue_min, hsv_hue_max, hsv_sat_min, hsv_val_min)
        mask_ycrcb = self._ycrcb_filter(
            frame, ycrcb_cr_min, ycrcb_cr_max, ycrcb_cb_min, ycrcb_cb_max
        )

        skin_combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        hand_mask_tuned = cv2.bitwise_or(skin_combined, hand_mask)
        return hand_mask_tuned

    def _hsv_filter(self, image, hsv_hue_min, hsv_hue_max, hsv_sat_min, hsv_val_min):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        roi_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([hsv_hue_min, hsv_sat_min, hsv_val_min], dtype=np.uint8)
        upper_hsv = np.array([hsv_hue_max, 255, 255], dtype=np.uint8)
        return cv2.inRange(roi_hsv, lower_hsv, upper_hsv)

    def _ycrcb_filter(self, image, cr_min, cr_max, cb_min, cb_max):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        roi_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        lower_ycrcb = np.array([0, cr_min, cb_min], dtype=np.uint8)
        upper_ycrcb = np.array([255, cr_max, cb_max], dtype=np.uint8)
        return cv2.inRange(roi_ycrcb, lower_ycrcb, upper_ycrcb)

    # ------------------------------------------------------------------
    # Private helpers for normalizing rfdetr result formats
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_class_names(results) -> Optional[list]:
        """
        Extract class names from RF-DETR results.

        Args:
            results: RF-DETR prediction results object

        Returns:
            List of class names (strings) or None if not available
        """
        try:
            # Try to access class names from results
            if hasattr(results, 'names'):
                # results.names might be a dict mapping class_id -> name
                if isinstance(results.names, dict):
                    # Get class IDs for each detection
                    if hasattr(results, 'cls'):
                        cls_ids = results.cls
                        if hasattr(cls_ids, 'cpu'):
                            cls_ids = cls_ids.cpu().numpy()
                        else:
                            cls_ids = np.asarray(cls_ids)

                        # Map IDs to names
                        return [results.names[int(cid)] for cid in cls_ids]

                # results.names might be a list
                elif isinstance(results.names, (list, tuple)):
                    if hasattr(results, 'cls'):
                        cls_ids = results.cls
                        if hasattr(cls_ids, 'cpu'):
                            cls_ids = cls_ids.cpu().numpy()
                        else:
                            cls_ids = np.asarray(cls_ids)

                        return [results.names[int(cid)] for cid in cls_ids]

            # Alternative: check for 'classes' attribute
            if hasattr(results, 'classes'):
                classes = results.classes
                if isinstance(classes, (list, tuple)):
                    return list(classes)

            # Try getattr for 'class' since it's a keyword
            if hasattr(results, 'class'):
                classes = getattr(results, 'class')
                if isinstance(classes, (list, tuple)):
                    return list(classes)

            return None
        except Exception as e:
            print(f"Warning: Could not extract class names: {e}")
            return None

    @staticmethod
    def _to_numpy_1d(tensor_or_array) -> Optional[np.ndarray]:
        """Convert tensor or array to a flat numpy array, or return None."""
        if tensor_or_array is None:
            return None
        try:
            if hasattr(tensor_or_array, 'cpu'):
                return tensor_or_array.cpu().numpy().flatten()
            return np.asarray(tensor_or_array).flatten()
        except Exception:
            return None

    @staticmethod
    def _to_numpy_2d(tensor_or_array) -> Optional[np.ndarray]:
        """Convert bounding box tensor/array to (N, 4) numpy array, or None."""
        if tensor_or_array is None:
            return None
        try:
            if hasattr(tensor_or_array, 'cpu'):
                arr = tensor_or_array.cpu().numpy()
            else:
                arr = np.asarray(tensor_or_array)
            if arr.ndim == 1 and len(arr) == 4:
                return arr.reshape(1, 4)
            return arr
        except Exception:
            return None

    @staticmethod
    def _extract_mask(masks_obj, idx: int) -> Optional[np.ndarray]:
        """
        Extract a single mask at index idx from the masks container.

        Handles:
          - Object with .data attribute of shape (N, H, W)
          - Tensor/array of shape (N, H, W)
          - Single mask of shape (H, W) when N=1
        """
        try:
            if hasattr(masks_obj, 'data'):
                mask_data = masks_obj.data
            else:
                mask_data = masks_obj

            if hasattr(mask_data, 'cpu'):
                arr = mask_data.cpu().numpy()
            else:
                arr = np.asarray(mask_data)

            if arr.ndim == 3:
                return arr[idx]
            elif arr.ndim == 2:
                return arr
            else:
                return None
        except (IndexError, Exception):
            return None
