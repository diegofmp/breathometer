"""
GrabCut-based segmentation
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .base_segmenter import BaseSegmenter


class GrabCutSegmenter(BaseSegmenter):
    """
    Segment bird using GrabCut algorithm
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.iterations = config.get('grabcut_iterations', 5)
        self.rect_margin = config.get('grabcut_rect_margin', 0.05)

        print(f"✓ GrabCutSegmenter initialized (iterations={self.iterations}, rect_margin={self.rect_margin})")
    
    def segment(self, hand_region: np.ndarray, hand_mask_local: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Segment bird using GrabCut

        Args:
            hand_region: Hand region image
            hand_mask_local: Optional hand segmentation mask in local coordinates
            **kwargs: Optional 'iterations' parameter

        Returns:
            bird_mask: Binary mask
        """
        h, w = hand_region.shape[:2]

        # Initialize mask
        mask = np.zeros((h, w), np.uint8)

        # Background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Create initial rectangle for GrabCut
        # Note: If using inner_bbox from pipeline, we're already in a reduced region,
        # so we don't need aggressive margins here
        rect = (
            int(w * self.rect_margin),
            int(h * self.rect_margin),
            int(w * (1 - 2 * self.rect_margin)),
            int(h * (1 - 2 * self.rect_margin))
        )

        # If hand mask is provided, use it to guide GrabCut
        # Mark pixels outside hand mask as definite background
        if hand_mask_local is not None:
            mask[hand_mask_local == 0] = cv2.GC_BGD

        try:
            # Run GrabCut
            iterations = kwargs.get('iterations', self.iterations)
            cv2.grabCut(
                hand_region, mask, rect, bgd_model, fgd_model,
                iterations, cv2.GC_INIT_WITH_RECT
            )
        except Exception as e:
            print(f"GrabCut failed: {e}")
            return np.zeros((h, w), dtype=np.uint8)

        # Create binary mask
        mask_binary = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype('uint8')

        # Clean up
        mask_binary = self._clean_mask(mask_binary)

        return mask_binary

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = binary_fill_holes(mask).astype(np.uint8) * 255
        return mask

    def get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract largest connected component
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if num_labels <= 1:
            return mask

        # Find largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Create mask with only largest component
        largest_mask = (labels == largest_label).astype(np.uint8) * 255

        return largest_mask
