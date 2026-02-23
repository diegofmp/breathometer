from collections import namedtuple
from typing import Tuple
import numpy as np
import cv2

Segment = namedtuple("Segment", ["bbox", "confidence", "mask", "source"])



def verify_hand_segmentation(bbox: Tuple, mask: np.array, confidence: float, min_fill_ratio: float=0.20) -> bool:
    bx, by, bw, bh = bbox
    bbox_area = bw * bh
    mask_area = np.sum(mask > 0)
    
    fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0
    
    # 1. Fill Ratio Check
    if fill_ratio < min_fill_ratio:
        return False
        
    # 2. Centroid Check (Optional)
    # Is the mask centered in its box? Fragments are often at the very edges.
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # If centroid is too close to the BBox edge, it's likely a shard
        if not (bx + 0.1*bw < cx < bx + 0.9*bw):
            return False

    return True

def extract_bird_mask(frame, hand_mask, bird_bbox):
    """
    Extract a bird-only mask by removing the hand mask from the bird bbox region.
    
    Args:
        frame: BGR image (H, W, 3)
        hand_mask: binary/grayscale mask of the hand (H, W), same size as frame
        bird_bbox: (x, y, w, h) bounding box of the bird in corner format
    
    Returns:
        bird_mask: binary mask (H, W) where 255 = bird pixels, 0 = background/hand
    """
    h, w = frame.shape[:2]
    bx, by, bw, bh = bird_bbox

    # Clamp bbox to frame bounds
    x1 = max(0, int(bx))
    y1 = max(0, int(by))
    x2 = min(w, int(bx + bw))
    y2 = min(h, int(by + bh))

    # Start with full bbox region as the bird area
    bird_mask = np.zeros((h, w), dtype=np.uint8)
    bird_mask[y1:y2, x1:x2] = 255

    # Binarize hand mask if not already binary
    if hand_mask is not None:
        hand_binary = (hand_mask > 127).astype(np.uint8) * 255 if hand_mask.max() > 1 else hand_mask
        # Subtract the hand from the bird region
        bird_mask = cv2.subtract(bird_mask, hand_binary)

    return bird_mask

