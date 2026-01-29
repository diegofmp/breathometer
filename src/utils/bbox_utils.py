"""
Bounding box utility functions
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_inner_hand_bbox(hand_mask: np.ndarray,
                        hand_bbox: Tuple[int, int, int, int],
                        method: str = 'percentile',
                        margin_ratio: float = 0.15,
                        erosion_iterations: int = 3) -> Tuple[int, int, int, int]:
    """
    Extract inner bounding box from hand mask that focuses on the interior region
    where the bird is likely to be (avoiding fingers/edges).

    This handles both closed and open hands by finding the dense interior region.

    Args:
        hand_mask: Full-frame hand segmentation mask (uint8, 255=hand, 0=background)
        hand_bbox: Original hand bounding box (x, y, w, h)
        method: Method to compute inner bbox
            - 'percentile': Use percentile-based cropping (works with open hands)
            - 'erosion': Morphological erosion (works best with closed hands)
            - 'contour': Find largest inner contour after erosion
        margin_ratio: For 'percentile' method, percentage to crop from edges (0.15 = 15%)
        erosion_iterations: For 'erosion'/'contour' methods, number of erosion iterations

    Returns:
        inner_bbox: Inner bounding box (x, y, w, h) in frame coordinates
    """
    # Ensure bbox coordinates are integers
    x, y, w, h = [int(v) for v in hand_bbox]

    # Extract hand mask in local coordinates
    hand_mask_local = hand_mask[y:y+h, x:x+w].copy()

    if np.sum(hand_mask_local) < 100:
        # Not enough hand pixels, return original bbox
        return hand_bbox

    if method == 'percentile':
        # Percentile-based cropping: works well with open/irregular hands
        # Find pixel positions of all hand pixels
        hand_pixels = np.argwhere(hand_mask_local > 0)  # Returns (row, col) = (y, x)

        if len(hand_pixels) < 100:
            return hand_bbox

        # Get percentile bounds to exclude outer edges
        # This finds the "core" region where most hand pixels are concentrated
        y_coords = hand_pixels[:, 0]
        x_coords = hand_pixels[:, 1]

        # Use margin_ratio to determine percentiles
        # e.g., margin_ratio=0.15 -> use 15th to 85th percentile
        lower_p = margin_ratio * 100
        upper_p = (1 - margin_ratio) * 100

        y_min = int(np.percentile(y_coords, lower_p))
        y_max = int(np.percentile(y_coords, upper_p))
        x_min = int(np.percentile(x_coords, lower_p))
        x_max = int(np.percentile(x_coords, upper_p))

        # Convert to bbox
        inner_w = max(x_max - x_min, 20)  # Minimum width
        inner_h = max(y_max - y_min, 20)  # Minimum height

        # Convert to frame coordinates
        inner_x = x + x_min
        inner_y = y + y_min

    elif method == 'erosion':
        # Morphological erosion: shrinks mask inward, removing thin protrusions (fingers)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eroded = cv2.erode(hand_mask_local, kernel, iterations=erosion_iterations)

        if np.sum(eroded) < 100:
            # Erosion removed too much, try with less iterations
            eroded = cv2.erode(hand_mask_local, kernel, iterations=max(1, erosion_iterations // 2))

        if np.sum(eroded) < 100:
            # Still too aggressive, return original
            return hand_bbox

        # Get bounding box of eroded mask
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return hand_bbox

        largest_contour = max(contours, key=cv2.contourArea)
        ex, ey, ew, eh = cv2.boundingRect(largest_contour)

        # Convert to frame coordinates
        inner_x = x + ex
        inner_y = y + ey
        inner_w = ew
        inner_h = eh

    elif method == 'contour':
        # Hybrid: erosion + contour analysis
        # First erode to remove fingers
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded = cv2.erode(hand_mask_local, kernel, iterations=erosion_iterations)

        # Find all remaining contours (handles open hands with multiple regions)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return hand_bbox

        # Get the largest contour (main palm/interior region)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get convex hull to "close" small gaps (handles slightly open hands)
        hull = cv2.convexHull(largest_contour)

        # Get bounding box of hull
        ex, ey, ew, eh = cv2.boundingRect(hull)

        # Convert to frame coordinates
        inner_x = x + ex
        inner_y = y + ey
        inner_w = ew
        inner_h = eh

    else:
        # Unknown method, return original
        return hand_bbox

    # Ensure minimum size
    inner_w = max(inner_w, 30)
    inner_h = max(inner_h, 30)

    # Ensure inner bbox is within original bbox
    inner_x = max(x, min(inner_x, x + w - inner_w))
    inner_y = max(y, min(inner_y, y + h - inner_h))
    inner_w = min(inner_w, x + w - inner_x)
    inner_h = min(inner_h, y + h - inner_y)

    return (inner_x, inner_y, inner_w, inner_h)


def visualize_bbox_comparison(frame: np.ndarray,
                               hand_bbox: Tuple[int, int, int, int],
                               inner_bbox: Tuple[int, int, int, int],
                               hand_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualize comparison between full hand bbox and inner bbox

    Args:
        frame: Input frame
        hand_bbox: Full hand bounding box (x, y, w, h)
        inner_bbox: Inner hand bounding box (x, y, w, h)
        hand_mask: Optional hand mask for visualization

    Returns:
        Visualization frame
    """
    vis = frame.copy()

    # Draw hand mask if available
    if hand_mask is not None:
        overlay = vis.copy()
        overlay[hand_mask > 0] = overlay[hand_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # Draw full hand bbox (blue)
    hx, hy, hw, hh = [int(v) for v in hand_bbox]
    cv2.rectangle(vis, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 2)
    cv2.putText(vis, "Full Hand", (hx, hy-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw inner bbox (yellow/green)
    ix, iy, iw, ih = [int(v) for v in inner_bbox]
    cv2.rectangle(vis, (ix, iy), (ix+iw, iy+ih), (0, 255, 255), 3)
    cv2.putText(vis, "Inner (Bird Region)", (ix, iy-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show size reduction
    area_reduction = (1 - (iw * ih) / (hw * hh)) * 100
    info_text = f"Area reduction: {area_reduction:.1f}%"
    cv2.putText(vis, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis
