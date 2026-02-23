"""
Manual detector for user-selected bounding box

This detector allows the user to manually select a chest ROI on the first frame,
bypassing hand detection, bird segmentation, and chest localization steps.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .base_detector import BaseDetector


class ManualDetector(BaseDetector):
    """
    Manual bounding box selection detector

    Shows the first frame and allows user to select chest ROI with mouse.
    The selected bbox is cached and used for all subsequent frames (with tracking).

    This mode bypasses:
    - Hand detection
    - Bird segmentation
    - Chest localization

    Usage:
    - Click and drag to select chest region
    - Press ENTER/SPACE to confirm
    - Press 'r' to reset selection
    - Press ESC to cancel
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Check if ROI is provided in config (for fair comparisons)
        manual_roi = config.get('manual_roi', None)
        if manual_roi is not None:
            # ROI provided directly, no need for user selection
            self.cached_bbox = tuple(manual_roi)
            self.selection_done = True
            print(f"✓ Using pre-specified ROI: x={manual_roi[0]:.0f}, y={manual_roi[1]:.0f}, "
                  f"w={manual_roi[2]:.0f}, h={manual_roi[3]:.0f}")
        else:
            self.cached_bbox = None
            self.selection_done = False

        self.window_name = "Manual Chest ROI Selection"

        # ROI selection state
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.current_frame = None

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[np.ndarray]]:
        """
        Detect chest ROI (manual selection on first call, cached thereafter)

        Args:
            frame: Input frame (BGR image)

        Returns:
            chest_bbox: (x, y, w, h) of manually selected chest ROI
            confidence: Always 1.0 (user selection)
            mask: Binary mask covering the selected bbox region
        """
        # If already selected, return cached bbox
        if self.selection_done and self.cached_bbox is not None:
            bbox = self.cached_bbox
            mask = self._create_mask_from_bbox(bbox, frame.shape)
            return bbox, 1.0, mask

        # First time: prompt user for selection
        print("\n" + "="*60)
        print("MANUAL CHEST ROI SELECTION")
        print("="*60)
        print("Instructions:")
        print("  1. Click and drag to select the chest region")
        print("  2. Press ENTER or SPACE to confirm selection")
        print("  3. Press 'r' to reset and select again")
        print("  4. Press ESC to cancel (will return None)")
        print("="*60 + "\n")

        self.current_frame = frame.copy()

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Selection loop
        while True:
            display_frame = self.current_frame.copy()

            # Draw current selection rectangle
            if self.start_point is not None and self.end_point is not None:
                cv2.rectangle(display_frame, self.start_point, self.end_point,
                            (0, 255, 0), 2)

                # Show bbox dimensions
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                text = f"ROI: {w}x{h} px"
                cv2.putText(display_frame, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show instructions on frame
            instructions = [
                "Click & drag to select chest ROI",
                "ENTER/SPACE: Confirm | 'r': Reset | ESC: Cancel"
            ]
            y_offset = 30
            for i, instruction in enumerate(instructions):
                cv2.putText(display_frame, instruction, (10, y_offset + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # ENTER or SPACE: Confirm selection
            if key in [13, 32]:  # Enter or Space
                if self.start_point is not None and self.end_point is not None:
                    bbox = self._get_bbox_from_points()
                    if bbox is not None:
                        self.cached_bbox = bbox
                        self.selection_done = True
                        cv2.destroyWindow(self.window_name)

                        print(f"✓ Chest ROI selected: x={bbox[0]:.0f}, y={bbox[1]:.0f}, "
                              f"w={bbox[2]:.0f}, h={bbox[3]:.0f}")

                        mask = self._create_mask_from_bbox(bbox, frame.shape)
                        return bbox, 1.0, mask
                else:
                    print("⚠ No selection made. Please select a region first.")

            # 'r': Reset selection
            elif key == ord('r'):
                self.start_point = None
                self.end_point = None
                self.selecting = False
                print("Selection reset. Please select again.")

            # ESC: Cancel
            elif key == 27:  # ESC
                cv2.destroyWindow(self.window_name)
                print("✗ Selection cancelled.")
                return None, 0.0, None

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start selection
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update selection while dragging
            if self.selecting:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish selection
            self.selecting = False
            self.end_point = (x, y)

    def _get_bbox_from_points(self) -> Optional[Tuple[float, float, float, float]]:
        """Convert start/end points to (x, y, w, h) bbox"""
        if self.start_point is None or self.end_point is None:
            return None

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # Ensure x1 < x2 and y1 < y2
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        # Validate minimum size
        if w < 10 or h < 10:
            print("⚠ Selection too small (min 10x10 pixels)")
            return None

        return (float(x), float(y), float(w), float(h))

    def _create_mask_from_bbox(self, bbox: Tuple[float, float, float, float],
                               frame_shape: tuple) -> np.ndarray:
        """Create binary mask covering the bbox region"""
        x, y, w, h = [int(v) for v in bbox]
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        return mask

    def reset(self):
        """Reset cached selection (useful for processing multiple videos)"""
        self.cached_bbox = None
        self.selection_done = False
        self.start_point = None
        self.end_point = None
        print("Manual selection reset.")

    def post_process(self, frame, bbox, mask):
        pass