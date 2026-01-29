"""
Create side-by-side comparison showing stabilization effect visually
By keeping both views in the SAME reference frame (first frame's hand position)
"""

import sys
sys.path.append('.')

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.detectors import get_detector
from src.segmenters import get_segmenter, convert_mask_to_frame_coords
from src.localizers import get_localizer

# Load config
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
detector = get_detector(config['detection'])
segmenter = get_segmenter(config['segmentation'])
localizer = get_localizer(config['localization'])

# Video setup
video_path = 'data/videos/bird_sample.mp4'
output_path = 'data/results/stabilization_visual_comparison.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = config['tracking'].get('start_frame', 0)
max_frames = config['tracking'].get('max_frames', 500)

# Skip to start frame
if start_frame > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

print(f"Creating visual stabilization comparison...")
print(f"Video: {video_path}")
print(f"Start frame: {start_frame}")
print(f"Max frames: {max_frames}")

# Read first frame and initialize
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    sys.exit(1)

# Detect hand and get chest ROI
hand_bbox, confidence, hand_mask = detector.detect(frame)
if hand_bbox is None:
    print("Error: Could not detect hand in first frame")
    sys.exit(1)

x, y, w, h = [int(v) for v in hand_bbox]
hand_region = frame[y:y+h, x:x+w]

# Extract hand mask in local coordinates if available
hand_mask_local = None
if hand_mask is not None:
    hand_mask_local = hand_mask[y:y+h, x:x+w]

# Segment bird
bird_mask_local = segmenter.segment(hand_region, hand_mask_local=hand_mask_local)
bird_mask_local = segmenter.get_largest_component(bird_mask_local)
bird_mask = convert_mask_to_frame_coords(bird_mask_local, hand_bbox, frame.shape)

# Locate chest
chest_roi = localizer.locate(bird_mask)
if chest_roi is None:
    print("Error: Could not locate chest in first frame")
    sys.exit(1)

print(f"Initial detection successful")

# Initialize trackers
hand_tracker = cv2.TrackerKCF_create()
chest_tracker = cv2.TrackerKCF_create()

hand_bbox_tuple = tuple(int(v) for v in hand_bbox)
chest_roi_tuple = tuple(int(v) for v in chest_roi)

hand_tracker.init(frame, hand_bbox_tuple)
chest_tracker.init(frame, chest_roi_tuple)

# Store reference hand position (first frame)
reference_hand_center = np.array([
    hand_bbox[0] + hand_bbox[2]/2,
    hand_bbox[1] + hand_bbox[3]/2
])

# Chest region size for output
chest_display_size = (250, 250)

# Output video setup (side by side + labels)
output_width = chest_display_size[0] * 2 + 30
output_height = chest_display_size[1] + 80
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (output_width, output_height))

# Reset video to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_idx = start_frame
frames_processed = 0

print("\nProcessing video...")

with tqdm(total=max_frames if max_frames else total_frames - start_frame, desc="Processing") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frames_processed >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track hand and chest
        success_hand, hand_bbox_curr = hand_tracker.update(frame)
        success_chest, chest_roi_curr = chest_tracker.update(frame)

        if not (success_hand and success_chest):
            frames_processed += 1
            frame_idx += 1
            pbar.update(1)
            continue

        # Get current hand center
        current_hand_center = np.array([
            hand_bbox_curr[0] + hand_bbox_curr[2]/2,
            hand_bbox_curr[1] + hand_bbox_curr[3]/2
        ])

        # Calculate hand motion from reference frame
        hand_motion = current_hand_center - reference_hand_center

        # Extract chest region
        cx, cy, cw, ch = [int(v) for v in chest_roi_curr]

        # Ensure we're within frame bounds
        if cy < 0 or cx < 0 or cy+ch > gray.shape[0] or cx+cw > gray.shape[1]:
            frames_processed += 1
            frame_idx += 1
            pbar.update(1)
            continue

        chest_region_raw = gray[cy:cy+ch, cx:cx+cw]

        # LEFT: Chest as-is (moves with hand)
        chest_no_stab = chest_region_raw.copy()

        # RIGHT: Chest with stabilization (compensate for hand motion)
        # Shift chest region to undo hand motion
        M = np.float32([[1, 0, -hand_motion[0]],
                       [0, 1, -hand_motion[1]]])

        chest_with_stab = cv2.warpAffine(
            chest_region_raw, M,
            (chest_region_raw.shape[1], chest_region_raw.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128
        )

        # Resize both to display size
        if chest_no_stab.size > 0:
            chest_display_no_stab = cv2.resize(chest_no_stab, chest_display_size)
        else:
            chest_display_no_stab = np.zeros(chest_display_size, dtype=np.uint8)

        if chest_with_stab.size > 0:
            chest_display_with_stab = cv2.resize(chest_with_stab, chest_display_size)
        else:
            chest_display_with_stab = np.zeros(chest_display_size, dtype=np.uint8)

        # Convert to BGR for colored text
        chest_display_no_stab = cv2.cvtColor(chest_display_no_stab, cv2.COLOR_GRAY2BGR)
        chest_display_with_stab = cv2.cvtColor(chest_display_with_stab, cv2.COLOR_GRAY2BGR)

        # Create output frame (white background)
        output_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        # Add title
        cv2.putText(output_frame, "Chest Region Comparison", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add labels
        cv2.putText(output_frame, "Raw (moves with hand)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(output_frame, "Stabilized", (chest_display_size[0] + 30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add frame number and hand motion
        cv2.putText(output_frame, f"Frame: {frame_idx}", (10, output_height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(output_frame, f"Hand motion: ({hand_motion[0]:.1f}, {hand_motion[1]:.1f})",
                   (10, output_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Place chest regions
        output_frame[70:70+chest_display_size[1],
                    10:10+chest_display_size[0]] = chest_display_no_stab
        output_frame[70:70+chest_display_size[1],
                    chest_display_size[0]+20:chest_display_size[0]+20+chest_display_size[0]] = chest_display_with_stab

        # Draw border around regions
        cv2.rectangle(output_frame, (10, 70),
                     (10+chest_display_size[0], 70+chest_display_size[1]),
                     (0, 0, 255), 2)
        cv2.rectangle(output_frame, (chest_display_size[0]+20, 70),
                     (chest_display_size[0]+20+chest_display_size[0], 70+chest_display_size[1]),
                     (0, 255, 0), 2)

        out.write(output_frame)

        frame_idx += 1
        frames_processed += 1
        pbar.update(1)

cap.release()
out.release()

print(f"\nComparison video saved to: {output_path}")
print("The RIGHT side (green) should appear more stable than the LEFT (red)!")
print("Look for: the bird's chest staying in the same position on the right,")
print("          while it moves around on the left due to hand shake.")
