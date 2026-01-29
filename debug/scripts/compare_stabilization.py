"""
Create side-by-side comparison video of chest region with/without stabilization
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
from src.measurements import get_measurement

# Load config
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
detector = get_detector(config['detection'])
segmenter = get_segmenter(config['segmentation'])
localizer = get_localizer(config['localization'])
measurement = get_measurement(config['measurement'])

# Video setup
video_path = 'data/videos/bird_sample.mp4'
output_path = 'data/results/stabilization_comparison.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = config['tracking'].get('start_frame', 0)

# Skip to start frame
if start_frame > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

print(f"Creating stabilization comparison video...")
print(f"Video: {video_path}")
print(f"Start frame: {start_frame}")
print(f"FPS: {fps}")

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
print(f"Hand bbox: {hand_bbox}")
print(f"Chest ROI: {chest_roi}")

# Initialize trackers
hand_tracker_no_stab = cv2.TrackerKCF_create()
chest_tracker_no_stab = cv2.TrackerKCF_create()
hand_tracker_with_stab = cv2.TrackerKCF_create()
chest_tracker_with_stab = cv2.TrackerKCF_create()

hand_bbox_tuple = tuple(int(v) for v in hand_bbox)
chest_roi_tuple = tuple(int(v) for v in chest_roi)

hand_tracker_no_stab.init(frame, hand_bbox_tuple)
chest_tracker_no_stab.init(frame, chest_roi_tuple)
hand_tracker_with_stab.init(frame, hand_bbox_tuple)
chest_tracker_with_stab.init(frame, chest_roi_tuple)

# Chest region size for output
chest_display_size = (200, 200)

# Output video setup (side by side)
output_width = chest_display_size[0] * 2 + 20  # 2 chest regions + padding
output_height = chest_display_size[1] + 60  # Extra space for labels
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (output_width, output_height))

prev_hand_bbox_no_stab = hand_bbox
prev_hand_bbox_with_stab = hand_bbox
frame_idx = start_frame

# Reset video to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

with tqdm(total=total_frames - start_frame, desc="Processing") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track both hand and chest
        success_hand_no_stab, hand_bbox_no_stab = hand_tracker_no_stab.update(frame)
        success_chest_no_stab, chest_roi_no_stab = chest_tracker_no_stab.update(frame)
        success_hand_stab, hand_bbox_with_stab = hand_tracker_with_stab.update(frame)
        success_chest_stab, chest_roi_with_stab = chest_tracker_with_stab.update(frame)

        # WITHOUT STABILIZATION - just extract the chest region
        if success_chest_no_stab:
            cx, cy, cw, ch = [int(v) for v in chest_roi_no_stab]
            chest_region_no_stab = gray[cy:cy+ch, cx:cx+cw]
        else:
            chest_region_no_stab = np.zeros(chest_display_size, dtype=np.uint8)

        # WITH STABILIZATION - apply hand motion compensation
        if success_hand_stab and success_chest_stab:
            cx, cy, cw, ch = [int(v) for v in chest_roi_with_stab]
            chest_region_raw = gray[cy:cy+ch, cx:cx+cw]

            # Apply stabilization (compensate for hand motion)
            if prev_hand_bbox_with_stab is not None:
                prev_center = np.array([
                    prev_hand_bbox_with_stab[0] + prev_hand_bbox_with_stab[2]/2,
                    prev_hand_bbox_with_stab[1] + prev_hand_bbox_with_stab[3]/2
                ])
                curr_center = np.array([
                    hand_bbox_with_stab[0] + hand_bbox_with_stab[2]/2,
                    hand_bbox_with_stab[1] + hand_bbox_with_stab[3]/2
                ])

                hand_motion = curr_center - prev_center

                # Create translation matrix to compensate for hand motion
                M = np.float32([[1, 0, -hand_motion[0]],
                               [0, 1, -hand_motion[1]]])

                # Apply translation to stabilize
                chest_region_with_stab = cv2.warpAffine(
                    chest_region_raw, M, (chest_region_raw.shape[1], chest_region_raw.shape[0])
                )
            else:
                chest_region_with_stab = chest_region_raw

            prev_hand_bbox_with_stab = hand_bbox_with_stab
        else:
            chest_region_with_stab = np.zeros(chest_display_size, dtype=np.uint8)

        # Resize both chest regions to same size
        if chest_region_no_stab.size > 0:
            chest_display_no_stab = cv2.resize(chest_region_no_stab, chest_display_size)
        else:
            chest_display_no_stab = np.zeros(chest_display_size, dtype=np.uint8)

        if chest_region_with_stab.size > 0:
            chest_display_with_stab = cv2.resize(chest_region_with_stab, chest_display_size)
        else:
            chest_display_with_stab = np.zeros(chest_display_size, dtype=np.uint8)

        # Convert to BGR for colored text
        chest_display_no_stab = cv2.cvtColor(chest_display_no_stab, cv2.COLOR_GRAY2BGR)
        chest_display_with_stab = cv2.cvtColor(chest_display_with_stab, cv2.COLOR_GRAY2BGR)

        # Create output frame
        output_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        # Add labels
        cv2.putText(output_frame, "WITHOUT Stabilization", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_frame, "WITH Stabilization", (chest_display_size[0] + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add frame number
        cv2.putText(output_frame, f"Frame: {frame_idx}", (10, output_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Place chest regions
        output_frame[40:40+chest_display_size[1], 10:10+chest_display_size[0]] = chest_display_no_stab
        output_frame[40:40+chest_display_size[1],
                    chest_display_size[0]+20:chest_display_size[0]+20+chest_display_size[0]] = chest_display_with_stab

        out.write(output_frame)

        frame_idx += 1
        pbar.update(1)

cap.release()
out.release()

print(f"\nComparison video saved to: {output_path}")
print("You can now see the difference between stabilized and non-stabilized chest regions!")
