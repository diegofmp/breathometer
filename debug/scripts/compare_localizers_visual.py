"""
Visual comparison of all chest localization methods on a single frame
Shows: Hand detection, Bird segmentation, and Chest ROI for each method
"""

import sys
sys.path.append('.')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.detectors import get_detector
from src.segmenters import get_segmenter, convert_mask_to_frame_coords
from src.localizers import get_localizer

# Configuration
video_path = 'data/videos/bird_sample.mp4'
config_path = 'configs/default.yaml'
frame_number = 40  # Which frame to visualize (adjust as needed)

# All localization methods to compare
localization_methods = [
    'simple',
    'contour',
    'variance',
    'motion',
    'optical_flow'
]

print("="*70)
print("VISUAL COMPARISON: CHEST LOCALIZATION METHODS")
print("="*70)

# Load configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load video and extract frame
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not read frame {frame_number} from video")
    sys.exit(1)

print(f"Analyzing frame {frame_number} (FPS: {fps:.1f})")
print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
print()

# Step 1: Hand Detection (same for all methods)
print("Step 1: Detecting hand...")
detector = get_detector(config['detection'])
hand_bbox, confidence, hand_mask = detector.detect(frame)

if hand_bbox is None:
    print("Error: No hand detected in frame")
    sys.exit(1)

print(f"✓ Hand detected with confidence: {confidence:.2f}")
print(f"  Hand bbox: {hand_bbox}")

# Step 2: Bird Segmentation (same for all methods)
print("\nStep 2: Segmenting bird...")
segmenter = get_segmenter(config['segmentation'])

x, y, w, h = [int(v) for v in hand_bbox]
hand_region = frame[y:y+h, x:x+w]

# Extract hand mask in local coordinates if available
hand_mask_local = None
if hand_mask is not None:
    hand_mask_local = hand_mask[y:y+h, x:x+w]

bird_mask_local = segmenter.segment(hand_region, hand_mask_local=hand_mask_local)
bird_mask_local = segmenter.get_largest_component(bird_mask_local)
bird_mask = convert_mask_to_frame_coords(bird_mask_local, hand_bbox, frame.shape)

bird_pixels = np.sum(bird_mask > 0)
print(f"✓ Bird segmented: {bird_pixels} pixels")

# Step 3: Chest Localization (different for each method)
print("\nStep 3: Localizing chest with different methods...")
print("  (Building frame buffers for advanced methods...)")
chest_rois = {}

for method in localization_methods:
    print(f"  Testing {method}...")

    # Get localizer for this method
    loc_config = config['localization'].copy()
    loc_config['method'] = method
    localizer = get_localizer(loc_config)

    # Determine buffer requirements
    buffer_needs = {
        'simple': 1,
        'contour': 1,
        'variance': 30,
        'motion': 60,  # Reduced from 60 for faster testing
        'optical_flow': 60
    }
    frames_needed = buffer_needs.get(method, 1)

    # Re-open video to process frames for buffer
    cap_buffer = cv2.VideoCapture(video_path)
    cap_buffer.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - frames_needed))

    # Feed frames to localizer to build buffer
    chest_roi = None
    for i in range(frames_needed + 1):
        ret, frame_buffer = cap_buffer.read()
        if not ret:
            break

        # Use the same bird_mask for all frames (simplification)
        # In real usage, bird_mask would be re-segmented per frame
        chest_roi = localizer.locate(bird_mask, frame=frame_buffer, fps=fps)

    cap_buffer.release()

    if chest_roi is not None:
        chest_rois[method] = chest_roi
        cx, cy, cw, ch = [int(v) for v in chest_roi]
        print(f"    ✓ Chest ROI: x={cx}, y={cy}, w={cw}, h={ch} (after {frames_needed} frames)")
    else:
        print(f"    ✗ Failed to locate chest")

if not chest_rois:
    print("\nError: No methods successfully located the chest")
    sys.exit(1)

print(f"\n✓ Successfully located chest with {len(chest_rois)}/{len(localization_methods)} methods")

# Visualization
print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

n_methods = len(chest_rois)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Define colors for each method (BGR format for OpenCV)
method_colors = {
    'simple': (255, 0, 0),        # Blue
    'contour': (0, 255, 0),       # Green
    'variance': (0, 0, 255),      # Red
    'motion': (255, 255, 0),      # Cyan
    'optical_flow': (255, 0, 255) # Magenta
}

# Panel 0: Original frame with hand detection
vis0 = frame.copy()
cv2.rectangle(vis0, (x, y), (x+w, y+h), (255, 0, 0), 3)
cv2.putText(vis0, f"Hand (conf: {confidence:.2f})", (x, y-10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
axes[0].imshow(cv2.cvtColor(vis0, cv2.COLOR_BGR2RGB))
axes[0].set_title('1. Hand Detection', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Panel 1: Bird segmentation
vis1 = frame.copy()
overlay = vis1.copy()
overlay[bird_mask > 0] = [0, 255, 0]
vis1 = cv2.addWeighted(vis1, 0.6, overlay, 0.4, 0)
axes[1].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
axes[1].set_title('2. Bird Segmentation', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Panel 2: All chest ROIs overlaid
vis2 = frame.copy()
# Draw bird mask first
overlay = vis2.copy()
overlay[bird_mask > 0] = [0, 255, 0]
vis2 = cv2.addWeighted(vis2, 0.7, overlay, 0.3, 0)

# Draw all chest ROIs
for method, chest_roi in chest_rois.items():
    cx, cy, cw, ch = [int(v) for v in chest_roi]
    color = method_colors.get(method, (255, 255, 255))
    cv2.rectangle(vis2, (cx, cy), (cx+cw, cy+ch), color, 2)

    # Add method label inside the box
    label_y = cy + 20
    cv2.putText(vis2, method.upper(), (cx + 5, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

axes[2].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
axes[2].set_title('3. All Methods Overlaid', fontsize=14, fontweight='bold')
axes[2].axis('off')

# Add legend for colors
legend_text = "Legend:\n"
for method, color in method_colors.items():
    if method in chest_rois:
        # Convert BGR to RGB for display
        rgb_color = (color[2]/255, color[1]/255, color[0]/255)
        legend_text += f"■ {method.upper()}\n"
axes[2].text(1.02, 0.5, legend_text, transform=axes[2].transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panels 3-7: Individual methods
for idx, (method, chest_roi) in enumerate(chest_rois.items(), start=3):
    if idx >= len(axes):
        break

    vis = frame.copy()

    # Draw bird mask
    overlay = vis.copy()
    overlay[bird_mask > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # Draw chest ROI
    cx, cy, cw, ch = [int(v) for v in chest_roi]
    color = method_colors.get(method, (255, 255, 255))
    cv2.rectangle(vis, (cx, cy), (cx+cw, cy+ch), color, 3)

    # Add label
    cv2.putText(vis, f"{method.upper()}", (cx, cy-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add ROI dimensions
    dim_text = f"{cw}x{ch}px"
    cv2.putText(vis, dim_text, (cx, cy+ch+20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    axes[idx].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f'{method.upper()} Method', fontsize=12, fontweight='bold')
    axes[idx].axis('off')

# Hide unused panels
for idx in range(3 + len(chest_rois), len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Chest Localization Comparison - Frame {frame_number}',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save figure
output_path = 'data/results/localizers_visual_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print comparison table
print("\n" + "="*70)
print("CHEST ROI COMPARISON TABLE")
print("="*70)
print(f"{'Method':<15} {'X':<8} {'Y':<8} {'Width':<8} {'Height':<8} {'Area':<10}")
print("-"*70)

for method, chest_roi in chest_rois.items():
    cx, cy, cw, ch = [int(v) for v in chest_roi]
    area = cw * ch
    print(f"{method:<15} {cx:<8} {cy:<8} {cw:<8} {ch:<8} {area:<10}")

print("="*70)

# Calculate ROI overlap statistics
print("\n" + "="*70)
print("ROI POSITION STATISTICS")
print("="*70)

# Extract coordinates
xs = [int(roi[0]) for roi in chest_rois.values()]
ys = [int(roi[1]) for roi in chest_rois.values()]
ws = [int(roi[2]) for roi in chest_rois.values()]
hs = [int(roi[3]) for roi in chest_rois.values()]

print(f"X position: mean={np.mean(xs):.1f}, std={np.std(xs):.1f}, range=[{min(xs)}, {max(xs)}]")
print(f"Y position: mean={np.mean(ys):.1f}, std={np.std(ys):.1f}, range=[{min(ys)}, {max(ys)}]")
print(f"Width:      mean={np.mean(ws):.1f}, std={np.std(ws):.1f}, range=[{min(ws)}, {max(ws)}]")
print(f"Height:     mean={np.mean(hs):.1f}, std={np.std(hs):.1f}, range=[{min(hs)}, {max(hs)}]")

# Calculate center positions
centers_x = [x + w/2 for x, w in zip(xs, ws)]
centers_y = [y + h/2 for y, h in zip(ys, hs)]

print(f"\nCenter X:   mean={np.mean(centers_x):.1f}, std={np.std(centers_x):.1f}")
print(f"Center Y:   mean={np.mean(centers_y):.1f}, std={np.std(centers_y):.1f}")

# Interpretation
center_std_avg = (np.std(centers_x) + np.std(centers_y)) / 2
if center_std_avg < 10:
    consistency = "VERY HIGH - All methods agree closely"
elif center_std_avg < 20:
    consistency = "HIGH - Methods show good agreement"
elif center_std_avg < 40:
    consistency = "MODERATE - Some variation between methods"
else:
    consistency = "LOW - Significant differences between methods"

print(f"\nConsistency: {consistency}")
print("="*70)

print("\n✓ Done! View the plot window or check the saved image.")
plt.show()
