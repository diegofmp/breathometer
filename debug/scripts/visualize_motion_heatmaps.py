"""
Visualize motion detection heatmaps for all motion-based localizer methods
Shows how each method detects movement across the bird mask
"""

import sys
sys.path.append('.')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import LinearSegmentedColormap

from src.detectors import get_detector
from src.segmenters import get_segmenter, convert_mask_to_frame_coords
from src.localizers import get_localizer

# Configuration
video_path = 'data/videos/bird_sample.mp4'
config_path = 'configs/default.yaml'
frame_number = 40  # Which frame to visualize (adjust as needed)

# Motion-based methods to compare
motion_methods = ['variance', 'motion', 'optical_flow']

print("="*70)
print("MOTION DETECTION HEATMAP VISUALIZATION")
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

# Step 1: Hand Detection
print("Step 1: Detecting hand...")
detector = get_detector(config['detection'])
hand_bbox, confidence, hand_mask = detector.detect(frame)

if hand_bbox is None:
    print("Error: No hand detected in frame")
    sys.exit(1)

print(f"✓ Hand detected with confidence: {confidence:.2f}")

# Step 2: Bird Segmentation
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

print(f"✓ Bird segmented: {np.sum(bird_mask > 0)} pixels")

# Get bird bounding box
contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bird_contour = max(contours, key=cv2.contourArea)
bx, by, bw, bh = cv2.boundingRect(bird_contour)

print(f"  Bird bbox: x={bx}, y={by}, w={bw}, h={bh}")

# Step 3: Generate motion heatmaps for each method
print("\nStep 3: Generating motion heatmaps...")
print("  (Building frame buffers for motion analysis...)")

heatmaps = {}
chest_rois = {}

for method in motion_methods:
    print(f"\n  Processing {method.upper()} method...")

    # Get localizer for this method
    loc_config = config['localization'].copy()
    loc_config['method'] = method
    localizer = get_localizer(loc_config)

    # Determine buffer requirements
    buffer_needs = {
        'variance': 30,
        'motion': 60,
        'optical_flow': 60
    }
    frames_needed = buffer_needs.get(method, 60)

    # Re-open video to process frames for buffer
    cap_buffer = cv2.VideoCapture(video_path)
    cap_buffer.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - frames_needed))

    # Feed frames to localizer to build buffer
    chest_roi = None
    for i in range(frames_needed + 1):
        ret, frame_buffer = cap_buffer.read()
        if not ret:
            break

        chest_roi = localizer.locate(bird_mask, frame=frame_buffer, fps=fps)

    cap_buffer.release()

    if chest_roi is not None:
        chest_rois[method] = chest_roi
        print(f"    ✓ Chest ROI located")

    # Now extract the motion heatmap based on method
    # We need to recreate the analysis that each localizer does

    if method == 'variance':
        # Variance map: temporal variance of pixels
        if len(localizer.frame_buffer) >= localizer.buffer_size:
            buffer_array = np.array(localizer.frame_buffer, dtype=np.float32)
            variance_map = np.var(buffer_array, axis=0)

            # Create full-frame heatmap
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            heatmap[by:by+bh, bx:bx+bw] = variance_map

            # Normalize for visualization
            if variance_map.max() > 0:
                heatmap = (heatmap / variance_map.max()) * 255

            heatmaps[method] = heatmap.astype(np.uint8)
            print(f"    ✓ Variance heatmap created (max variance: {variance_map.max():.2f})")

    elif method == 'motion':
        # Optical flow magnitude averaged over buffer
        if len(localizer.frame_buffer) >= 2:
            flow_magnitudes = []
            for i in range(len(localizer.frame_buffer) - 1):
                prev_bird = localizer.frame_buffer[i]
                curr_bird = localizer.frame_buffer[i + 1]

                flow = cv2.calcOpticalFlowFarneback(
                    prev_bird, curr_bird, None,
                    pyr_scale=0.5, levels=2, winsize=10,
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                )

                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitudes.append(magnitude)

            bird_flow = np.mean(flow_magnitudes, axis=0)

            # Create full-frame heatmap
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            heatmap[by:by+bh, bx:bx+bw] = bird_flow

            # Normalize for visualization
            if bird_flow.max() > 0:
                heatmap = (heatmap / bird_flow.max()) * 255

            heatmaps[method] = heatmap.astype(np.uint8)
            print(f"    ✓ Motion heatmap created (max flow: {bird_flow.max():.2f})")

    elif method == 'optical_flow':
        # Frequency power map: FFT power at breathing frequency
        if len(localizer.frame_buffer) >= localizer.buffer_frames:
            # Divide bird region into grid and compute FFT power per cell
            cell_h = bh // localizer.grid_size
            cell_w = bw // localizer.grid_size

            power_map = np.zeros((localizer.grid_size, localizer.grid_size), dtype=np.float32)

            for gy in range(localizer.grid_size):
                for gx in range(localizer.grid_size):
                    cell_y_start = gy * cell_h
                    cell_x_start = gx * cell_w
                    cell_y_end = min((gy + 1) * cell_h, bh)
                    cell_x_end = min((gx + 1) * cell_w, bw)

                    # Extract motion time series for this cell
                    motion_series = []
                    for i in range(len(localizer.frame_buffer) - 1):
                        prev = localizer.frame_buffer[i][cell_y_start:cell_y_end, cell_x_start:cell_x_end]
                        curr = localizer.frame_buffer[i + 1][cell_y_start:cell_y_end, cell_x_start:cell_x_end]

                        if prev.size > 0 and curr.size > 0:
                            diff = np.abs(curr.astype(float) - prev.astype(float))
                            motion_series.append(np.mean(diff))
                        else:
                            motion_series.append(0)

                    if len(motion_series) >= 10:
                        # FFT analysis
                        fft_result = np.fft.fft(motion_series)
                        frequencies = np.fft.fftfreq(len(motion_series), d=1.0/fps)

                        # Sum power in breathing frequency range
                        mask = (frequencies >= localizer.freq_range[0]) & (frequencies <= localizer.freq_range[1])
                        power = np.sum(np.abs(fft_result[mask]))
                        power_map[gy, gx] = power

            # Resize power map to bird size for visualization
            power_map_resized = cv2.resize(power_map, (bw, bh), interpolation=cv2.INTER_LINEAR)

            # Create full-frame heatmap
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            heatmap[by:by+bh, bx:bx+bw] = power_map_resized

            # Normalize for visualization
            if power_map_resized.max() > 0:
                heatmap = (heatmap / power_map_resized.max()) * 255

            heatmaps[method] = heatmap.astype(np.uint8)
            print(f"    ✓ Optical flow frequency heatmap created (max power: {power_map_resized.max():.2f})")

if not heatmaps:
    print("\nError: No heatmaps generated")
    sys.exit(1)

print(f"\n✓ Successfully generated {len(heatmaps)} motion heatmaps")

# Visualization
print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

# Create custom colormap: white -> yellow -> orange -> red
colors = ['white', 'yellow', 'orange', 'red', 'darkred']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('motion', colors, N=n_bins)

n_methods = len(heatmaps)
fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 12))

if n_methods == 1:
    axes = axes.reshape(2, 1)

for idx, (method, heatmap) in enumerate(heatmaps.items()):
    # Top row: Original frame with bird mask overlay
    vis_original = frame.copy()
    overlay = vis_original.copy()
    overlay[bird_mask > 0] = [0, 255, 0]  # Green bird mask
    vis_original = cv2.addWeighted(vis_original, 0.7, overlay, 0.3, 0)

    # Draw chest ROI if available
    if method in chest_rois:
        cx, cy, cw, ch = [int(v) for v in chest_rois[method]]
        cv2.rectangle(vis_original, (cx, cy), (cx+cw, cy+ch), (0, 255, 255), 2)

    axes[0, idx].imshow(cv2.cvtColor(vis_original, cv2.COLOR_BGR2RGB))
    axes[0, idx].set_title(f'{method.upper()}\nOriginal + Bird Mask + Chest ROI',
                           fontsize=12, fontweight='bold')
    axes[0, idx].axis('off')

    # Bottom row: Motion heatmap
    # Apply mask to show only bird region
    masked_heatmap = heatmap.copy()
    masked_heatmap[bird_mask == 0] = 0

    im = axes[1, idx].imshow(masked_heatmap, cmap=cmap, vmin=0, vmax=255)
    axes[1, idx].set_title(f'{method.upper()}\nMotion Heatmap',
                           fontsize=12, fontweight='bold')
    axes[1, idx].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
    cbar.set_label('Motion Intensity', rotation=270, labelpad=20)

# Add description
method_descriptions = {
    'variance': 'Temporal variance of pixel intensity\n(Higher variance = more movement)',
    'motion': 'Optical flow magnitude averaged over time\n(Higher flow = more motion)',
    'optical_flow': 'FFT power at breathing frequency (0.5-4.0 Hz)\n(Higher power = periodic breathing motion)'
}

desc_text = "Motion Detection Methods:\n\n"
for method in heatmaps.keys():
    desc_text += f"• {method.upper()}: {method_descriptions.get(method, '')}\n\n"

fig.text(0.5, 0.02, desc_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Motion Heatmap Comparison - Frame {frame_number}',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.08, 1, 0.97])

# Save figure
output_path = 'data/results/motion_heatmaps_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print statistics
print("\n" + "="*70)
print("MOTION STATISTICS")
print("="*70)
print(f"{'Method':<15} {'Max Value':<12} {'Mean (bird)':<12} {'Std (bird)':<12}")
print("-"*70)

for method, heatmap in heatmaps.items():
    # Stats only on bird region
    bird_region_values = heatmap[bird_mask > 0]
    max_val = bird_region_values.max()
    mean_val = bird_region_values.mean()
    std_val = bird_region_values.std()

    print(f"{method:<15} {max_val:<12.2f} {mean_val:<12.2f} {std_val:<12.2f}")

print("="*70)
print("\nInterpretation:")
print("  - WHITE areas: No/minimal motion detected")
print("  - YELLOW areas: Low motion")
print("  - ORANGE areas: Moderate motion")
print("  - RED areas: High motion")
print("  - DARK RED areas: Very high motion (breathing hotspot)")
print("\n✓ Done! View the plot window or check the saved image.")
plt.show()
