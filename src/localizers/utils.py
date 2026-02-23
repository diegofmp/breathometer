import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_breathing_energy_map_BIRDMASK(buffer_list, hand_mask=None, bird_mask=None, visualize=False):
    """
    Compute a breathing energy map that prioritizes stable periodic motion.

    This function uses optical flow analysis to detect breathing motion while
    suppressing erratic movements (e.g., tail wags). It favors "steady glow"
    patterns - regions with consistent periodic expansion rather than sporadic bursts.

    The bird_mask restricts energy to the bird body only. It is applied *after*
    per-frame Jacobian computation but *before* temporal aggregation (mean/std),
    so that statistics are computed only over bird pixels. Applying it earlier
    (before optical flow / Sobel) would cause border artifacts from the mask edge.

    Parameters
    ----------
    buffer_list : list of np.ndarray
        List of grayscale frames (already preprocessed/smoothed)
    hand_mask : np.ndarray, optional
        Binary mask of the hand region to suppress (CROPPED)
    bird_mask : np.ndarray, optional
        Binary mask of the bird body (CROPPED).
        Energy outside this mask is zeroed out.

    Returns
    -------
    energy_final : np.ndarray
        Cleaned breathing energy map (float32, range 0-1)
    divergence : np.ndarray
        (Mean) Raw divergence

    Notes
    -----
    The algorithm:
    1. Uses last 8 frames (~200ms window for 310 BPM breathing)
    2. Computes optical flow between consecutive frames
    3. Calculates Jacobian determinant (area change rate)
    4. Masks energy to bird body (suppresses background/hand noise)
    5. Filters by temporal stability (mean/std ratio)
    6. Removes speckles via morphological opening
    """
    # Use last 8 frames (~200ms window, enough for one high-speed breath cycle)
    stack = np.array(buffer_list[-30:]) # TODO: move to param!!!!
    t, h, w = stack.shape

    # Pre-build bird mask (float binary, frame-sized).
    # Applied after Jacobian computation so Sobel gradients are not corrupted
    # by hard mask edges. Zeroing outside the bird body here means all temporal
    # statistics (mean/std) are computed only over bird pixels.
    if bird_mask is not None:
        bird_binary = (cv2.resize(bird_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    else:
        bird_binary = np.ones((h, w), dtype=np.float32)

    step_energies = []
    step_divergence = []
    stride = 1

    # ========================================================================
    # STEP 1: Optical Flow Analysis
    # ========================================================================

    for i in range(0, t - stride, stride):
        prev = stack[i]
        curr = stack[i + stride]

        # Compute dense optical flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5,      # Image pyramid scale
            levels=3,           # Pyramid levels
            winsize=15,         # Averaging window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Polynomial expansion neighborhood
            poly_sigma=1.2,     # Gaussian std for polynomial expansion
            flags=0
        )

        # Center flow components (remove global motion bias)
        u_x_res = flow[..., 0] - np.median(flow[..., 0])
        u_y_res = flow[..., 1] - np.median(flow[..., 1])

        # Compute spatial gradients of flow (Jacobian matrix components)
        du_x_dx = cv2.Sobel(u_x_res, cv2.CV_32F, 1, 0, ksize=5)  # ∂u_x/∂x
        du_y_dy = cv2.Sobel(u_y_res, cv2.CV_32F, 0, 1, ksize=5)  # ∂u_y/∂y
        du_x_dy = cv2.Sobel(u_x_res, cv2.CV_32F, 0, 1, ksize=5)  # ∂u_x/∂y
        du_y_dx = cv2.Sobel(u_y_res, cv2.CV_32F, 1, 0, ksize=5)  # ∂u_y/∂x

        # Divergence: magnitude of expansion/contraction
        divergence = np.abs(du_x_dx + du_y_dy)
        step_divergence.append(divergence)

        # Jacobian determinant: rate of area change (breathing signature)
        # det(J) = (∂u_x/∂x)(∂u_y/∂y) - (∂u_x/∂y)(∂u_y/∂x)
        det_J = (du_x_dx * du_y_dy) - (du_x_dy * du_y_dx)

        # Convert to positive energy (breathing expansion)
        frame_energy = np.sqrt(np.maximum(0, det_J))


        # Restrict energy to bird body BEFORE stacking for temporal stats.
        # Background and hand motion are zeroed here so they don't pollute the
        # stability ratio computed in Step 2.
        frame_energy = frame_energy * bird_binary

        step_energies.append(frame_energy)

    # ========================================================================
    # STEP 2: Temporal Stability Filtering (Steady Glow)
    # ========================================================================

    energy_stack = np.array(step_energies)  # Shape: (steps, H, W)
    
    # Mean energy: average brightness over time
    mean_energy = np.mean(energy_stack, axis=0)
    
    # Standard deviation: temporal variability
    # High std = erratic motion (tail wags), Low std = periodic motion (breathing)
    std_energy = np.std(energy_stack, axis=0) + 1e-6  # Add epsilon for numerical stability

    # Stability score: favor steady periodic motion
    # Use power of 1.5 to suppress erratic movements while preserving moderate stability
    ratio = mean_energy / std_energy

    # Gentler exponent (1.5 instead of 2) to avoid over-suppression
    # ratio=0.5 → stability=0.35 (vs 0.25 with power 2)
    # ratio=2.0 → stability=2.83 (vs 4.0 with power 2)
    # ratio=10  → stability=31.6 (vs 100 with power 2)
    stability_map = np.power(ratio, 1.5)

    # Option A: Add a minimum floor to prevent complete suppression
    # stability_floor = 0.1
    # stability_map = np.maximum(stability_map, stability_floor)

    # Option B: Normalize to 0-1 range with percentile clipping
    # This prevents extreme outliers from dominating
    stability_99th = np.percentile(stability_map, 99)
    stability_map_clipped = np.clip(stability_map, 0, stability_99th)
    stability_map = stability_map_clipped / stability_99th if stability_99th > 0 else stability_map_clipped

    # Scale to more useful range (0.1 to 1.0) to preserve signal after morphology
    stability_map = 0.1 + 0.9 * stability_map

    # Combine: dim regions with erratic motion
    combined_energy = mean_energy * stability_map

    # ========================================================================
    # STEP 3: Morphological Cleaning
    # ========================================================================
    
    # Normalize to 0-255 for morphological operations
    energy_for_morph = cv2.normalize(combined_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Remove small isolated speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    energy_clean = cv2.morphologyEx(energy_for_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # ========================================================================
    # STEP 4: Final Refinement
    # ========================================================================
    
    # Convert back to normalized float and smooth
    energy_final = energy_clean.astype(np.float32) / 255.0
    energy_final = cv2.GaussianBlur(energy_final, (15, 15), 0)

    # Hard gate: zero out anything outside the bird body.
    # The Gaussian blur above can spread energy slightly beyond the mask edge,
    # so we re-apply it here to guarantee no leakage into background.
    energy_final = energy_final * bird_binary

    if visualize:
        # For better visualization of high dynamic range, use log scale for stability
        stability_map_log = np.log1p(stability_map)  # log(1 + x) to handle zeros

        # Visualize processing stages
        plot_matrices([
            (bird_binary, "Bird Mask"),
            (mean_energy, "Mean Energy"),
            (std_energy, "Std Deviation"),
            (ratio, "Ratio (mean/std)"),
            (stability_map_log, f"Stability Map (log scale)\nmax={np.max(stability_map):.0f}"),
            (combined_energy, "Combined Energy"),
            (energy_clean, "Morphological Clean"),
            (energy_final, "Final Energy (bird-masked)"),
        ])

    raw_divergence_avg = np.mean(np.array(step_divergence), axis=0)


    return energy_final, raw_divergence_avg


def get_breathing_energy_map(buffer_list, hand_mask, bird_mask=None, visualize=False):

    # DILATE HAND MASK TO CLEAR SMALL HAND MOVEMENTS
    # Constants for mask expansion
    DILATION_KERNEL_SIZE = (15, 15)
    DILATION_ITERATIONS = 3
    MOMENT_THRESHOLD = 1e-5

    # Step 1: Expand hand mask to create buffer zone
    kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
    expanded_mask = cv2.dilate(
        hand_mask.astype(np.uint8),
        kernel,
        iterations=DILATION_ITERATIONS
    )

    last_frame = buffer_list[-1]

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(last_frame, cmap="gray")
        if bird_mask is not None:
            bird_overlay = np.zeros((*last_frame.shape[:2], 4), dtype=np.float32)
            bird_overlay[bird_mask > 0] = [0, 1, 0, 0.4]   # green = bird body
            ax.imshow(bird_overlay)
        hand_overlay = np.zeros((*last_frame.shape[:2], 4), dtype=np.float32)
        hand_overlay[hand_mask > 0] = [1, 0, 0, 0.5]        # red = hand
        hand_overlay[expanded_mask > 0] = [1, 0.5, 0, 0.25] # orange = expanded hand zone
        ax.imshow(hand_overlay)
        ax.set_title("Last frame | green=bird  red=hand  orange=expanded hand zone")
        #ax.axis("off")
        plt.tight_layout()
        plt.show()

    return get_breathing_energy_map_BIRDMASK(buffer_list, hand_mask, bird_mask, visualize=visualize)


def clip_to_mask_smart(mask, energy_map, l, r, t, b, bx, by, mode='energy', min_pct=0.75):
    """
    Finds the optimal sub-rectangle within [l, r, t, b] that avoids mask > 0.
    'area' mode: Uses Global Maximum Rectangle (Histogram algorithm).
    'energy' mode: Uses Greedy Shrink based on energy_map sums.
    """
    roi_region = mask[t:b+1, l:r+1]
    
    # If the whole area is already clean, return it
    if np.all(roi_region == 0):
        return l, r, t, b

    if mode == 'area':
        # --- GLOBAL MAXIMUM RECTANGLE (Histogram Algorithm) ---
        rows, cols = roi_region.shape
        heights = np.zeros(cols, dtype=int)
        max_area = -1
        best_coords = (0, cols - 1, 0, rows - 1) # local: l, r, t, b

        for row_idx in range(rows):
            # Update histogram heights: if pixel is 0, height increases; else resets to 0
            for col_idx in range(cols):
                if roi_region[row_idx, col_idx] == 0:
                    heights[col_idx] += 1
                else:
                    heights[col_idx] = 0
            
            # Find largest rectangle in the current histogram
            # Standard O(n) stack-based approach
            stack = []
            for i, h in enumerate(np.append(heights, 0)): # Append 0 to flush stack
                while stack and heights[stack[-1]] >= h:
                    H = heights[stack.pop()]
                    W = i if not stack else i - stack[-1] - 1
                    current_area = H * W
                    if current_area > max_area:
                        max_area = current_area
                        # Calculate local coordinates
                        # width ends at i-1, height ends at row_idx
                        rect_r = i - 1
                        rect_l = rect_r - W + 1
                        rect_b = row_idx
                        rect_t = rect_b - H + 1
                        best_coords = (rect_l, rect_r, rect_t, rect_b)
                stack.append(i)

        cl, cr, ct, cb = best_coords
        return l + cl, l + cr, t + ct, t + cb

    # --- ENERGY MODE (Existing Greedy Logic) ---
    curr_t, curr_b, curr_l, curr_r = 0, roi_region.shape[0]-1, 0, roi_region.shape[1]-1
    
    while True:
        edge_t = roi_region[curr_t, curr_l:curr_r+1]
        edge_b = roi_region[curr_b, curr_l:curr_r+1]
        edge_l = roi_region[curr_t:curr_b+1, curr_l]
        edge_r = roi_region[curr_t:curr_b+1, curr_r]

        bad_t = np.any(edge_t > 0)
        bad_b = np.any(edge_b > 0)
        bad_l = np.any(edge_l > 0)
        bad_r = np.any(edge_r > 0)

        if not (bad_t or bad_b or bad_l or bad_r):
            break

        # Map to energy_map local coordinates
        lt, rt = (l + curr_l) - bx, (l + curr_r) - bx
        tt, bt = (t + curr_t) - by, (t + curr_b) - by
        
        candidates = []
        # We only consider shrinking sides that actually touch the mask
        if bad_t: candidates.append((np.sum(energy_map[tt, lt:rt+1]), 't'))
        if bad_b: candidates.append((np.sum(energy_map[bt, lt:rt+1]), 'b'))
        if bad_l: candidates.append((np.sum(energy_map[tt:bt+1, lt]), 'l'))
        if bad_r: candidates.append((np.sum(energy_map[tt:bt+1, rt]), 'r'))

        if not candidates: break
        # Sort by energy (least energy lost first)
        candidates.sort(key=lambda x: x[0])
        move = candidates[0][1]

        if move == 't': curr_t += 1
        elif move == 'b': curr_b -= 1
        elif move == 'l': curr_l += 1
        elif move == 'r': curr_r -= 1
        
        if curr_r <= curr_l or curr_b <= curr_t: break

    return l + curr_l, l + curr_r, t + curr_t, t + curr_b


def _find_energy_center(energy_cleaned, hand_mask, bird_mask, bw, bh, visualize=False, reliable_bird_mask=False):
    """
    Docstring for _find_energy_center
    
    All 3 params comes in CROPPED size, corresponding to bw, bh

    Parameters
    ----------
    :param energy_cleaned: Energy map
    :param hand_mask: Description
    :param bird_mask: Description
    :param bw: Description
    :param bh: Description
    :param visualize: Description
    :return: Description
    :rtype: NoReturn
    """
    h, w = energy_cleaned.shape

    energy_weighted = energy_cleaned

    if not reliable_bird_mask:
        # Do some processing to remove external energy (e.g. head/tail). If we have a relaiable bird mask (e.g. from segmentator), those parts WONT be part of it.
        # --- Step 1: Anatomical Depth Mapping ---
        # Points furthest from the mask edges are the 'core'
        bird_depth = cv2.distanceTransform(bird_mask.astype(np.uint8), cv2.DIST_L2, 5)
        cv2.normalize(bird_depth, bird_depth, 0, 1, cv2.NORM_MINMAX)
        
        # --- Step 2: Body Prior (Dynamic Gaussian) ---
        # Find the single deepest point of the bird (the torso center)
        _, _, _, max_depth_loc = cv2.minMaxLoc(bird_depth)
        
        Y, X = np.indices((h, w))
        # Create a Gaussian anchored to the deepest point of the bird
        # This acts as a 'gravity' pulling the center away from the head/beak
        dist_from_torso = np.sqrt((X - max_depth_loc[0])**2 + (Y - max_depth_loc[1])**2)
        bio_prior = np.exp(-dist_from_torso**2 / (2 * (max(h, w)/5)**2))

        # --- Step 3: Weighted Energy Map ---
        # Chest = (Raw Energy) * (Body Depth) * (Proximity to Torso Core)
        energy_weighted = energy_cleaned * bird_depth * bio_prior
    
    
    # --- Step 4: Blob Analysis on Weighted Map ---
    # Normalize to 0-255 for thresholding
    peak = np.max(energy_weighted)
    if peak < 1e-6: return w//2, h//2, energy_cleaned
    
    energy_norm = (energy_weighted / peak * 255).astype(np.uint8)
    _, thresh = cv2.threshold(energy_norm, 100, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    
    if num_labels > 1:
        # Find the blob with the highest total 'bio-energy'
        # This will almost always be the chest because of the bio_prior and depth weights
        best_blob_idx = 1
        max_bio_val = -1
        
        for i in range(1, num_labels):
            blob_mask = (labels == i)
            total_bio_energy = np.sum(energy_weighted[blob_mask])
            if total_bio_energy > max_bio_val:
                max_bio_val = total_bio_energy
                best_blob_idx = i
        
        cx_seed, cy_seed = int(centroids[best_blob_idx][0]), int(centroids[best_blob_idx][1])
    else:
        cx_seed, cy_seed = max_depth_loc # Fallback to deepest point of bird

    if not reliable_bird_mask: #  TODO: FIXXX!!!
        if visualize:
            visualize_anatomical_centering(energy_cleaned, bird_mask, bird_depth, bio_prior, energy_weighted, cx_seed, cy_seed)

    return cx_seed, cy_seed, energy_weighted




def visualize_anatomical_centering(energy_cleaned, bird_mask, bird_depth, bio_prior, energy_weighted, cx, cy):
    """
    Diagnostic plots to show how we isolated the chest from the head.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: The Inputs
    axes[0, 0].imshow(energy_cleaned, cmap='hot')
    axes[0, 0].set_title("1. Raw Masked Energy\n(Still has Head/Tail noise)")
    
    axes[0, 1].imshow(bird_depth, cmap='viridis')
    axes[0, 1].set_title("2. Bird Depth Map\n(Torso is thicker than Head)")
    
    axes[0, 2].imshow(bio_prior, cmap='magma')
    axes[0, 2].set_title("3. Bio-Prior (Gaussian)\n(Anchored to deepest point)")

    # Row 2: The Logic
    axes[1, 0].imshow(energy_weighted, cmap='hot')
    axes[1, 0].set_title("4. Final Weighted Energy\n(Head signal is suppressed)")
    
    # Overlay Plot
    axes[1, 1].imshow(energy_cleaned, cmap='gray', alpha=0.5)
    axes[1, 1].contour(bird_mask, levels=[0.5], colors='cyan', linewidths=1)
    axes[1, 1].scatter(cx, cy, marker='*', s=200, color='yellow', label='Chest Center')
    axes[1, 1].set_title("5. Final Center Overlay")
    axes[1, 1].legend()

    # The Resulting ROI
    roi_size = 100
    axes[1, 2].imshow(energy_cleaned[cy-roi_size:cy+roi_size, cx-roi_size:cx+roi_size], cmap='hot')
    axes[1, 2].set_title("6. Zoomed Chest ROI")

    plt.tight_layout()
    plt.show()




# VIZ  METHODS ########

def plot_matrices(
    pairs,
    cmap="hot",
    figsize=None,
    suptitle=None,
    auto_size=True,
    base_width=5,
    base_height=4,
    show_colorbar=False,
    show_axis=False,
    return_fig=False,
    overlays=None
):
    """
    Plot multiple matrices in a row with optional enhancements.

    Parameters
    ----------
    pairs : list of tuples
        List of (matrix, title) tuples to plot
    cmap : str, default="hot"
        Colormap to use for imshow
    figsize : tuple, optional
        Figure size (width, height). If None and auto_size=True, automatically calculated
    suptitle : str, optional
        Overall title for the entire figure
    auto_size : bool, default=True
        If True and figsize is None, automatically calculate figure size based on number of subplots
    base_width : float, default=5
        Base width per subplot when auto_size=True
    base_height : float, default=4
        Base height per subplot when auto_size=True
    show_colorbar : bool, default=False
        If True, add a colorbar to each subplot
    show_axis : bool, default=False
        If True, show axis ticks and labels
    return_fig : bool, default=False
        If True, return (fig, axes) instead of None
    overlays : list of dict, optional
        List of overlay specifications for each subplot. Each dict can contain:
        - 'scatter': dict with keys 'x', 'y', and optional matplotlib scatter kwargs
        - 'plot': dict with keys 'x', 'y', and optional matplotlib plot kwargs
        - 'text': dict with keys 'x', 'y', 'text', and optional matplotlib text kwargs
        Example: [None, None, {'scatter': {'x': [10, 20], 'y': [10, 20], 'c': 'red'}}]

    Returns
    -------
    None or (fig, axes)
        Returns None by default, or (fig, axes) if return_fig=True

    Examples
    --------
    # Basic usage
    plot_matrices([(matrix1, "Title 1"), (matrix2, "Title 2")])

    # With scatter overlay on last subplot
    plot_matrices(
        [(mat1, "A"), (mat2, "B")],
        overlays=[None, {'scatter': {'x': [5], 'y': [5], 'c': 'yellow', 's': 100, 'marker': '*'}}]
    )

    # With multiple overlays on same subplot
    plot_matrices(
        [(mat, "Title")],
        overlays=[{
            'scatter': {'x': [10], 'y': [10], 'c': 'red', 'marker': 'o'},
            'plot': {'x': [0, 20], 'y': [0, 20], 'color': 'blue', 'linestyle': '--'}
        }]
    )
    """
    n = len(pairs)

    if n == 0:
        print("Warning: No matrices to plot")
        return None

    # Auto-calculate figsize if not provided
    if figsize is None and auto_size:
        figsize = (base_width * n, base_height)
    elif figsize is None:
        figsize = (20, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    # axes is a single object when n == 1, so normalize to a list
    if n == 1:
        axes = [axes]

    # Normalize overlays to match number of subplots
    if overlays is None:
        overlays = [None] * n
    elif len(overlays) < n:
        overlays = list(overlays) + [None] * (n - len(overlays))

    for ax, (mat, title), overlay in zip(axes, pairs, overlays):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(title)

        if not show_axis:
            ax.axis("off")

        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Apply overlays if provided
        if overlay is not None:
            # Handle scatter plots
            if 'scatter' in overlay:
                scatter_data = overlay['scatter']
                x = scatter_data.pop('x', [])
                y = scatter_data.pop('y', [])
                # Store the kwargs and restore them after plotting
                scatter_kwargs = scatter_data.copy()
                ax.scatter(x, y, **scatter_kwargs)
                # Restore the original dict
                scatter_data['x'] = x
                scatter_data['y'] = y

            # Handle line plots
            if 'plot' in overlay:
                plot_data = overlay['plot']
                x = plot_data.pop('x', [])
                y = plot_data.pop('y', [])
                plot_kwargs = plot_data.copy()
                ax.plot(x, y, **plot_kwargs)
                # Add legend if label was provided
                if 'label' in plot_kwargs:
                    ax.legend()
                # Restore the original dict
                plot_data['x'] = x
                plot_data['y'] = y

            # Handle text annotations
            if 'text' in overlay:
                text_data = overlay['text']
                x = text_data.pop('x', 0)
                y = text_data.pop('y', 0)
                text_str = text_data.pop('text', '')
                text_kwargs = text_data.copy()
                ax.text(x, y, text_str, **text_kwargs)
                # Restore the original dict
                text_data['x'] = x
                text_data['y'] = y
                text_data['text'] = text_str

    # Add overall title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if return_fig:
        return fig, axes
    else:
        plt.show()
        return None