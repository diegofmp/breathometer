import numpy as np
import cv2
import matplotlib.pyplot as plt

def clip_to_mask_smart(mask, l, r, t, b):
    """
    Finds the optimal sub-rectangle within [l, r, t, b] that avoids mask > 0.
    Uses Global Maximum Rectangle (Histogram algorithm) to maximize area.

    Args:
        mask: Binary mask where >0 indicates areas to avoid
        l, r, t, b: Initial rectangle boundaries (left, right, top, bottom)

    Returns:
        Tuple of (left, right, top, bottom) coordinates of the largest
        rectangle that avoids the masked areas
    """
    roi_region = mask[t:b+1, l:r+1]

    # If the whole area is already clean, return it
    if np.all(roi_region == 0):
        return l, r, t, b

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