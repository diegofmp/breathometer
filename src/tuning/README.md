# Hyperparameter Tuning

This folder contains tools for optimizing breathing detection parameters through systematic evaluation on labeled datasets.

## Overview

Hyperparameter tuning involves:
1. Extracting breathing signals from videos (slowest step - done once)
2. Caching signals for reuse (speeds up optimization dramatically)
3. Running grid search to find optimal parameters
4. Comparing different configurations

## Workflow

### 1. Generate ROIs (Optional - if needed)

If you don't have a `roi_manifest.json` file, generate automatic ROIs:

```bash
python src/tuning/generate_automatic_rois.py \
  --input-dir data/videos \
  --output roi_manifest.json \
  --config configs/default.yaml
```

**What it does:** Automatically detects chest ROIs for each video and saves them to a manifest file.

**When to use:** First time setup or when adding new videos to your dataset.

### 2. Extract and Cache Signals

Extract breathing signals from videos and cache them for reuse:

```bash
python src/tuning/extract_signals.py \
  --roi-file roi_manifest.json \
  --config configs/default.yaml \
  --output cache/
```

**What it does:**
- Extracts breathing signals using optical flow measurement
- Caches signals to disk for fast parameter tuning
- This is the **slowest and most repetitive part** - only needs to run once

**Why cache signals?**
Signal extraction (optical flow measurement) is computationally expensive and identical across all parameter combinations. Caching eliminates this bottleneck, allowing optimization to iterate quickly through parameter combinations.

**Inputs:**
- `roi_manifest.json`: Chest ROI locations and ground truth for each video
- `configs/default.yaml`: Configuration for measurement parameters

**Output:**
- `cache/` directory containing pickled signal files

**Resume capability:** Use `--no-resume` to force re-extraction. By default, skips videos with existing cached signals.

### 3. Optimize Parameters

Run grid search to find optimal autocorrelation parameters:

```bash
# Full optimization (recommended)
python src/tuning/optimize_acf_params.py \
  --cache-dir cache/ \
  --config configs/default.yaml \
  --output configs/tuned_windowed_autocorr.yaml \
  --jobs 6

# Quick test (fewer combinations)
python src/tuning/optimize_acf_params.py \
  --cache-dir cache/ \
  --config configs/default.yaml \
  --output configs/tuned.yaml \
  --quick \
  --jobs 4
```

**What it does:**
- Loads cached signals (fast!)
- Tests all combinations of parameters via grid search
- Evaluates each combination against ground truth
- Saves best parameters to output config file

**Supported search parameters:**

*Autocorrelation (ACF) Parameters:*
- `acf_min_prominence`: Peak detection sensitivity (0.03-0.15)
- `window_size` or `acf_window_size`: Analysis window in seconds (5.0-15.0)
- `acf_peak_selection`: Peak selection strategy ('first' or 'prominent')
- `low_correlation_threshold`: Quality filtering threshold (0.2-0.4)
- `overlap` or `acf_overlap`: Window overlap fraction (0.3-0.9)
- `min_confidence` or `acf_min_confidence`: Minimum confidence threshold (0.2-0.5)
- `acf_min_bpm`: Minimum breathing rate to consider (60-120 BPM)
- `acf_max_bpm`: Maximum breathing rate to consider (200-500 BPM)

*Bandpass Filter Parameters:*
- `bandpass_low_freq`: Low cutoff frequency in Hz (0.5-2.0)
- `bandpass_high_freq`: High cutoff frequency in Hz (5.0-10.0)
- `bandpass_order`: Filter order (2-4)

**Note:** To optimize any of these parameters, add them to the search space dictionary in `optimize_acf_params.py` (lines 29-42).

**Parallelization:** Use `--jobs N` to run evaluations in parallel (recommended: number of CPU cores)

**Output:**
- Updated config file with optimized parameters
- `optimization_metadata` section documenting search space and results

### 4. Compare Configurations

Compare performance across different configurations:

```bash
python src/tuning/compare_methods.py \
  --cache-dir cache/ \
  --configs configs/acf_10s.yaml configs/acf_30s.yaml configs/acf_60s.yaml \
  --output comparison_results.json
```

**What it does:**
- Evaluates multiple configurations on the same cached signals
- Reports performance metrics for each configuration
- Helps validate optimization results and compare approaches

**Useful for:**
- Comparing optimized vs default parameters
- Testing different window sizes for different video lengths
- Validating that optimization improved performance

## File Descriptions

### Scripts

- **`generate_automatic_rois.py`**: Automatic chest ROI detection for video datasets
- **`extract_signals.py`**: Signal extraction and caching (run once, reuse many times)
- **`optimize_acf_params.py`**: Grid search hyperparameter optimization
- **`compare_methods.py`**: Configuration comparison and validation

### Modules

- **`signal_cache.py`**: Signal caching utilities (save/load signals from disk)
- **`optimizer.py`**: Grid search optimization implementation

## Adding New Parameters to Optimize

To add a new parameter to the optimization search space:

### 1. Check if parameter is already supported

Look in `optimizer.py` at the `param_mapping` dictionary (lines 220-236). Supported parameters include:
- All autocorrelation parameters (acf_min_prominence, window_size, overlap, etc.)
- Bandpass filter parameters (bandpass_low_freq, bandpass_high_freq, bandpass_order)

### 2. Add to search space in `optimize_acf_params.py`

Edit the `DEFAULT_SEARCH_SPACE` dictionary (lines 29-42):

```python
DEFAULT_SEARCH_SPACE = {
    'acf_min_prominence': [0.03, 0.05, 0.12, 0.15],
    'window_size': [5.0, 7.0, 8.0, 10.0, 15.0],
    'acf_peak_selection': ['first', 'prominent'],
    'low_correlation_threshold': [0.2, 0.25, 0.3, 0.4],

    # Add new parameters here
    'overlap': [0.5, 0.75],  # Example: add overlap tuning
    'bandpass_low_freq': [1.0, 1.5, 2.0],  # Example: tune bandpass filter
}
```

### 3. If parameter is NOT in param_mapping

If you need to optimize a parameter not in the mapping, add it to `optimizer.py`:

```python
param_mapping = {
    # ... existing mappings ...
    'your_new_param': ['signal_processing', 'breath_counting', 'your_new_param'],
}
```

The path should match the YAML config structure. Please refer to the [Configuration Reference](../../docs/CONFIG_REFERENCE.md) for all available parameters and their paths.

## Tips

1. **Start with signal caching**: Always extract and cache signals first - this is the slowest step and only needs to run once

2. **Use parallel jobs**: Optimization is CPU-bound and parallelizes well. Use `--jobs` equal to your CPU core count

3. **Quick test first**: Use `--quick` mode to verify your setup before running full optimization

4. **Window size matters**: Longer videos benefit from larger window sizes (e.g., 7-8s for 60s videos, 5s for 10-30s videos)

5. **Bandpass filter**: Default is 1.5-8.0 Hz (90-480 BPM). Adjust in config if your species has different breathing rates

6. **Grid search explosion**: Be mindful of combinatorial explosion. 4 parameters with 4 values each = 256 combinations. Use `--quick` mode or reduce search space for initial testing

## ROI Manifest Format

`roi_manifest.json` maps video paths to ROI information. Each entry includes:

```json
{
  "/path/to/video1.mp4": {
    "roi": [x, y, width, height],
    "frame_number": 300,
    "ground_truth_bpm": 180.0,
    "roi_source": "automatic",
    "timestamp": "2026-02-26T23:13:45.987925",
    "metadata": {
      "fps": 30.0,
      "total_frames": 2651,
      "detection_method": "automatic"
    }
  }
}
```

**Required fields:**
- `roi`: `[x, y, width, height]` chest bounding box in pixels
- `frame_number`: Frame where ROI was detected/defined
- `ground_truth_bpm`: True breathing rate for validation (required for optimization)

**Optional fields:**
- `roi_source`: "automatic" or "manual"
- `metadata`: Video properties and detection configuration
- `timestamp`: When ROI was generated
