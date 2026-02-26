# Configuration Reference

This document describes all configuration parameters available in the breathometer pipeline.

## Minimal Configuration

The production config should only expose parameters that users need to adjust. All other parameters use carefully tuned defaults.

```yaml
# ROI Localization (chest detection)
roi_localization:
  mode: 'auto'  # Options: 'auto' (use detectors) or 'manual' (user selects ROI)

  buffer_frames: 30  # Total frames buffered for mask aggregation
  hand_mask_buffer_frames: 15  # Frames used for hand/bird mask detection

  bird_detector:
    model_path: "/path/to/your/bird_detector_model.pth"

# Tracking
tracking:
  redetect_interval: 0
  start_frame: 300
  max_frames: 900

# Measurement - All parameters use defaults
# Signal Processing - All parameters use defaults
```

**All other parameters (measurement, signal processing, detection thresholds, etc.) use optimized defaults and are rarely needed.**

---

## Detailed Parameter Reference

### ROI Localization

The ROI localization system detects the chest region of the bird for breathing measurement. It can work in two modes:
- **Auto mode**: Uses RF-DETR models to detect hand (holding the bird) and bird, then localizes the chest
- **Manual mode**: User manually selects the chest ROI on the first frame

#### Exposed Parameters

| Parameter | Type | Default | Options/Range | Description | When to Adjust |
|-----------|------|---------|---------------|-------------|----------------|
| `mode` | String | 'auto' | 'auto', 'manual' | ROI localization mode | Use 'manual' for debugging or when automatic detection fails |
| `buffer_frames` | Integer | 30 | 20-50 | Total frames buffered for mask aggregation | Increase for more stable localization, decrease for faster processing |
| `hand_mask_buffer_frames` | Integer | 15 | 10-30 | Frames used for hand/bird mask detection (must be ≤ buffer_frames). **Values >15 may cause GPU OOM** | Increase for more robust detection, decrease to reduce GPU memory usage |
| `bird_detector.model_path` | String | None | File path | Path to trained bird detector model (RF-DETR checkpoint) | **Required in auto mode**: Must specify path to your trained model |

#### Hidden Parameters (Use Defaults)

**General:**

| Parameter | Type | Default | Options | Description | When to Override |
|-----------|------|---------|---------|-------------|------------------|
| `hand_detector.model_path` | String | None | File path | Path to custom hand detector model (.pth checkpoint) | Only if using a custom trained hand detector (default uses built-in RF-DETR) |
| `hand_detector.rfdetr_variant` | String | 'medium' | 'nano', 'small', 'medium', 'large', 'xlarge' | Hand detector model size (only used if model_path not provided) | Only for performance tuning when using pre-trained models |
| `hand_detector.confidence_threshold` | Float | 0.3 | 0.0-1.0 | Minimum confidence for hand detection | Only for stricter (higher) or looser (lower) filtering |
| `hand_detector.device` | String | 'auto' | 'auto', 'cpu', 'cuda' | Device for hand detector inference | Rarely; 'auto' handles GPU availability |
| `bird_detector.confidence_threshold` | Float | 0.2 | 0.0-1.0 | Minimum confidence for bird detection | Only if bird masks are too noisy (increase) or too strict (decrease) |
| `bird_detector.device` | String | 'auto' | 'auto', 'cpu', 'cuda' | Device for bird detector inference | Rarely; 'auto' handles GPU availability |
| `bird_detector.rely_segmentator` | Boolean | false | true/false | Whether to rely solely on bird masks without hand validation | Set to true only if bird detection is very accurate |

**How to override**: Add to config file:
```yaml
roi_localization:
  mode: 'auto'
  buffer_frames: 30
  hand_mask_buffer_frames: 15

  hand_detector:
    confidence_threshold: 0.3
    device: 'auto'
    rfdetr_variant: 'medium'

  bird_detector:
    model_path: "/path/to/bird_model.pth"
    confidence_threshold: 0.2
    device: 'auto'
    rely_segmentator: false
```

**Advanced Hidden Parameters** (Chest Localization):

| Parameter | Type | Default | Range | Description | When to Override |
|-----------|------|---------|-------|-------------|------------------|
| `localization.smooth_kernel_size` | Integer (odd) | 21 | 11-31 | Kernel size for Gaussian smoothing during chest localization | Only if you need different noise reduction levels |

**How to override advanced parameters**: Add to config file:
```yaml
localization:
  smooth_kernel_size: 21
```

---

### Tracking

#### Exposed Parameters

| Parameter | Type | Default | Range | Description | When to Adjust |
|-----------|------|---------|-------|-------------|----------------|
| `redetect_interval` | Integer | 0 | 0 or 100-500 | Number of frames between re-running chest localization. Set to 0 to disable periodic re-detection (only re-detect when tracking fails). | 0: Most efficient, only re-detects on tracking failure<br>100-500: Periodic re-detection for very long recordings with bird movement |
| `start_frame` | Integer | 300 | 0-500 | Frame number to start processing. Skips initial frames to avoid camera stabilization artifacts. | Increase if your videos have longer initial stabilization periods |
| `max_frames` | Integer or null | 1800 | 300-3000 or null | Maximum number of frames to process. Set to null to process entire video. | Limit for testing/debugging or when you only need a specific duration |

#### Hidden Parameters (Use Defaults)

| Parameter | Type | Default | Options | Description | When to Override |
|-----------|------|---------|---------|-------------|------------------|
| `chest_tracker` | String | 'KCF' | 'KCF', 'CSRT', 'MIL' | OpenCV tracker algorithm for chest ROI tracking between localizations. KCF = Kernelized Correlation Filters | Only if KCF fails for your specific use case (rare) |

**How to override**: Add to config file:
```yaml
tracking:
  chest_tracker: 'KCF'  # or 'CSRT' or 'MIL'
  redetect_interval: 0
  start_frame: 300
  max_frames: 1800
```

---

### Measurement

**All measurement parameters are hidden and use sensible defaults.** The default optical flow configuration is optimized for breathing measurement and rarely needs adjustment.

**Note:** The entire `measurement` section can be omitted from the config file - all parameters will use their defaults automatically.

#### Optical Flow Parameters (Hidden - Use Defaults)

| Parameter | Type | Default | Range | Description | When to Override |
|-----------|------|---------|-------|-------------|------------------|
| `pyr_scale` | Float | 0.5 | 0.3-0.7 | Image pyramid scale (0.5 = 50% reduction per level) | Only if dealing with extremely large or small breathing motions |
| `levels` | Integer | 3 | 2-5 | Number of pyramid levels for optical flow | Rarely needed; increase for very large motions |
| `winsize` | Integer | 15 | 7-25 | Averaging window size for optical flow | Only for noisy videos (increase) or need finer detail (decrease) |
| `iterations` | Integer | 3 | 2-5 | Iterations at each pyramid level | Rarely; increase only if flow accuracy is insufficient |
| `poly_n` | Integer | 5 | 5-7 | Polynomial expansion neighborhood size | Almost never |
| `poly_sigma` | Float | 1.2 | 1.0-1.5 | Gaussian sigma for polynomial expansion | Almost never |
| `patch_rows` | Integer | 3 | 2-5 | Number of patches to split ROI height for robust measurement | Only for specialized use cases with unusual ROI shapes |
| `patch_cols` | Integer | 3 | 2-5 | Number of patches to split ROI width for robust measurement | Only for specialized use cases with unusual ROI shapes |

**How to override**: Add to config file:
```yaml
measurement:
  pyr_scale: 0.5
  levels: 3
  winsize: 15
  iterations: 3
  poly_n: 5
  poly_sigma: 1.2
  patch_rows: 3
  patch_cols: 3
```

#### Advanced Parameters (Hidden - DO NOT Override)

| Parameter | Type | Default | Description | When to Override |
|-----------|------|---------|-------------|------------------|
| `use_patches` | Boolean | true | Enable patch-based measurement (always recommended) | **Never** (required for production quality) |
| `use_median` | Boolean | false | Use median instead of mean for patch aggregation | **Never** (mean performs better for breathing signals) |

---

### Signal Processing

The signal processing pipeline transforms raw breathing signals into clean, analyzable data and estimates breathing rate. All parameters have carefully tuned defaults and rarely need adjustment.

**Note:** The entire `signal_processing` section can be omitted from the config file - all parameters will use their defaults automatically.

#### Hidden Parameters (Use Defaults)

#### Top-Level Signal Processing

| Parameter | Type | Default | Description | When to Override |
|-----------|------|---------|-------------|------------------|
| `fps` | Integer | 30 | Video frame rate (auto-detected from video metadata) | Only if auto-detection fails |

#### Breath Counting Method

| Parameter | Type | Default | Options | Description | When to Override |
|-----------|------|---------|---------|-------------|------------------|
| `breath_counting.method` | String | 'autocorrelation_windowed' | 'autocorrelation', 'autocorrelation_windowed' | Method used for breath rate estimation | Use 'autocorrelation' for single-window analysis (faster), but 'autocorrelation_windowed' is recommended for production |

**Method Descriptions:**
- **autocorrelation_windowed** (Default, Recommended): Detects periodicity using autocorrelation in sliding windows. Most robust to amplitude variation and noise.
- **autocorrelation**: Single-window autocorrelation. Faster but less robust than windowed version.

#### Bandpass Filter

Applied after preprocessing to isolate breathing frequencies:

| Parameter | Type | Default | Range | Description | When to Override |
|-----------|------|---------|-------|-------------|------------------|
| `bandpass_filter.low_freq` | Float | 1.5 | 0.5-2.0 | Low cutoff frequency in Hz (90 BPM) | Only for species with very different breathing rates |
| `bandpass_filter.high_freq` | Float | 8.0 | 5.0-10.0 | High cutoff frequency in Hz (480 BPM) | Only for species with very different breathing rates |
| `bandpass_filter.order` | Integer | 2 | 2-4 | Butterworth filter order (higher = sharper cutoff) | Rarely; 2 is optimal for most cases |

#### Preprocessing Pipeline

Applied in order: bandpass → integration → second bandpass → outlier_clip → normalize

**Normalization:**

| Parameter | Type | Default | Options | Description | When to Override |
|-----------|------|---------|---------|-------------|------------------|
| `preprocessing.normalize.enabled` | Boolean | true | true/false | Normalize signal amplitude | Never disable (required for consistent processing) |
| `preprocessing.normalize.method` | String | 'zscore' | 'zscore', 'minmax', 'robust' | Normalization method | Rarely; zscore is optimal for breathing signals |

**Outlier Clipping:**

| Parameter | Type | Default | Range | Description | When to Override |
|-----------|------|---------|-------|-------------|------------------|
| `preprocessing.outlier_clip.std_threshold` | Float | 3.0 | 2.0-5.0 | Clip outliers beyond N standard deviations | Only for very noisy signals (decrease) or to preserve extreme values (increase) |

#### Autocorrelation Method Parameters

Only used when `breath_counting.method` = 'autocorrelation' or 'autocorrelation_windowed':

| Parameter | Type | Default | Range | Description | When to Override |
|-----------|------|---------|-------|-------------|------------------|
| `breath_counting.autocorrelation.min_breathing_rate_bpm` | Integer | 90 | 60-120 | Minimum expected breathing rate (BPM) | Adjust for species with different breathing rates |
| `breath_counting.autocorrelation.max_breathing_rate_bpm` | Integer | 400 | 200-500 | Maximum expected breathing rate (BPM) | Adjust for species with different breathing rates |
| `breath_counting.autocorrelation.acf_min_prominence` | Float | 0.03 | 0.01-0.3 | Minimum autocorrelation peak prominence (0-1 scale) | Increase for stricter peak detection, decrease for more sensitivity |
| `breath_counting.autocorrelation.acf_peak_selection` | String | 'first' | 'first', 'highest' | Peak selection strategy | 'first' = fundamental frequency (recommended), 'highest' = strongest peak |
| `breath_counting.autocorrelation.min_confidence` | Float | 0.3 | 0.2-0.5 | Minimum confidence to trust estimate | Increase for stricter quality filtering |
| `breath_counting.autocorrelation.low_correlation_threshold` | Float | 0.25 | 0.2-0.5 | ACF peak value below this triggers low-confidence warning | Increase for stricter quality requirements |
| `breath_counting.autocorrelation.window_size` | Float | 7.0 | 5.0-30.0 | Window duration in seconds (for windowed version) | Longer = better periodicity detection but less responsive |
| `breath_counting.autocorrelation.overlap` | Float | 0.5 | 0.3-0.9 | Window overlap fraction for windowed version | Rarely; 0.5 is optimal |

#### How to Override Hidden Parameters

Add the parameters you want to override to your config file:

```yaml
signal_processing:
  fps: 30  # Override auto-detected FPS

  # Override bandpass filter
  bandpass_filter:
    low_freq: 1.5
    high_freq: 8.0
    order: 2

  # Override preprocessing
  preprocessing:
    normalize:
      enabled: true
      method: 'zscore'
    outlier_clip:
      std_threshold: 3.0

  # Breath counting method
  breath_counting:
    method: 'autocorrelation_windowed'

    # Autocorrelation parameters
    autocorrelation:
      min_breathing_rate_bpm: 90
      max_breathing_rate_bpm: 400
      acf_min_prominence: 0.03
      acf_peak_selection: 'first'
      min_confidence: 0.3
      low_correlation_threshold: 0.25
      window_size: 7.0
      overlap: 0.5
```
