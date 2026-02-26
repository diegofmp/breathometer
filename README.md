# 🐦 Breathometer

A computer vision pipeline for non-invasive bird breathing rate estimation from video recordings.

## Overview

Breathometer automatically analyzes video recordings to estimate bird breathing rates using optical flow and autocorrelation-based signal processing. The system detects the bird's chest area, tracks respiratory motion, and outputs breathing rate (BPM) with a confidence metric.

### How It Works

The pipeline operates in four main phases:

1. **ROI Localization** - Automatically detects the bird's chest area:
   - Uses a trained segmentation model (RF-DETR) to detect the bird and handler's hand
   - Aggregates detections across multiple frames for robustness
   - Identifies the optimal chest ROI by maximizing area coverage within the detected bird region

2. **Motion Tracking** - Monitors chest movement across frames:
   - Tracks the chest ROI using OpenCV trackers (KCF, CSRT, or MIL)
   - Periodically re-detects the ROI to handle tracking drift
   - Maintains consistent ROI dimensions to prevent size drift

3. **Breathing Signal Extraction** - Quantifies respiratory motion:
   - Computes optical flow within the chest ROI
   - Evaluates flow radiality to isolate breathing motion
   - Generates a time-series signal representing chest expansion/contraction

4. **Rate Estimation** - Analyzes the signal to compute BPM:
   - Preprocesses signal: bandpass filtering, outlier clipping, normalization
   - Applies windowed autocorrelation with overlap (default 7s windows, 50% overlap)
   - Identifies breathing cycles across multiple windows
   - Uses Kernel Density Estimation (KDE) to find consensus among window estimates
   - Returns estimated BPM and confidence score (based on peak prominence and correlation strength)

**Confidence Metric**: The confidence score (0-1) reflects the reliability of the estimate based on autocorrelation peak prominence and consistency across windows. A score below 0.5 (empirical threshold) indicates potentially unreliable results that may require manual validation.

## Project Structure

```
breathometer/
├── configs/                    # Configuration files for pipeline parameters
│   ├── default.yaml           # Default configuration
│   └── *.yaml                 # Experiment-specific configs
├── src/                       # Core pipeline implementation
│   ├── pipeline.py            # Main BreathingAnalyzer class
│   ├── detectors/             # Object detection modules (RF-DETR, manual)
│   ├── localizers/            # ROI localization algorithms
│   ├── measurements/          # Breathing signal extraction methods
│   ├── signal_processing/     # ACF-based rate estimation
│   ├── utils/                 # Helper utilities and validation
│   └── validation.py          # Configuration validation
├── ui/                        # Streamlit web interface
│   ├── streamlit_app.py       # Web UI application
│   ├── requirements_streamlit.txt
│   └── README.md              # UI documentation
├── scripts/                   # Evaluation and analysis scripts
├── models/                    # Pre-trained models (bird detector)
│   └── README.md              # Model setup instructions
├── data/                      # Video datasets and ground truth
│   ├── videos/                # Input video files
│   ├── results/               # Processing results
│   └── baseline/              # Ground truth data
├── output/                    # Output files (plots, CSV results)
├── docs/                      # Documentation
│   ├── DOCKER.md              # Docker usage guide
│   └── CONFIG_REFERENCE.md    # Configuration parameters reference
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── process_single_video.py    # Single video processing script
├── batch_process_videos.py    # Batch video processing script
├── environment.yml            # Conda environment specification
└── README.md                  # This file
```

## Installation

### Option 1: Docker

For consistent, reproducible environments and easy deployment:

```bash
# Clone the repository
git clone <repository-url>
cd breathometer

# Build and start with Docker Compose
docker-compose up
```

Access the web UI at [http://localhost:8501](http://localhost:8501)

See [docs/DOCKER.md](docs/DOCKER.md) for comprehensive Docker usage, GPU support, batch processing, and troubleshooting.

### Option 2: Conda

For local development and faster iteration:

#### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA-capable GPU (recommended, but CPU mode is supported)

#### Setup Environment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd breathometer
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate breathometer
   ```

3. (Optional) If using the web UI, install additional dependencies:
   ```bash
   pip install -r ui/requirements_streamlit.txt
   ```

### Model Setup (Required for Auto Mode)

The pipeline requires two models for automatic ROI detection:

#### 1. Hand Detector (Automatic)
The hand detector model is **automatically downloaded** by the `rfdetr` package on first use. No manual setup required!

#### 2. Bird Detector (Manual - Required)
The bird detector is a **custom-trained model** that must be obtained separately (~385 MB).

**Setup steps:**

1. **Obtain the bird detector model** from the research coordinator.

2. **Place the model in the `models/` directory**.


See [models/README.md](models/README.md) for detailed setup instructions and troubleshooting.

**Note**: If you don't have the bird detector model yet, you can use manual ROI selection mode (see Configuration section).

## Usage

### Command Line Interface

#### Single Video Processing

Process a single video file with [process_single_video.py](process_single_video.py):

```bash
python process_single_video.py --video data/videos/your_video.mp4 --config configs/default.yaml
```

**Options:**
- `--video`: Path to input video file (default: `data/videos/H5_F_2.mp4`)
- `--config`: Path to configuration file (default: `configs/default.yaml`)
- `--plot`: Generate and save result plots to `output/` folder (optional)

**Example:**
```bash
# Basic processing
python process_single_video.py --video data/videos/H5_F_2.mp4

# With plot generation (saves to output/{video_name}_pipeline_results.png)
python process_single_video.py \
    --video data/videos/H5_F_2.mp4 \
    --config configs/default.yaml \
    --plot
```

#### Batch Video Processing

Process all videos in a directory with [batch_process_videos.py](batch_process_videos.py):

```bash
# Basic usage (saves to output/results.csv by default)
python batch_process_videos.py --directory data/videos/

# With custom config and output path
python batch_process_videos.py \
    --directory data/videos/ \
    --config configs/default.yaml \
    --output output/my_results.csv
```

**Options:**
- `--directory`: Directory containing video files (required)
- `--config`: Configuration file path (default: `configs/default.yaml`)
- `--output`: Path to output CSV file (default: `output/results.csv`)
- `--recursive`: Search subdirectories recursively (optional flag)
- `--extensions`: Video file extensions to process (default: .mp4 .avi .mov .mkv)

**Example with recursive search:**
```bash
python batch_process_videos.py \
    --directory data/ \
    --recursive
```

### Web Interface

For an interactive web-based interface with ROI selection and visualization:

```bash
streamlit run ui/streamlit_app.py
```

See [ui/README.md](ui/README.md) for detailed UI documentation.

## Configuration

Pipeline behavior is controlled via YAML configuration files in [configs/](configs/).

### Minimal Configuration (Recommended)

The pipeline uses optimized defaults for most parameters. You typically only need to specify:

```yaml
roi_localization:
  mode: 'auto'                  # 'auto' or 'manual'
  buffer_frames: 30             # Frames for mask aggregation
  hand_mask_buffer_frames: 15   # Frames for detection (max 15 to avoid GPU OOM)

  bird_detector:
    model_path: "/path/to/bird_model.pth"  # Required for auto mode

tracking:
  redetect_interval: 0          # 0 = only re-detect on failure
  start_frame: 300              # Skip initial frames for stabilization
  max_frames: 1800              # Limit processing (null = entire video)
```

**All other parameters** (measurement, signal processing, detection thresholds) use carefully tuned defaults and rarely need adjustment.

For detailed parameter documentation, see [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md).

## Output

### Output Files

All output files are saved to the `output/` directory by default:

- **Plots**: `process_single_video.py --plot` saves plots as `output/{video_name}_pipeline_results.png`
- **CSV Results**: `batch_process_videos.py` saves results to `output/results.csv` (customizable)

The `output/` directory is automatically created if it doesn't exist.

### Results Dictionary

The pipeline returns a dictionary with:

```python
{
    'breathing_rate_bpm': float,      # Estimated breathing rate
    'confidence': float,              # Confidence score (0-1)
    'frequency_hz': float,            # Breathing frequency in Hz
    'signal_length': int,             # Number of signal samples
    'breathing_signal': list,         # Raw breathing signal
    'tracking_status': list,          # Frame-by-frame tracking success
    'metadata': dict,                 # Quality metrics per frame
    'validation': dict,               # Consistency validation results
    'breath_counts': dict             # Breath counts per window
}
```

### Confidence Interpretation

- **confidence < 0.5**: Results may be unreliable and should be validated manually
- Low confidence can indicate:
  - Weak autocorrelation peaks (poor signal periodicity)
  - Inconsistent estimates across windows
  - Poor tracking quality
  - Excessive motion or noise

The confidence threshold can be adjusted via `signal_processing.breath_counting.autocorrelation.min_confidence` in the config (see [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)).

## Examples

### Basic Video Analysis

```python
from src.pipeline import BreathingAnalyzer

# Initialize analyzer
analyzer = BreathingAnalyzer(config_path='configs/default.yaml')

# Process video
results = analyzer.process_video('data/videos/bird.mp4')

# Print results
print(f"Breathing Rate: {results['breathing_rate_bpm']:.1f} BPM")
print(f"Confidence: {results['confidence']:.2f}")
```

### With Manual ROI Selection

```yaml
# configs/manual.yaml
roi_localization:
  mode: manual
  manual_roi: [100, 150, 80, 60]  # [x, y, width, height]
```

```bash
python process_single_video.py --video data/videos/bird.mp4 --config configs/manual.yaml
```

## Performance Notes

- **GPU Acceleration**: RF-DETR detection uses GPU when available, falling back to CPU
- **Batch Inference**: Detections are processed in batches for 2-5x speedup
- **Processing Speed**: Typical videos (~30 fps, 30 seconds) process in 1-3 minutes on GPU

