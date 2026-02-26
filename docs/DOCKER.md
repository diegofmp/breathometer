# Docker Usage Guide

This guide covers how to use Breathometer with Docker for consistent, reproducible deployments across different environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage Scenarios](#usage-scenarios)
  - [Web UI (Streamlit)](#web-ui-streamlit)
  - [Single Video Processing](#single-video-processing)
  - [Batch Video Processing](#batch-video-processing)
- [GPU Support](#gpu-support)
- [Volume Mounting](#volume-mounting)
- [Configuration Management](#configuration-management)


## Prerequisites

### Required

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Usually included with Docker Desktop

### Optional (for GPU support)

- **NVIDIA GPU** with CUDA support
- **NVIDIA Container Toolkit**: [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Verify Installation

```bash
# Check Docker
docker --version
docker-compose --version

# Check GPU support (optional)
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Build the Docker Image

From the project root directory:

```bash
# Build the image
docker-compose build

# Or build manually
docker build -t breathometer .
```

**First build takes 10-15 minutes** due to PyTorch and dependencies. Subsequent builds are faster thanks to Docker layer caching.

### 2. Start the Web UI

```bash
docker-compose up
```

Access the Streamlit interface at [http://localhost:8501](http://localhost:8501)

### 3. Stop the Container

```bash
# Stop and remove containers
docker-compose down

# Or press Ctrl+C in the terminal where it's running
```

## Usage Scenarios

### Web UI (Streamlit)

The default configuration starts the Streamlit web interface for interactive video analysis.

**Start the UI:**

```bash
docker-compose up
```

**Access the interface:**
- URL: [http://localhost:8501](http://localhost:8501)
- Upload videos directly through the web interface
- Select ROI manually or use automatic detection
- View results and download processed videos

**Run in detached mode:**

```bash
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Single Video Processing

Process a single video using the CLI inside the container.

**Basic usage:**

```bash
docker-compose run --rm breathometer \
  python process_single_video.py \
  --video /app/data/your_video.mp4 \
  --config /app/configs/default.yaml
```

**With plot generation:**

```bash
docker-compose run --rm breathometer \
  python process_single_video.py \
  --video /app/data/H5_F_2.mp4 \
  --config /app/configs/default.yaml \
  --plot
```

**With output video:**

```bash
docker-compose run --rm breathometer \
  python process_single_video.py \
  --video /app/data/bird.mp4 \
  --config /app/configs/default.yaml \
  --output /app/output/processed_bird.mp4
```

**Results** are saved to `./output/` on your host machine (mounted as volume).

### Batch Video Processing

Process multiple videos in a directory.

**Basic batch processing:**

```bash
docker-compose run --rm breathometer \
  python batch_process_videos.py \
  --directory /app/data \
  --output /app/output/results.csv
```

**With output videos:**

```bash
docker-compose run --rm breathometer \
  python batch_process_videos.py \
  --directory /app/data \
  --config /app/configs/default.yaml \
  --output /app/output/results.csv \
  --output-videos /app/output/processed_videos/
```

**Recursive directory search:**

```bash
docker-compose run --rm breathometer \
  python batch_process_videos.py \
  --directory /app/data \
  --output /app/output/results.csv \
  --recursive
```

**Custom video extensions:**

```bash
docker-compose run --rm breathometer \
  python batch_process_videos.py \
  --directory /app/data \
  --output /app/output/results.csv \
  --extensions .mp4 .avi .mov
```

## GPU Support

### Enable GPU in Docker Compose

Edit `docker-compose.yml` and uncomment the GPU configuration:

```yaml
services:
  breathometer:
    environment:
      - CUDA_VISIBLE_DEVICES=0  # Use first GPU

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Run with GPU (Manual Docker Command)

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/configs:/app/configs:ro \
  breathometer \
  python process_single_video.py \
  --video /app/data/bird.mp4 \
  --config /app/configs/default.yaml
```

### Verify GPU Access

```bash
docker-compose run --rm breathometer python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### GPU Performance Notes

- **Detection phase**: 5-10x faster with GPU (RF-DETR model)
- **Tracking/signal processing**: Minimal GPU benefit (CPU-bound)
- **Fallback**: Pipeline automatically uses CPU if GPU unavailable
- **Memory**: GPU detection requires ~2-4 GB VRAM

## Volume Mounting

The Docker configuration mounts four directories by default:

### 1. Data Directory (`./data` → `/app/data`)

**Purpose**: Input videos

```bash
# Example structure
breathometer/
├── data/
│   ├── bird1.mp4
│   ├── bird2.mp4
│   └── experiments/
│       └── test_video.mp4
```

**Access in container**: `/app/data/bird1.mp4`

### 2. Output Directory (`./output` → `/app/output`)

**Purpose**: Results, processed videos, plots

```bash
# Example structure
breathometer/
├── output/
│   ├── results.csv
│   ├── processed_videos/
│   └── plots/
```

**Access in container**: `/app/output/results.csv`

### 3. Configs Directory (`./configs` → `/app/configs`)

**Purpose**: Pipeline configuration files (read-only)

```bash
# Example structure
breathometer/
├── configs/
│   ├── default.yaml
│   ├── manual.yaml
│   └── custom_experiment.yaml
```

**Access in container**: `/app/configs/default.yaml`

**Note**: Configs are mounted read-only (`:ro`) to prevent accidental modifications.

### 4. Models Directory (`./models` → `/app/models`)

**Purpose**: Custom-trained model files (read-only)

```bash
# Example structure
breathometer/
├── models/
│   ├── bird_detector.pth        # Custom bird detector (~385 MB) - REQUIRED
│   └── README.md                # Setup instructions
```

**Access in container**: `/app/models/bird_detector.pth`

**Note**:
- Models are mounted read-only (`:ro`)
- Model files are NOT included in the Docker image (to keep image size small)
- You must obtain the **bird_detector.pth** from your coordinator before running
- Hand detector model is auto-downloaded by rfdetr package (not stored here)
- See [models/README.md](../models/README.md) for setup instructions

### Custom Volume Mounts

Override default mounts with custom directories:

```bash
docker run --rm \
  -v /path/to/my/videos:/app/data \
  -v /path/to/my/results:/app/output \
  -v /path/to/my/configs:/app/configs:ro \
  -v /path/to/my/models:/app/models:ro \
  breathometer \
  python batch_process_videos.py \
  --directory /app/data \
  --output /app/output/results.csv
```

## Configuration Management

### Using Different Configs

**Option 1: Mount custom config directory**

```bash
docker run --rm \
  -v $(pwd)/my_configs:/app/configs:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  breathometer \
  python process_single_video.py \
  --video /app/data/bird.mp4 \
  --config /app/configs/my_custom.yaml
```

**Option 2: Edit configs locally before running**

Since `./configs` is mounted, you can edit YAML files on your host and they're immediately available in the container.

### Model File Configuration

The pipeline uses two models:

#### 1. Hand Detector (Auto-downloaded)
**No setup required!** The `rfdetr` package automatically downloads the hand detector model (~137 MB) on first use.

On first run inside Docker, you'll see:
```
Loading RF-DETR model (variant=medium, target_class=hand)
  Loading pre-trained medium model
Downloading...
```

The model is cached and reused on subsequent runs.

#### 2. Bird Detector (Manual Setup)
**The custom bird detector must be placed in the `models/` directory** before running.

**Setup Steps:**

1. **Obtain the bird detector** from your research coordinator (USB drive or internal server)

2. **Place in `models/` directory**:
   ```bash
   cp /path/to/bird_detector.pth models/
   ```

3. **Verify it's in place**:
   ```bash
   ls -lh models/
   # Expected:
   # bird_detector.pth (~385 MB)
   ```

4. **Start Docker** - models directory is automatically mounted:
   ```bash
   docker-compose up
   ```

**Configuration**: The bird detector is referenced in [configs/default.yaml](../configs/default.yaml):
```yaml
roi_localization:
  hand_detector: {}  # Uses auto-downloaded pretrained model
  bird_detector:
    model_path: "models/bird_detector.pth"  # Custom model
```

**See**: [models/README.md](../models/README.md) for detailed setup and troubleshooting.


## Summary

**Quick commands reference:**

```bash
# Build image
docker-compose build

# Start UI
docker-compose up

# Process single video
docker-compose run --rm breathometer \
  python process_single_video.py \
  --video /app/data/video.mp4

# Batch processing
docker-compose run --rm breathometer \
  python batch_process_videos.py \
  --directory /app/data \
  --output /app/output/results.csv

# Interactive shell
docker-compose run --rm breathometer bash

# Clean up
docker-compose down
```

For more information:
- [Main README](../README.md) - General usage and pipeline documentation
- [Configuration Reference](CONFIG_REFERENCE.md) - Detailed config parameters
- [UI Documentation](../ui/README.md) - Streamlit interface guide
