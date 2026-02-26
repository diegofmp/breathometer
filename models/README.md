# Model Files Directory

This directory contains custom-trained models for the Breathometer pipeline.

## Model Overview

The pipeline uses two types of models for automatic ROI detection:

### 1. Hand Detector Model (Auto-downloaded by rfdetr package)
- **Model**: RF-DETR Segmentation (Medium variant)
- **Size**: ~137 MB
- **Purpose**: Detects handler's hand in videos
- **Download**: **Automatically downloaded** by the `rfdetr` package on first use
- **Storage**: Cached by the rfdetr package (not in this directory)
- **Configuration**: Uses default pretrained weights unless custom path specified

**Note**: You don't need to manually download this model! The rfdetr package handles it automatically.

### 2. Bird Detector Model (Custom trained - REQUIRED)
- **Filename**: `bird_detector.pth` (or custom name)
- **Size**: ~385 MB (varies by model)
- **Purpose**: Detects and segments bird in videos
- **Download**: Must be obtained from research coordinator
- **Distribution**: USB drive or internal server
- **Required**: YES - this is a custom-trained model specific to your bird species

## Directory Structure

```
models/
├── README.md                    # This file
├── bird_detector.pth           # Custom bird detector (download required)
└── .gitkeep                    # Keeps directory in git

Note: Hand detector model is NOT stored here - it's auto-downloaded by rfdetr package
```

## Setup Instructions

### Step 1: Hand Detector (Automatic - No Action Needed)

The hand detector model is **automatically downloaded** by the `rfdetr` package when you first run the pipeline.

On first use, you'll see:
```
Loading RF-DETR model (variant=medium, target_class=hand) on cuda
  Loading pre-trained medium model
Downloading: "https://..." ...
```

The model is cached by the rfdetr package and reused on subsequent runs.

### Step 2: Bird Detector (Manual Setup Required)

#### Obtain the Custom Bird Detector Model

The bird detector is a **custom-trained model**. Please contact to get the model.


#### Place Bird Detector in Directory

Copy the bird detector model to this directory.


### Step 3: Update Configuration (if needed)

The default configuration ([configs/default.yaml](../configs/default.yaml)) is pre-configured to use models from this directory:

```yaml
roi_localization:
  bird_detector:
    model_path: "models/bird_detector.pth"
```

If you rename model files, update these paths accordingly.

## Docker Usage

When using Docker, the models directory is mounted as a volume:

```bash
# Models directory is automatically mounted
docker-compose up
```

The docker-compose configuration mounts `./models:/app/models` so models are accessible inside the container.

## Manual Mode

If you don't have model files yet, you can use manual ROI selection mode:

```yaml
# configs/manual.yaml
roi_localization:
  mode: 'manual'
  manual_roi: [x, y, width, height]  # Set via UI or config
```

See [UI documentation](../ui/README.md) for manual ROI selection via the web interface.
