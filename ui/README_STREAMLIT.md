# Breathometer Streamlit Web UI

Simple web interface for video breathing rate analysis.

## Quick Start

### 1. Install Dependencies

```bash
cd ui
pip install -r requirements_streamlit.txt
```

### 2. Run the App

**Option 1: Using the run script (recommended)**
```bash
./ui/run_streamlit.sh
```

**Option 2: Direct command**
```bash
cd ui
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select Configuration**: Choose a config file from the sidebar (default: `configs/default.yaml`)
2. **Upload Video**: Upload a video file (MP4, AVI, MOV, or MKV)
3. **Process**: Click "Process Video" to analyze
4. **View Results**: See breathing rate, confidence, and detailed metrics
5. **Download**: Optionally download processed video and results

## Features

- 📹 **Compact video preview** - Small, efficient video preview in the upload section
- 🚀 **One-click processing** - Simple button to start analysis (disabled during processing)
- 📊 **Progress tracking** - Real-time progress bar showing processing status
- 📋 **Real-time logs** - Live streaming of processing output as it happens
- 📈 **Interactive results** - Breathing rate, confidence, tracking metrics
- 📊 **Signal analysis plots** - Visual analysis of breathing signal, tracking, and motion
- 📥 **Export options** - Download processed videos and JSON results
- ⚙️ **Multiple configurations** - Select from available config profiles
- 🔒 **Safe processing** - Button disabled during analysis to prevent conflicts

## Configuration

Config files are located in the `configs/` directory. You can select different configurations from the sidebar to experiment with different analysis parameters.

## Containerization (Docker)

### Using Docker Compose (Recommended)

```bash
cd ui
docker-compose up -d
```

The app will be available at `http://localhost:8501`

To stop:
```bash
cd ui
docker-compose down
```

### Using Docker directly

```bash
# Build from project root
docker build -f ui/Dockerfile -t breathometer-ui .

# Run
docker run -p 8501:8501 breathometer-ui
```

## Notes

- Processing time depends on video length and resolution
- Processed videos are temporarily stored and cleaned up automatically
- Results can be exported as JSON for further analysis
