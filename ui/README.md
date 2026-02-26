# 🐦 Breathometer Web UI

A Streamlit-based web interface for analyzing bird breathing rates from video recordings.

## Installation

Install the required dependencies:

```bash
pip install -r requirements_streamlit.txt
```

## Running the Application

### Quick Start

Run the application with Streamlit:

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

## Features

### Single Video Analysis
- **Video upload and preview**: Upload video files for analysis
- **Interactive ROI selection**: For manual detection mode, select chest ROI using drawable canvas or coordinate entry
- **Results visualization**: View breathing rate estimates, signal plots, and quality metrics

### Batch Processing
- **Multiple video upload**: Process multiple videos at once
- **CSV-based input**: Upload a CSV file with video paths and optional ground truth breathing rates
- **Batch results export**: Download results as CSV with metrics for all processed videos
- **Ground truth comparison**: When provided, compare estimated rates against known values

### Configuration Management
- **Live validation**: Configuration changes are validated before processing

## UI Configuration

The UI behavior can be customized via `ui_config.yaml`:

- **batch.max_videos**: Maximum number of videos for batch processing
- **batch.max_log_lines**: Maximum log lines to display during processing
- **formatting**: Number formatting for results display