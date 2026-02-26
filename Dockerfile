# Dockerfile for Breathometer Pipeline
# Supports both UI (Streamlit) and CLI (batch processing, single video)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch from the PyTorch wheel index
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages from PyPI
RUN pip install --no-cache-dir \
    # Detection & tracking
    rfdetr==1.4.1 \
    supervision==0.27.0 \
    # Computer vision
    opencv-python-headless==4.10.0.84 \
    opencv-contrib-python==4.13.0.90 \
    pillow==10.4.0 \
    # Signal processing & ML
    numpy==2.2.6 \
    scipy==1.15.3 \
    scikit-learn \
    pandas==2.3.3 \
    polars==1.37.1 \
    pyarrow==23.0.1 \
    # Streamlit UI
    streamlit==1.40.2 \
    streamlit-drawable-canvas==0.9.3 \
    plotly==6.5.2 \
    # Visualization
    matplotlib \
    seaborn \
    pyyaml==6.0.3 \
    # Utilities
    pydantic==2.12.5 \
    tqdm

# Copy the application code
COPY src/ ./src/
COPY ui/ ./ui/
COPY configs/ ./configs/
COPY batch_process_videos.py .
COPY process_single_video.py .

# Create output directory for results
RUN mkdir -p /app/output /app/data

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default to Streamlit UI
# Can be overridden with: docker run breathometer python batch_process_videos.py ...
CMD ["streamlit", "run", "ui/streamlit_app.py"]
