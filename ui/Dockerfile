# Dockerfile for Breathometer Streamlit App
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ui/requirements_streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy the application code
COPY ui/streamlit_app.py ./ui/
COPY src/ ./src/
COPY configs/ ./configs/

# Set working directory to ui folder
WORKDIR /app/ui

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
CMD ["streamlit", "run", "streamlit_app.py"]
