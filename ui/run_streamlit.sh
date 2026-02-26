#!/bin/bash
# Quick start script for Breathometer Streamlit UI

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "======================================"
echo "Starting Breathometer Streamlit UI"
echo "======================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r "$SCRIPT_DIR/requirements_streamlit.txt"
fi

# Run the app from the ui directory
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$SCRIPT_DIR"
streamlit run streamlit_app.py
