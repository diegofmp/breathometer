#!/bin/bash
# Quick test to verify Streamlit app syntax

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Testing Streamlit app syntax..."
cd "$SCRIPT_DIR"
python3 -m py_compile streamlit_app.py

if [ $? -eq 0 ]; then
    echo "✅ Syntax check passed!"
    echo ""
    echo "To run the app:"
    echo "  cd ui && streamlit run streamlit_app.py"
    echo "  or"
    echo "  ./ui/run_streamlit.sh"
else
    echo "❌ Syntax errors found!"
    exit 1
fi
