#!/bin/bash

echo "============================================================"
echo "Starting Healthcare Data Generation Frontend"
echo "============================================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/frontend"

# Check for Streamlit
if ! python -m streamlit --version &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting Streamlit app..."
echo ""
python -m streamlit run app.py
