#!/bin/bash
# Linux/macOS shell script to start the API server
# Run this from the backend directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to current directory
export PYTHONPATH="$SCRIPT_DIR"

# Check if pipenv is available
if command -v pipenv &> /dev/null; then
    echo "Using Pipenv..."
    pipenv run python run_api.py
else
    echo "Using standard Python..."
    python run_api.py
fi
