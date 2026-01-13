#!/bin/bash

echo "============================================================"
echo "Starting Healthcare Data Generation Backend API"
echo "============================================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/backend"

# Check for Pipenv
if ! command -v pipenv &> /dev/null; then
    echo "ERROR: Pipenv not found. Please install Pipenv first."
    echo "Install with: pip install pipenv"
    exit 1
fi

echo "Installing dependencies (if needed)..."
pipenv install

echo ""
echo "Starting API server..."
echo ""
pipenv run python run_api.py
