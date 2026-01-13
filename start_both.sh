#!/bin/bash

echo "============================================================"
echo "Starting Both Backend and Frontend"
echo "============================================================"
echo ""
echo "This will start both services:"
echo "  1. Backend API Server (port 8001)"
echo "  2. Frontend Streamlit App (port 8501)"
echo ""
read -p "Press Enter to continue..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start backend in background
cd "$SCRIPT_DIR/backend"
gnome-terminal -- bash -c "pipenv run python run_api.py; exec bash" 2>/dev/null || \
xterm -e "cd '$SCRIPT_DIR/backend' && pipenv run python run_api.py" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd '$SCRIPT_DIR/backend' && pipenv run python run_api.py\"" 2>/dev/null || \
echo "Please start backend manually: cd backend && pipenv run python run_api.py"

# Wait a bit
sleep 3

# Start frontend in background
cd "$SCRIPT_DIR/frontend"
gnome-terminal -- bash -c "python -m streamlit run app.py; exec bash" 2>/dev/null || \
xterm -e "cd '$SCRIPT_DIR/frontend' && python -m streamlit run app.py" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd '$SCRIPT_DIR/frontend' && python -m streamlit run app.py\"" 2>/dev/null || \
echo "Please start frontend manually: cd frontend && python -m streamlit run app.py"

echo ""
echo "Both services are starting..."
echo ""
echo "Backend: http://localhost:8001"
echo "Frontend: http://localhost:8501"
echo ""
