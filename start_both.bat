@echo off
echo ============================================================
echo Starting Both Backend and Frontend
echo ============================================================
echo.
echo This will open two new windows:
echo   1. Backend API Server (port 8001)
echo   2. Frontend Streamlit App (port 8501)
echo.
echo Press any key to continue...
pause >nul

start "Backend API" cmd /k "cd /d %~dp0backend && pipenv run python run_api.py"
timeout /t 3 /nobreak >nul
start "Frontend Streamlit" cmd /k "cd /d %~dp0frontend && python -m streamlit run app.py"

echo.
echo Both services are starting in separate windows.
echo.
echo Backend: http://localhost:8001
echo Frontend: http://localhost:8501
echo.
echo Press any key to exit this window (services will keep running)...
pause >nul
