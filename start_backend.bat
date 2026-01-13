@echo off
echo ============================================================
echo Starting Healthcare Data Generation Backend API
echo ============================================================
echo.

cd /d "%~dp0backend"

echo Checking for Pipenv...
pipenv --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Pipenv not found. Please install Pipenv first.
    echo Install with: pip install pipenv
    pause
    exit /b 1
)

echo Installing dependencies (if needed)...
pipenv install

echo.
echo Starting API server...
echo.
pipenv run python run_api.py

pause
