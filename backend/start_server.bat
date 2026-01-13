@echo off
REM Windows batch script to start the API server
REM Run this from the backend directory

echo Setting up environment...
cd /d "%~dp0"

REM Set PYTHONPATH to current directory
set PYTHONPATH=%CD%

REM Check if pipenv is available
where pipenv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using Pipenv...
    pipenv run python run_api.py
) else (
    echo Using standard Python...
    python run_api.py
)

pause
