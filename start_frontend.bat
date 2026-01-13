@echo off
echo ============================================================
echo Starting Healthcare Data Generation Frontend
echo ============================================================
echo.

cd /d "%~dp0frontend"

echo Checking for Streamlit...
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting Streamlit app...
echo.
python -m streamlit run app.py

pause
