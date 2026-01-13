#!/usr/bin/env python3
"""
run_api.py

Simple script to run the API server with proper path setup.
Run this from the backend directory.
"""

import sys
import os
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Get the backend directory (where this script is located)
backend_dir = Path(__file__).parent.absolute()

# Add backend directory to Python path
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Set PYTHONPATH environment variable for subprocesses
os.environ['PYTHONPATH'] = str(backend_dir)

# Now import and run
if __name__ == "__main__":
    import uvicorn
    from config.settings import API_CONFIG
    
    print("\n" + "="*60)
    print("HEALTHCARE DATA GENERATION API")
    print("="*60)
    print(f"[*] Starting server on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"[*] Max generated rows: {API_CONFIG['max_generated_rows']}")
    print(f"[*] Request timeout: {API_CONFIG['timeout_seconds']}s")
    print("\n[?] API Documentation:")
    print(f"    Swagger UI: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
    print(f"    ReDoc: http://{API_CONFIG['host']}:{API_CONFIG['port']}/redoc")
    print("\n[>] Example queries:")
    print("    - 'Generate 100 patients with diabetes'")
    print("    - 'ICU patients with sepsis and cardiovascular disease'")
    print("    - 'Elderly patients with multiple comorbidities'")
    print("\n[!] Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run uvicorn with the app string (it will import correctly now)
    uvicorn.run(
        "src.api.app:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=False,  # Set to True for development with auto-reload
        log_level="info"
    )
