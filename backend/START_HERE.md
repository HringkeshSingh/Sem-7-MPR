# 🚀 How to Start the Application

## Quick Start (Easiest Method)

### Step 1: Navigate to backend directory

```bash
cd backend
```

### Step 2: Install dependencies (if not already installed)

```bash
# Using Pipenv (recommended)
pipenv install

# OR using pip with venv
pip install -r requirements.txt
```

### Step 3: Run the application

**Option A: Using the simple script (RECOMMENDED)**

```bash
# With Pipenv
pipenv run python run_api.py

# OR with standard Python (if in venv)
python run_api.py
```

**Option B: Using the original script**

```bash
pipenv run python scripts/08_run_api.py
```

**Option C: Using Uvicorn directly**

```bash
# Make sure you're in the backend directory first!
cd backend

# Set PYTHONPATH (Windows)
set PYTHONPATH=%CD%
pipenv run uvicorn src.api.app:app --host 0.0.0.0 --port 8001

# Set PYTHONPATH (macOS/Linux)
export PYTHONPATH=$(pwd)
pipenv run uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

**Option D: Using the batch/shell script**

```bash
# Windows
start_server.bat

# macOS/Linux
chmod +x start_server.sh
./start_server.sh
```

## ✅ Verify It's Running

After starting, you should see:

```
============================================================
HEALTHCARE DATA GENERATION API
============================================================
🚀 Starting server on http://0.0.0.0:8001
...
```

Then access:

- **API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## 🔧 Troubleshooting

### Error: "No module named 'src'"

**Solution 1: Use the new run_api.py script**

```bash
cd backend
pipenv run python run_api.py
```

**Solution 2: Set PYTHONPATH manually**

```bash
# Windows (Command Prompt)
cd backend
set PYTHONPATH=%CD%
python scripts/08_run_api.py

# Windows (PowerShell)
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/08_run_api.py

# macOS/Linux
cd backend
export PYTHONPATH=$(pwd)
python scripts/08_run_api.py
```

**Solution 3: Run from the correct directory**
Make sure you're ALWAYS in the `backend` directory when running commands:

```bash
cd backend  # Always do this first!
pipenv run python run_api.py
```

### Error: "No module named 'langchain_community'"

Install missing dependencies:

```bash
cd backend
pipenv install langchain-community langchain-core langchain-text-splitters chromadb sentence-transformers
```

### Error: Port 8001 already in use

Use a different port:

```bash
pipenv run uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

## 📝 Important Notes

1. **Always run from the `backend` directory** - This is crucial!
2. **Use virtual environment** - Either Pipenv or venv
3. **Set PYTHONPATH** - The new `run_api.py` does this automatically
4. **Install all dependencies** - Run `pipenv install` first

## 🎯 Recommended Command

```bash
cd backend
pipenv install
pipenv run python run_api.py
```

That's it! The server will start and be available at http://localhost:8001
