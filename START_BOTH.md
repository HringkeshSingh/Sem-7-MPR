# 🚀 Quick Start Guide - Run Both Frontend & Backend

This guide will help you run both the backend API and frontend Streamlit app simultaneously.

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Pipenv** installed (for backend) or use `pip` with virtual environment
3. **All dependencies installed**

---

## 🎯 Quick Start (Recommended)

### Step 1: Install Dependencies

**Backend:**

```bash
cd backend
pipenv install
```

**Frontend:**

```bash
cd frontend
pip install -r requirements.txt
```

### Step 2: Run Both Services

You'll need **TWO terminal windows** - one for backend, one for frontend.

#### Terminal 1 - Backend (API Server)

```bash
cd backend
pipenv run python run_api.py
```

Wait for: `🚀 Starting server on http://0.0.0.0:8001`

#### Terminal 2 - Frontend (Streamlit App)

```bash
cd frontend
python -m streamlit run app.py
```

Wait for: `You can now view your Streamlit app in your browser.`

---

## 🌐 Access the Application

Once both are running:

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

---

## 🔧 Alternative: Using Scripts

### Windows (PowerShell)

**Backend:**

```powershell
cd backend
.\start_server.bat
```

**Frontend:**

```powershell
cd frontend
python -m streamlit run app.py
```

### macOS/Linux

**Backend:**

```bash
cd backend
chmod +x start_server.sh
./start_server.sh
```

**Frontend:**

```bash
cd frontend
python -m streamlit run app.py
```

---

## ✅ Verify Everything is Working

1. **Check Backend**: Open http://localhost:8001/health

   - Should return: `{"status": "healthy"}`

2. **Check Frontend**: Open http://localhost:8501

   - Should show the Healthcare Data Generation System interface

3. **Check Connection**: In the frontend sidebar, you should see "✅ API Connected"

---

## 🐛 Troubleshooting

### Backend Issues

**Error: "No module named 'src'"**

```bash
# Make sure you're in the backend directory
cd backend
pipenv run python run_api.py
```

**Error: Port 8001 already in use**

```bash
# Find and kill the process using port 8001
# Windows:
netstat -ano | findstr :8001
taskkill /PID <PID> /F

# Then try again
```

**Error: Missing dependencies**

```bash
cd backend
pipenv install
```

### Frontend Issues

**Error: "streamlit not found"**

```bash
cd frontend
pip install streamlit requests pandas plotly numpy
```

**Error: "Cannot connect to API"**

- Make sure backend is running on port 8001
- Check that `API_BASE_URL` in `frontend/app.py` is `http://localhost:8001`
- Verify backend health: http://localhost:8001/health

**Error: Port 8501 already in use**

```bash
# Use a different port
python -m streamlit run app.py --server.port 8502
```

---

## 📝 Step-by-Step Detailed Instructions

### 1. First Time Setup

**Backend:**

```bash
# Navigate to backend
cd backend

# Install dependencies (if using Pipenv)
pipenv install

# OR if using pip with venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**Frontend:**

```bash
# Navigate to frontend
cd frontend

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Services

**Terminal 1 - Start Backend:**

```bash
cd backend
pipenv run python run_api.py
```

You should see:

```
============================================================
HEALTHCARE DATA GENERATION API
============================================================
🚀 Starting server on http://0.0.0.0:8001
...
```

**Terminal 2 - Start Frontend:**

```bash
cd frontend
python -m streamlit run app.py
```

You should see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### 3. Using the Application

1. Open your browser to http://localhost:8501
2. The frontend will automatically connect to the backend
3. Check the sidebar - you should see "✅ API Connected"
4. Start using the Query & RAG tab to generate data!

---

## 🎨 Development Mode

### Backend with Auto-Reload

Edit `backend/run_api.py` and change:

```python
reload=False,  # Change to True
```

Then restart the backend.

### Frontend with Auto-Reload

Streamlit automatically reloads on file changes by default.

---

## 🛑 Stopping the Services

- **Backend**: Press `Ctrl+C` in Terminal 1
- **Frontend**: Press `Ctrl+C` in Terminal 2

---

## 💡 Pro Tips

1. **Keep both terminals visible** - Easier to see errors
2. **Check backend logs** - If frontend shows errors, check backend terminal
3. **Use browser DevTools** - F12 to see network requests
4. **Clear browser cache** - If UI seems outdated, hard refresh (Ctrl+F5)

---

## 📞 Need Help?

1. Check backend logs in Terminal 1
2. Check frontend logs in Terminal 2
3. Verify both services are running:
   - Backend: http://localhost:8001/health
   - Frontend: http://localhost:8501
4. Check the README files:
   - `backend/README.md`
   - `frontend/README.md`

---

## ✨ Quick Command Reference

```bash
# Backend
cd backend && pipenv run python run_api.py

# Frontend
cd frontend && python -m streamlit run app.py

# Check backend health
curl http://localhost:8001/health

# View API docs
# Open http://localhost:8001/docs in browser
```

Happy coding! 🎉
