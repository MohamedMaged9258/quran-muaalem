# Quran Muaalem - Quick Start Guide

## Prerequisites

⚠️ **IMPORTANT:** Before starting, ensure:
- **Windows 11** (tested on this version)
- **Python 3.11+** (check: `python3.14 --version`)
- **~10GB free disk space** (for PyTorch and dependencies)
- **15+ minutes for first-time setup** (downloads ~7GB)

## Installation (Step-by-Step)

### Step 1: Verify Python and Install uv

```bash
# Check Python version
python3.14 --version

# Install uv (package manager)
pip install uv

# Verify uv installation
uv --version
```

### Step 2: Install Project Dependencies

```bash
cd c:\Users\moham\Projects\muaalem

# Install all dependencies (engine, app, and UI)
python3.14 -m uv sync --extra engine --extra ui
```

**⏱️ Wait for this to complete** (5-20 minutes depending on internet speed)
- You'll see downloads for: PyTorch, Gradio, Librosa, Transformers, etc.
- It will say "Installed X packages" when done

### Step 3: Verify Installation

```bash
# Test that the project is properly installed
python3.14 -m uv run quran-muaalem-engine --help
python3.14 -m uv run quran-muaalem-app --help
python3.14 -m uv run quran-muaalem-ui --help
```

If all three commands show help text without errors, you're ready! ✅

## Running the Project (The Exact Steps)

### Option A: Full System with UI (Recommended)

Open **3 separate terminal windows** and run these commands in order:

**Terminal 1 - Engine (Run this FIRST)**
```bash
cd c:\Users\moham\Projects\muaalem
python3.14 -m uv run quran-muaalem-engine
```

**Wait for output:**
```
INFO:     Started server process [XXXX]
INFO:     Application startup complete.
Swagger UI is available at http://0.0.0.0:8000/docs
```

**Terminal 2 - App (Run this SECOND after Engine is ready)**
```bash
cd c:\Users\moham\Projects\muaalem
python3.14 -m uv run quran-muaalem-app
```

**Wait for output:**
```
INFO:     Application startup complete.
```

**Terminal 3 - UI (Run this THIRD after App is ready)**
```bash
cd c:\Users\moham\Projects\muaalem
python3.14 -m uv run quran-muaalem-ui
```

**Wait for output:**
```
Running on local URL:  http://127.0.0.1:7860
```

### Step 4: Verify Everything is Running

Open your browser and check:

1. **Engine Health**: http://localhost:8000/health
   - Should show: `{"status":"ok"}`

2. **App Health**: http://localhost:8001/health
   - Should show: `{"status":"healthy","engine_status":"ok"}`

3. **Gradio UI**: http://localhost:7860
   - Should show the interactive web interface

✅ If all three show green and respond, you're good to go!

### Option B: API Only (No UI)

If you only want to test the REST API:

```bash
# Terminal 1
python3.14 -m uv run quran-muaalem-engine

# Terminal 2
python3.14 -m uv run quran-muaalem-app
```

Then access the API docs at: http://localhost:8001/docs

## Common Errors & Solutions

### Error: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
python3.14 -m uv sync --extra engine --extra ui
```
Make sure this completes without errors before running the services.

### Error: "Address already in use" (port 8000, 8001, or 7860)
**Solution:** Kill the process using that port:
```bash
# Windows PowerShell
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force

# Or find and kill specific port
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Error: "CUDA GPUs are not available"
**This is expected!** The system automatically falls back to CPU mode.
- CPU inference is slower (~10-30s per audio) but works fine for testing
- The `.env` file is pre-configured for CPU with float32 dtype

### Error: Engine or App won't start
**Common causes and fixes:**
1. **Engine not started first** → Start Terminal 1 (Engine) before Terminal 2
2. **Port still in use** → Kill old processes (see error above)
3. **Missing dependencies** → Run `python3.14 -m uv sync --extra engine --extra ui` again
4. **Out of disk space** → Need ~10GB free

## Using the System

### Via Gradio UI (Easiest)
1. Go to http://localhost:7860
2. Upload an audio file or paste phonetic text
3. Click analyze to get results

### Via API (Advanced)
```bash
# Search in Quran by phonetic text
curl -X POST "http://localhost:8001/search?phonetic_text=bismi"

# Correct a recitation (upload audio)
# Use http://localhost:8001/docs for interactive testing
```

### Via API Docs (Recommended for Testing)
Visit http://localhost:8001/docs and use the "Try it out" buttons

## Stopping the Services

Press **Ctrl+C** in each terminal to stop:
```
INFO:     Shutting down
INFO:     Shutdown complete
```

## Project Structure

```
muaalem/
├── src/quran_muaalem/
│   ├── engine/              # ML model server
│   ├── app/                 # REST API backend
│   ├── gradio_app.py        # Web UI
│   ├── inference.py         # Model wrapper
│   └── modeling/            # Neural architecture
├── pyproject.toml           # Dependencies
├── .env                     # Configuration (CPU mode enabled)
└── QUICKSTART.md            # This file
```

## Key Components

| Component | Port | Status Check | Purpose |
|-----------|------|--------------|---------|
| **Engine** | 8000 | http://localhost:8000/health | Loads AI model, converts audio to phonemes |
| **App** | 8001 | http://localhost:8001/health | REST API for search & correction |
| **UI** | 7860 | http://localhost:7860 | Web interface |

## System Requirements

- **Memory**: 2GB minimum (Engine uses ~1.5GB when loaded)
- **CPU**: Any modern processor (inference is slow on CPU, takes 10-30s per audio)
- **Disk**: 10GB for installation, 2GB for model cache
- **Network**: Internet needed for first-time model download

## Notes

- All services run on **CPU** by default (no GPU required)
- Model: `obadx/muaalem-model-v3_2` (~660MB)
- Inference speed: ~10-30 seconds per 15-second audio on CPU
- For production/faster inference, GPU acceleration is recommended

## Next Steps

1. **Explore the UI**: http://localhost:7860 - Upload audio, test functionality
2. **Test the API**: http://localhost:8001/docs - Try different endpoints
3. **Fine-tune**: Read `src/quran_muaalem/modeling/` to understand architecture

## More Information

- **Model Hub**: https://huggingface.co/obadx/muaalem-model-v3_2
- **Dataset**: https://huggingface.co/datasets/obadx/muaalem-annotated-v3
- **GitHub**: https://github.com/obadx/quran-muaalem
- **Paper**: https://arxiv.org/abs/2509.00094
- **Discord Community**: https://discord.gg/hJWW6fCH
