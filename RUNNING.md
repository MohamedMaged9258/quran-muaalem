# RUNNING — Setup & Serving the System

This document covers how to install the project and run the live system (engine + REST API + Gradio UI). For training, see [TRAINING.md](TRAINING.md). For the model itself, see [MODEL.md](MODEL.md).

---

## 1. The System at a Glance

Two coexisting stacks share this repo:

### Upstream Quranic stack (3 processes)

```
   browser ─► Gradio UI :7860 ─► App (FastAPI) :8001 ─► Engine (LitServe) :8000
                                                          (Quranic multi-level model)
```

| Service | Port | Started by | Purpose |
|---|---|---|---|
| **Engine** | 8000 | `quran-muaalem-engine` | Loads the multi-level CTC model and serves `/predict`. |
| **App** | 8001 | `quran-muaalem-app` | Higher-level endpoints (search, recitation correction). Calls the engine. |
| **UI** | 7860 | `quran-muaalem-ui` | Gradio web interface. Calls the app. |

You can run any subset, but UI needs App, and App needs Engine.

### MSA stack (2 processes, fine-tuned model)

```
   browser ─► Gradio UI :7870 ─► API (FastAPI) :8010 ─► fine-tuned MSA checkpoint
```

| Service | Port | Started by | Purpose |
|---|---|---|---|
| **MSA API** | 8010 | `quran-muaalem-msa-api` | Loads the fine-tuned MSA model. Endpoints: `/transcribe`, `/align`, `/compare`, `/health`. |
| **MSA UI** | 7870 | `quran-muaalem-msa-ui` | Gradio frontend with three tabs (Transcribe, Align, Compare). |

The two stacks are independent — run either, both, or neither. They use different ports so nothing collides.

---

## 2. Prerequisites

- **OS**: Windows 11 (developed and tested), Linux/macOS should also work.
- **Python**: 3.11–3.12 (`pyproject.toml` requires `>=3.11,<3.13`).
- **Disk**: ~10 GB free (PyTorch + transformers + first-run model download).
- **Memory**: 2 GB minimum (engine uses ~1.5 GB once the model is loaded).
- **GPU**: optional. CPU works but is slow (10–30 s per 15-second clip).

---

## 3. Installation

```bash
# 1. Install uv if you don't have it
pip install uv

# 2. From the project root, install everything you'll need to serve the system
python3.14 -m uv sync --extra engine --extra ui
```

Verify the three console scripts are wired up:

```bash
python3.14 -m uv run quran-muaalem-engine --help
python3.14 -m uv run quran-muaalem-app --help
python3.14 -m uv run quran-muaalem-ui --help
```

If any of these errors out, re-run `uv sync` and check for missing dependencies.

---

## 4. Configuration (`.env`)

The repo ships with a `.env` at the project root that selects CPU mode:

```dotenv
ACCELERATOR=cpu
DTYPE=float32
ENGINE_URL=http://127.0.0.1:8000/predict
```

These are read by Pydantic settings classes:

| Variable | Read by | Default | Notes |
|---|---|---|---|
| `MODEL_NAME_OR_PATH` | engine | `obadx/muaalem-model-v3_2` | HuggingFace ID **or** a local path like `checkpoints/msa_model_v1/best_model`. |
| `ACCELERATOR` | engine | `cuda` | Set to `cpu` if you don't have a GPU. The engine also auto-falls back. |
| `DTYPE` | engine | `bfloat16` | Use `float32` on CPU. |
| `PORT` | engine | `8000` | Engine bind port. |
| `MAX_AUDIO_SECONDS` | engine | `15` | Hard cap per request. |
| `ENGINE_URL` | app | `http://0.0.0.0:8000/predict` | App's pointer to the engine. |
| `PORT` | app | `8001` | App bind port (override via env). |
| `MSA_MODEL_PATH` | msa-api | `checkpoints/msa_model_v1/best_model` | Fine-tuned MSA checkpoint. |
| `MSA_DEVICE` | msa-api | `cpu` | `cpu` or `cuda`. Auto-falls back to CPU. |
| `MSA_API_PORT` | msa-api | `8010` | MSA API bind port. |
| `MSA_API_URL` | msa-ui | `http://127.0.0.1:8010` | Where the UI looks for the API. |
| `MSA_UI_PORT` | msa-ui | `7870` | MSA UI bind port. |

For full lists see [src/quran_muaalem/engine/settings.py](src/quran_muaalem/engine/settings.py), [src/quran_muaalem/app/settings.py](src/quran_muaalem/app/settings.py), and [src/quran_muaalem/msa/settings.py](src/quran_muaalem/msa/settings.py).

### Use a fine-tuned MSA checkpoint

After running the training pipeline (see [TRAINING.md](TRAINING.md)), point the engine at your local checkpoint:

```dotenv
MODEL_NAME_OR_PATH=checkpoints/msa_model_v1/best_model
```

Restart the engine to pick up the change.

---

## 5. Starting the Services

Open **three terminals**, all in the project root.

### Terminal 1 — Engine (must start first)

```bash
python3.14 -m uv run quran-muaalem-engine
```

Wait for:

```
INFO:     Started server process [XXXX]
INFO:     Application startup complete.
Swagger UI is available at http://0.0.0.0:8000/docs
```

The first run downloads the model (~660 MB) and may take several minutes. Subsequent runs use the local HuggingFace cache.

### Terminal 2 — App

```bash
python3.14 -m uv run quran-muaalem-app
```

Wait for:

```
INFO:     Application startup complete.
```

### Terminal 3 — UI

```bash
python3.14 -m uv run quran-muaalem-ui
```

Wait for:

```
Running on local URL:  http://127.0.0.1:7860
```

### Health checks

| URL | Expected |
|---|---|
| http://localhost:8000/health | `{"status":"ok"}` |
| http://localhost:8001/health | `{"status":"healthy","engine_status":"ok"}` |
| http://localhost:7860 | Gradio interface renders |

### Starting the MSA stack

In two more terminals (the MSA stack is independent of the Quranic one):

```bash
# Terminal 4 — MSA API (loads the fine-tuned checkpoint)
python3.14 -m uv run quran-muaalem-msa-api

# Terminal 5 — MSA UI
python3.14 -m uv run quran-muaalem-msa-ui
```

| URL | Expected |
|---|---|
| http://localhost:8010/health | `{"status":"ok","model_path":"…","device":"cpu"}` |
| http://localhost:8010/docs | FastAPI interactive docs |
| http://localhost:7870 | MSA Gradio UI with three tabs |

The MSA API has three endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/transcribe` | `audio` (file) | `{"phonemes": "..."}` |
| POST | `/align` | `audio` (file) | phonemes + per-phoneme `start`/`end`/`confidence` |
| POST | `/compare` | `audio` (file), `expected_text` (form) | full diff with PER, substitutions, inserts, deletes |

Phonemization for `/compare` follows the project's training-time mapping, which **drops alif `ا` and alif maksura `ى`** so predicted and expected sequences are directly comparable. See [src/quran_muaalem/msa/phonemize.py](src/quran_muaalem/msa/phonemize.py) for the exact mapping.

---

## 6. Using the System

### Via the Gradio UI

1. Open http://localhost:7860.
2. Upload an audio file (or record from the microphone).
3. Run the analysis to see the phonetic transcription, alignment, and correction output.

### Via the REST API

Interactive docs: http://localhost:8001/docs.

```bash
# Phonetic search
curl -X POST "http://localhost:8001/search?phonetic_text=bismi"

# Recitation correction (multipart upload)
curl -X POST "http://localhost:8001/correct" \
     -F "audio=@my_clip.wav" \
     -F "expected_text=بِسْمِ ٱللَّهِ"
```

### Via the engine directly

If you only need raw audio → phonemes:

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "audio=@my_clip.wav"
```

The engine's docs live at http://localhost:8000/docs.

### Via the Python client

[client.py](client.py) at the project root is a small example that hits the running services from Python.

---

## 7. Stopping the Services

`Ctrl+C` in each terminal. Each process shuts down independently:

```
INFO:     Shutting down
INFO:     Shutdown complete
```

If a port is stuck in use afterwards (Windows often holds onto sockets briefly):

```powershell
# Find which PID is on the port
netstat -ano | findstr :8000

# Kill it
taskkill /PID <PID> /F
```

---

## 8. Common Errors

| Error | Why | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Dependencies not installed. | `python3.14 -m uv sync --extra engine --extra ui` |
| `Address already in use` on 8000 / 8001 / 7860 | A previous run is still bound. | Kill the old PID (see §7), or change the port via env. |
| `CUDA GPUs are not available` | No GPU / no driver. | Already auto-falls back to CPU; or set `ACCELERATOR=cpu` in `.env`. |
| App returns 502 / engine errors | Engine isn't up yet. | Always start the engine first and wait for "Application startup complete." |
| UI loads but predictions hang | App can't reach engine. | Check `ENGINE_URL` in `.env` matches the engine's actual host:port. |
| Engine OOM at startup | `bfloat16` not supported on your CPU, or model loaded in `float16`. | Set `DTYPE=float32` in `.env`. |

---

## 9. Project Layout (Runtime Side)

```
muaalem/
├── src/quran_muaalem/
│   ├── engine/             # LitServe model server (port 8000)
│   │   ├── main.py         # quran-muaalem-engine entry point
│   │   ├── serve.py        # QuranMuaalemAPI: load + predict
│   │   └── settings.py     # EngineSettings (env-driven)
│   ├── app/                # REST API (port 8001)
│   │   ├── main.py         # quran-muaalem-app entry point
│   │   ├── serve.py        # FastAPI routes
│   │   └── settings.py     # AppSettings
│   ├── gradio_app.py       # quran-muaalem-ui entry point (port 7860)
│   ├── inference.py        # Thin wrapper used by the engine
│   ├── decode.py           # CTC greedy/beam decoding helpers
│   └── modeling/           # Model class + MSA adapters
├── client.py               # Example Python client
├── pyproject.toml          # Dependencies + console scripts
└── .env                    # Runtime configuration
```

---

## 10. Quick Reference

```bash
# Install
python3.14 -m uv sync --extra engine --extra ui

# Quranic stack (3 terminals)
python3.14 -m uv run quran-muaalem-engine    # :8000
python3.14 -m uv run quran-muaalem-app       # :8001
python3.14 -m uv run quran-muaalem-ui        # :7860

# MSA stack (2 terminals)
python3.14 -m uv run quran-muaalem-msa-api   # :8010
python3.14 -m uv run quran-muaalem-msa-ui    # :7870

# Open
http://localhost:7860      # Quranic UI
http://localhost:7870      # MSA UI
http://localhost:8001/docs # Quranic App docs
http://localhost:8010/docs # MSA API docs
```
