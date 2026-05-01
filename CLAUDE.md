# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Two layered projects in one tree:

1. **Quran Muaalem (upstream)** — the original `obadx/quran-muaalem` package: a Wav2Vec2-BERT + Multi-Level CTC model that transcribes Quranic recitations into phonemes and predicts tajweed/sifat properties. Shipped as three services (engine, app, UI). Owned upstream; treat its files as read-only unless explicitly editing them.
2. **MSA fine-tuning extension (in-tree work)** — adapts the upstream model to recognize Modern Standard Arabic phonemes (31 classes instead of 43 Quranic classes) using Common Voice Arabic data. Lives under `src/quran_muaalem/data/`, `src/quran_muaalem/training/`, and the `msa_*.py` / `adapt_model_for_msa.py` files in `src/quran_muaalem/modeling/`.

The three project-level guides explain each layer in depth:
- [MODEL.md](MODEL.md) — architecture, the 31-token MSA inventory, the head-resize procedure
- [TRAINING.md](TRAINING.md) — full data prep + fine-tuning pipeline
- [RUNNING.md](RUNNING.md) — install + serve the three runtime services

When the user asks about training/serving/architecture, prefer pointing at (and editing) these three files over creating new docs.

## Common Commands

All commands assume `cwd = c:\Users\moham\Projects\muaalem` and use `uv` as the package manager. The user's Python launcher is `python3.14`.

### Install

```bash
# Runtime serving (engine + app + Gradio UI)
python3.14 -m uv sync --extra engine --extra ui

# Add the training extras for MSA fine-tuning
python3.14 -m uv sync --extra training

# For tests
python3.14 -m uv sync --extra test
```

### Run the three services (in three terminals, in order)

```bash
python3.14 -m uv run quran-muaalem-engine   # port 8000, must start first
python3.14 -m uv run quran-muaalem-app      # port 8001
python3.14 -m uv run quran-muaalem-ui       # port 7860
```

Configuration is via `.env` at the repo root. The repo ships with `ACCELERATOR=cpu`, `DTYPE=float32`, and `ENGINE_URL=http://127.0.0.1:8000/predict` for CPU-only systems.

### Tests

```bash
# All tests
python3.14 -m uv run pytest

# Skip the slow / model-loading tests
python3.14 -m uv run pytest --skip-slow

# A single file or test
python3.14 -m uv run pytest tests/test_align_phonemes.py
python3.14 -m uv run pytest tests/test_align_phonemes.py::test_specific_function
```

The custom `--skip-slow` flag is defined in `tests/conftest.py`. Tests marked `@pytest.mark.slow` exercise real model loading and inference and are slow on CPU.

### MSA fine-tuning workflow

```bash
# 1. Prepare Common Voice Arabic into datasets/msa_speech/
python3.14 -m uv run python -m quran_muaalem.data.prepare_common_voice

# 2. Resize phoneme head 43 -> 31 (one-shot, produces checkpoints/msa_model_adapted/)
python3.14 -m uv run python -c "from src.quran_muaalem.modeling.adapt_model_for_msa import adapt_model_for_msa; adapt_model_for_msa()"

# 3. Quick CPU smoke test (~10-25 min)
python3.14 -m uv run python train_msa_simple.py \
    --model_name checkpoints/msa_model_adapted \
    --device cpu --epochs 1 --batch_size 1 --max_samples 100

# 4. Full training (GPU recommended)
python3.14 -m uv run python train_msa_simple.py \
    --model_name checkpoints/msa_model_adapted \
    --device cuda --epochs 20 --batch_size 4
```

## High-Level Architecture

### Runtime: three independent processes

```
browser → Gradio UI :7860  →  App (FastAPI) :8001  →  Engine (LitServe) :8000
                                                       │
                                                       └─ Wav2Vec2BertForMultilevelCTC
```

- **Engine** ([src/quran_muaalem/engine/](src/quran_muaalem/engine/)) — a LitServe wrapper around `Muaalem` ([inference.py](src/quran_muaalem/inference.py)) that loads the model and exposes `/predict`. Settings come from `EngineSettings` (Pydantic, env-driven, [engine/settings.py](src/quran_muaalem/engine/settings.py)). `engine/main.py` checks `torch.cuda.is_available()` and downgrades `accelerator` to `"cpu"` if no GPU is present.
- **App** ([src/quran_muaalem/app/](src/quran_muaalem/app/)) — FastAPI service that calls the engine over HTTP and adds search (`/search`), recitation correction (`/correct`), and transcription endpoints. Phonetic search and phonetization run on a thread-pool executor (`max_workers_*` in `AppSettings`).
- **UI** ([src/quran_muaalem/gradio_app.py](src/quran_muaalem/gradio_app.py)) — Gradio frontend that calls the app.

The app expects the engine; the UI expects the app. Always start engine → app → UI.

### The model: `Wav2Vec2BertForMultilevelCTC`

Defined in [src/quran_muaalem/modeling/modeling_multi_level_ctc.py](src/quran_muaalem/modeling/modeling_multi_level_ctc.py). The key shape is:

- A frozen Wav2Vec2-BERT encoder produces hidden states `(batch, T_enc, 1024)` at ~50 Hz.
- A `nn.ModuleDict` called `level_to_lm_head` holds one `nn.Linear(1024, vocab_size)` per "level". Levels are configured in `config.level_to_vocab_size` (e.g. `phonemes: 43`, `tajweed: ...`, `sifat properties: ...`).
- `forward()` returns a `CausalLMOutput` whose `.logits` is **a dict** keyed by level name — not a tensor. Always index it as `outputs.logits["phonemes"]`.
- During training, CTC loss is applied per level (weighted by `config.level_to_loss_weight`).

The MSA extension (see [MODEL.md](MODEL.md)) keeps the encoder frozen and replaces only the `phonemes` head with a 31-class layer, copying the first 31 weight rows from the original 43-class head as a warm start. The other heads remain in place but are unused during MSA training.

### Multi-Level CTC + lengths gotcha

When computing CTC loss manually (as in `CTCTrainer` in [training/train_msa.py](src/quran_muaalem/training/train_msa.py)), `input_lengths` must match the **actual logits time dimension `T_enc`**, not the attention-mask sum from the feature extractor side. The encoder downsamples by ~2×, and `T_enc < attention_mask.sum()` is required for CTC to be valid. Reusing the attention-mask sum will raise `Expected input_lengths to have value at most N, but got M`.

### MSA fine-tuning data flow

```
Common Voice (mp3 + tsv)
    └── prepare_common_voice.py: resample to 16kHz WAV + char→phoneme map
        └── datasets/msa_speech/{train,val,test}/*.wav  +  manifest.json

manifest.json
    └── MSAPhonemeDataset (msa_dataset.py): pads features to fixed max_features,
        tokenizes phoneme strings via MSATokenizer, pads labels to length 256
        └── DataLoader → CTCTrainer (train_msa.py)
            └── checkpoints/msa_model_v1/{best_model,checkpoint_epoch_N}/
```

The adapted checkpoint MUST contain `preprocessor_config.json` (feature extractor) in addition to `config.json` and `model.safetensors`, otherwise `MSAPhonemeDataset` fails at `AutoFeatureExtractor.from_pretrained(...)`. The current `adapt_model_for_msa.py` saves the feature extractor; if you change the adapter, preserve that step.

## Environment Notes

- **Shell**: bash and PowerShell both work, but heavy PyTorch operations (loading the 2.3 GB MSA-adapted checkpoint, full CPU training) can segfault when invoked through the bash bridge in this environment. PowerShell is the more reliable shell for long-running training/inference. Reach for `python3.14 -m uv run python <script>` from PowerShell when stability matters.
- **Path style**: prefer forward slashes in arguments (`checkpoints/msa_model_adapted`); both shells accept them.
- **Console encoding**: Windows `cp1252` will choke on Unicode arrows like `→` in `print` statements. Use ASCII (`->`) when adding stdout for Windows users, or call `sys.stdout.reconfigure(encoding='utf-8')`.
- **First run downloads**: starting the engine the first time downloads `obadx/muaalem-model-v3_2` (~660 MB) into the HF cache. Subsequent runs are local.
- **CPU expectations**: model load is 60–180 s, first-batch warm-up is another 30–60 s, then ~5–15 s per batch. Don't conclude something is hung until you've waited at least 3 minutes after "Loading pre-trained model...".
- **Model size**: the adapted MSA checkpoint is ~2.3 GB on disk because it stores all 11 multi-level heads (only `phonemes` was resized). RAM footprint at `float32` is ~3 GB.

## Project Layout (where to look)

| Concern | Path |
|---|---|
| Upstream model class | [src/quran_muaalem/modeling/modeling_multi_level_ctc.py](src/quran_muaalem/modeling/modeling_multi_level_ctc.py) |
| Upstream inference wrapper | [src/quran_muaalem/inference.py](src/quran_muaalem/inference.py) |
| Engine (LitServe) | [src/quran_muaalem/engine/](src/quran_muaalem/engine/) |
| App (FastAPI) | [src/quran_muaalem/app/](src/quran_muaalem/app/) |
| UI (Gradio) | [src/quran_muaalem/gradio_app.py](src/quran_muaalem/gradio_app.py) |
| MSA vocab / tokenizer | [src/quran_muaalem/modeling/msa_vocab.py](src/quran_muaalem/modeling/msa_vocab.py), [msa_tokenizer.py](src/quran_muaalem/modeling/msa_tokenizer.py) |
| MSA head resize | [src/quran_muaalem/modeling/adapt_model_for_msa.py](src/quran_muaalem/modeling/adapt_model_for_msa.py) |
| MSA dataset / data prep | [src/quran_muaalem/data/](src/quran_muaalem/data/) |
| MSA trainer | [src/quran_muaalem/training/train_msa.py](src/quran_muaalem/training/train_msa.py) |
| Train entry point | [train_msa_simple.py](train_msa_simple.py) |
| Pytest config | [tests/conftest.py](tests/conftest.py) |
| Runtime config | [.env](.env), [pyproject.toml](pyproject.toml) |
