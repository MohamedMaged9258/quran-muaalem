# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Two layered projects in one tree:

1. **Quran Muaalem (upstream)** — the original `obadx/quran-muaalem` package: a Wav2Vec2-BERT + Multi-Level CTC model that transcribes Quranic recitations into phonemes and predicts tajweed/sifat properties. Shipped as three services (engine, app, UI). Owned upstream; treat its files as read-only unless explicitly editing them.
2. **MSA fine-tuning extension (in-tree work)** — adapts the upstream model to recognize Modern Standard Arabic phonemes (**35 classes** instead of the upstream 43 Quranic classes) using Common Voice Arabic. Lives under `src/quran_muaalem/data/`, `src/quran_muaalem/training/`, `src/quran_muaalem/modeling/msa_*` + `adapt_model_for_msa.py`, and the serving stack at `src/quran_muaalem/msa/`.

The four project-level guides explain each layer in depth:
- [MODEL.md](MODEL.md) — architecture, the 35-token MSA inventory, the head-resize procedure
- [DATASET.md](DATASET.md) — Common Voice Arabic download, extraction, and manifest preparation
- [TRAINING.md](TRAINING.md) — fine-tuning pipeline (assumes the manifest is ready)
- [RUNNING.md](RUNNING.md) — install + serve the runtime services (upstream Quranic + MSA stacks)

When the user asks about training/serving/architecture, prefer pointing at (and editing) these four files over creating new docs.

## Common Commands

All commands assume `cwd = c:\Users\moham\Projects\muaalem` and use `uv` as the package manager. The user's Python launcher is `python3.14`.

### Install

```bash
# Runtime serving (engine + app + Gradio UI + MSA UI). The `ui` extra now
# includes httpx because msa/ui.py talks to the MSA API over HTTP.
python3.14 -m uv sync --extra engine --extra ui

# Add the training extras for MSA fine-tuning
python3.14 -m uv sync --extra training

# For tests
python3.14 -m uv sync --extra test
```

### Run the upstream Quranic stack (three terminals, in order)

```bash
python3.14 -m uv run quran-muaalem-engine   # port 8000, must start first
python3.14 -m uv run quran-muaalem-app      # port 8001
python3.14 -m uv run quran-muaalem-ui       # port 7860
```

### Run the MSA stack (two terminals, independent of the Quranic one)

```bash
python3.14 -m uv run quran-muaalem-msa-api  # port 8010, FastAPI + fine-tuned model
python3.14 -m uv run quran-muaalem-msa-ui   # port 7870, Gradio UI (single page)
```

Configuration is via `.env` at the repo root. The shipped values are:

```dotenv
ACCELERATOR=cpu
DTYPE=float32
ENGINE_URL=http://127.0.0.1:8000/predict
MSA_MODEL_PATH=checkpoints/msa_model_v2/best_model
MSA_DEVICE=cpu
```

`MSA_MODEL_PATH` overrides the default in [src/quran_muaalem/msa/settings.py](src/quran_muaalem/msa/settings.py) (which still names `msa_model_v1` for backward compat) — keep `.env` in sync with whatever checkpoint dir actually exists.

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

# 2. Resize phoneme head 43 -> 35 (one-shot, produces checkpoints/msa_model_adapted/)
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

### Runtime: two independent stacks

```
# Quranic stack (upstream)
browser → Gradio UI :7860  →  App (FastAPI) :8001  →  Engine (LitServe) :8000
                                                       └─ Wav2Vec2BertForMultilevelCTC

# MSA stack (this fork)
browser → Gradio UI :7870  →  MSA API (FastAPI) :8010
                              └─ MSAInference + fine-tuned checkpoint
```

- **Engine** ([src/quran_muaalem/engine/](src/quran_muaalem/engine/)) — a LitServe wrapper around `Muaalem` ([inference.py](src/quran_muaalem/inference.py)) that loads the model and exposes `/predict`. Settings come from `EngineSettings` (Pydantic, env-driven, [engine/settings.py](src/quran_muaalem/engine/settings.py)). `engine/main.py` checks `torch.cuda.is_available()` and downgrades `accelerator` to `"cpu"` if no GPU is present.
- **App** ([src/quran_muaalem/app/](src/quran_muaalem/app/)) — FastAPI service that calls the engine over HTTP and adds search (`/search`), recitation correction (`/correct`), and transcription endpoints. Phonetic search and phonetization run on a thread-pool executor (`max_workers_*` in `AppSettings`).
- **UI** ([src/quran_muaalem/gradio_app.py](src/quran_muaalem/gradio_app.py)) — Gradio frontend that calls the app.
- **MSA API** ([src/quran_muaalem/msa/api.py](src/quran_muaalem/msa/api.py)) — FastAPI service. Endpoints: `/health`, `/transcribe`, `/align`, `/compare`, `/debug`. Loads the fine-tuned 35-class checkpoint via `MSAInference`.
- **MSA UI** ([src/quran_muaalem/msa/ui.py](src/quran_muaalem/msa/ui.py)) — single-page Gradio UI. Uses `type="filepath"` on `gr.Audio` and ships raw bytes to the API, so it does not need ffmpeg in the round-trip (microphone recordings come out as WAV; MP3 upload still requires ffmpeg system-wide for librosa's audioread fallback).

The Quranic app expects the engine; the Quranic UI expects the app. The MSA UI expects the MSA API. The two stacks share no runtime state.

### The model: `Wav2Vec2BertForMultilevelCTC`

Defined in [src/quran_muaalem/modeling/modeling_multi_level_ctc.py](src/quran_muaalem/modeling/modeling_multi_level_ctc.py). The key shape is:

- A Wav2Vec2-BERT encoder produces hidden states `(batch, T_enc, 1024)` at ~50 Hz.
- A `nn.ModuleDict` called `level_to_lm_head` holds one `nn.Linear(1024, vocab_size)` per "level". Levels are configured in `config.level_to_vocab_size` (e.g. `phonemes: 43`, `tajweed: ...`, `sifat properties: ...`).
- `forward()` returns a `CausalLMOutput` whose `.logits` is **a dict** keyed by level name — not a tensor. Always index it as `outputs.logits["phonemes"]`.
- During training, CTC loss is applied per level (weighted by `config.level_to_loss_weight`).

The MSA extension (see [MODEL.md](MODEL.md)) replaces the `phonemes` head with a **35-class** layer (copying the first 35 weight rows from the original 43-class head as a warm start). The size is read from `MSA_PHONEME_COUNT` in [msa_vocab.py](src/quran_muaalem/modeling/msa_vocab.py), so adding/removing tokens automatically resizes the head when you re-run `adapt_model_for_msa()`.

At training time, [`load_model_for_msa`](src/quran_muaalem/training/train_msa.py) **enforces** the encoder freeze: every parameter is set to `requires_grad=False`, then only the phoneme head is re-enabled. AdamW is built from the trainable subset, so no optimizer state is allocated for the ~605 M frozen params (this is what makes 4 GB GPU training feasible). The other heads remain in place but are unused during MSA training.

### MSA phoneme inventory (35 tokens — current)

- **28 consonants**: the canonical Arabic alphabet, including the four emphatic / pharyngealized consonants `ص ض ط ظ`.
- **5 vowels / diacritics**: `َ` (fatha), `ُ` (damma), `ِ` (kasra), `ْ` (sukun), `ة` (ta marbuta).
- **2 special**: `[PAD]=0` (also the CTC blank), `[UNK]=1`.

`ا` (alif) and `ى` (alif maksura) are intentionally **dropped** during phonemization (both at training-data prep and at inference comparison time) so predicted and expected sequences are directly comparable. See [src/quran_muaalem/msa/phonemize.py](src/quran_muaalem/msa/phonemize.py).

### Multi-Level CTC + lengths gotcha

When computing CTC loss manually (as in `CTCTrainer` in [training/train_msa.py](src/quran_muaalem/training/train_msa.py)), `input_lengths` must match the **actual logits time dimension `T_enc`**, not the attention-mask sum from the feature extractor side. The encoder downsamples by ~2×, and `T_enc < attention_mask.sum()` is required for CTC to be valid. Reusing the attention-mask sum will raise `Expected input_lengths to have value at most N, but got M`.

### Dtype handling

The model is loaded in **`bfloat16` on CUDA, `float32` on CPU** (see `load_model_for_msa`). The trainer's `_forward_loss` casts inputs to `model.dtype` before the forward pass — without it, fp32 audio features hitting bf16 weights raises `RuntimeError: expected scalar type Float but found BFloat16`. CTC log-probs are explicitly upcast to fp32 because `nn.CTCLoss` doesn't support fp16/bf16. Don't add `torch.autocast` — it's redundant when model and inputs already share a dtype.

### MSA fine-tuning data flow

```
Common Voice (mp3 + tsv)
    └── prepare_common_voice.py: resample to 16kHz WAV + char→phoneme map
        └── datasets/msa_speech/{train,val,test}/*.wav  +  manifest.json

manifest.json
    └── MSAPhonemeDataset (msa_dataset.py): pads features to fixed max_features,
        tokenizes phoneme strings via MSATokenizer, pads labels to length 256
        └── DataLoader → CTCTrainer (train_msa.py)
            └── checkpoints/msa_model_vN/{best_model,checkpoint_epoch_N}/
                └── MSAInference (msa/inference.py) loads best_model for serving
```

Any checkpoint we serve or train from MUST contain `preprocessor_config.json` (feature extractor) alongside `config.json` and `model.safetensors`, otherwise `AutoFeatureExtractor.from_pretrained(...)` fails. Both `adapt_model_for_msa.py` and `CTCTrainer._save_pretrained` save the feature extractor — if you change either, preserve that step. `MSAPhonemeDataset` and `MSAInference` both pre-validate the path and fail with a clear `FileNotFoundError` instead of letting transformers misinterpret a missing local dir as a HuggingFace repo id and surface a misleading 401.

The "looks local" heuristic in `MSAPhonemeDataset` checks for `\`, leading `.`, leading `/`, leading `checkpoints`, or a Windows drive letter — plain HF ids like `obadx/muaalem-model-v3_2` are passed through. Don't tighten the heuristic without verifying both code paths.

## Decisions Made and Why (for future sessions)

These are non-obvious calls that came out of past debugging — they're not derivable from the code alone, so future-you will want them.

- **35-class inventory, not 31.** The earlier 31-class version dropped `ص ض ط ظ`, which silently mapped them to `[UNK]` for ~10–15% of training labels. We expanded to 35 (28 + 5 + 2). Bumping the inventory invalidates any pre-existing `checkpoints/msa_model_adapted/` and `msa_model_v*/` — the head dimension changes, so they must be re-run.
- **Encoder always frozen.** It was pre-trained on 53k hours; fine-tuning it on ~17 h of MSA hurts more than it helps. Plus optimizer state for 605 M params doesn't fit on a 4 GB GPU. Don't unfreeze without a strong reason.
- **Inputs cast to `model.dtype` in `_forward_loss`.** The bf16/fp32 mismatch is silent in Python but blows up inside layer_norm. We deliberately do NOT use `torch.autocast` — it's redundant when model + inputs share a dtype, and it hid the dtype bug for a long time.
- **Label-length clip uses `shape[1]`.** `features["input_features"]` is `(batch=1, T_feat, 160)` *before* the `squeeze(0)` further down. An older bug used `shape[0]` (always 1), so `max_label_len = max(1, 0-5) = 1` and **every label was clipped to a single phoneme**. Symptom: empty greedy decode regardless of training duration. Old `msa_model_v1` and the in-flight `msa_model_v2` are both products of this bug — unsalvageable.
- **`MSA_MODEL_PATH` lives in `.env`, default in `settings.py` is stale.** The shipped `.env` overrides to whatever checkpoint actually exists (currently `msa_model_v2/best_model`, but that one is invalid and will be replaced). Keep the `.env` value in sync after each retrain rather than chasing the default.
- **MSA UI uses `gr.Audio(type="filepath")`, not `"numpy"`.** `type="numpy"` runs Gradio's pydub/ffmpeg path; `type="filepath"` ships the raw upload to the API, which decodes via librosa. Lets the system work without ffmpeg as long as inputs are WAV (microphone records as WAV; MP3 upload still needs system ffmpeg).
- **`MSAInference._logits` was promoted to a `diagnostics()` method.** The `/debug` endpoint uses it. Don't reach back into `_logits` from elsewhere — extend `diagnostics()` instead.
- **`MSASettings` uses `protected_namespaces=()`.** Pydantic v2 reserves `model_*` field names; the user-facing setting is `model_path`, so we silence the warning rather than rename to something less natural.
- **`pyproject.toml` `requires-python = ">=3.11,<3.15"`.** The user's launcher is `python3.14`. The earlier `<3.13` cap rejected installs on this machine.
- **`ui` extra now includes `httpx`.** `msa/ui.py` calls the API over HTTP, so the UI extra alone has to be enough to launch the MSA UI. Don't drop it.
- **Empty greedy decode after few epochs is *partly* expected even on a clean run.** CTC starts with near-100% blank predictions and gradually commits to non-blank tokens. `POST /debug` returns `blank_ratio`; values close to 1.0 mean "still early in training," not "broken." But if `blank_ratio` is high *and* the model has trained for many epochs, suspect a vocab/dataset bug.
- **Old stub files at repo root were deleted.** `gradio_app.py` was a one-line broken stub (a path string, not Python); `client.py` was the LitServe-autogenerated `{"input": 4.0}` JSON example, which doesn't match the real multipart `/predict` API. Don't recreate them — the real UI lives at `src/quran_muaalem/gradio_app.py`.

## Environment Notes

- **Shell**: bash and PowerShell both work, but heavy PyTorch operations (loading the 2.3 GB MSA-adapted checkpoint, full CPU training) can segfault when invoked through the bash bridge in this environment. PowerShell is the more reliable shell for long-running training/inference. Reach for `python3.14 -m uv run python <script>` from PowerShell when stability matters.
- **Path style**: prefer forward slashes in arguments (`checkpoints/msa_model_adapted`); both shells accept them.
- **Console encoding**: Windows `cp1252` will choke on Unicode emoji (✅, ❌) and arrows (`→`) in `print` statements. The previous emoji prints in `prepare_common_voice.py` and `msa_tokenizer.py` were ASCII-fied. When adding stdout, use ASCII or call `sys.stdout.reconfigure(encoding='utf-8')`.
- **Module-level prints**: don't add them — `msa_vocab.py` used to `print()` on every import, which was noisy. Inventory introspection should be a `__main__` block, not a side effect.
- **First run downloads**: starting the engine the first time downloads `obadx/muaalem-model-v3_2` (~660 MB) into the HF cache. Subsequent runs are local.
- **CPU expectations**: model load is 60–180 s, first-batch warm-up is another 30–60 s, then ~5–15 s per batch. Don't conclude something is hung until you've waited at least 3 minutes after "Loading pre-trained model...".
- **Model size**: the adapted MSA checkpoint is ~2.3 GB on disk because it stores all 11 multi-level heads (only `phonemes` was resized). RAM footprint at `float32` is ~3 GB.
- **Don't push large checkpoints**: `checkpoints/` is gitignored. If you accidentally commit one, use `git filter-branch` to strip it from unpushed commits — but be aware that filter-branch's internal `git reset --hard` will also remove tracked-then-untracked files from the working tree (this is how `msa_model_adapted/` disappeared once before; regeneration is via the adapter script).

## Project Layout (where to look)

| Concern | Path |
|---|---|
| Upstream model class | [src/quran_muaalem/modeling/modeling_multi_level_ctc.py](src/quran_muaalem/modeling/modeling_multi_level_ctc.py) |
| Upstream inference wrapper | [src/quran_muaalem/inference.py](src/quran_muaalem/inference.py) |
| Engine (LitServe) | [src/quran_muaalem/engine/](src/quran_muaalem/engine/) |
| App (FastAPI) | [src/quran_muaalem/app/](src/quran_muaalem/app/) |
| Quranic UI (Gradio) | [src/quran_muaalem/gradio_app.py](src/quran_muaalem/gradio_app.py) |
| MSA vocab / tokenizer | [src/quran_muaalem/modeling/msa_vocab.py](src/quran_muaalem/modeling/msa_vocab.py), [msa_tokenizer.py](src/quran_muaalem/modeling/msa_tokenizer.py) |
| MSA head resize | [src/quran_muaalem/modeling/adapt_model_for_msa.py](src/quran_muaalem/modeling/adapt_model_for_msa.py) |
| MSA dataset / data prep | [src/quran_muaalem/data/](src/quran_muaalem/data/) |
| MSA trainer | [src/quran_muaalem/training/train_msa.py](src/quran_muaalem/training/train_msa.py) |
| Train entry point | [train_msa_simple.py](train_msa_simple.py) |
| MSA serving (API + UI + helpers) | [src/quran_muaalem/msa/](src/quran_muaalem/msa/) |
| Pytest config | [tests/conftest.py](tests/conftest.py) |
| Runtime config | [.env](.env), [pyproject.toml](pyproject.toml) |
