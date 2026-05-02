# TRAINING — MSA Fine-Tuning Process

This document covers the full training pipeline: from prepared dataset to a fine-tuned MSA phoneme recognizer. For obtaining and preparing the dataset, see [DATASET.md](DATASET.md). For the model itself, see [MODEL.md](MODEL.md). For serving, see [RUNNING.md](RUNNING.md).

---

## 1. Pipeline Overview

```
  Common Voice Arabic        prepare_common_voice.py         manifest.json
  (mp3 + sentence.tsv)  ───►  (resample + phonemize)   ───►  + 16 kHz WAVs
                                                                   │
                                                                   ▼
   adapt_model_for_msa.py        msa_model_adapted          msa_dataset.py
   (resize phoneme head)   ───►  checkpoint (31 classes) ◄── (PyTorch Dataset)
                                                                   │
                                                                   ▼
                                                            train_msa.py
                                                            (CTC fine-tune)
                                                                   │
                                                                   ▼
                                                       checkpoints/msa_model_v1/
                                                       best_model/
```

Everything lives under [src/quran_muaalem/data/](src/quran_muaalem/data/), [src/quran_muaalem/modeling/](src/quran_muaalem/modeling/), and [src/quran_muaalem/training/](src/quran_muaalem/training/).

---

## 2. Step 1 — Install Training Dependencies

```bash
python3.14 -m uv sync --extra training
```

This adds `soundfile`, `librosa`, `tqdm`, and `accelerate` on top of the base install. Verify with:

```bash
python3.14 -m uv run python -c "import torch, librosa, soundfile; print(torch.__version__)"
```

If you have an NVIDIA GPU and want CUDA, the project's `pyproject.toml` already pins `torch` to the `pytorch-cu121` index. Confirm with:

```bash
python3.14 -m uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## 3. Step 2 — Prepare the Dataset

Download Common Voice Arabic, extract it under `datasets/common_voice_ar/`, then run the conversion script to produce `datasets/msa_speech/manifest.json`. Full instructions, expected layout, statistics, and troubleshooting are in **[DATASET.md](DATASET.md)**.

Quick version:

```bash
python3.14 -m uv run python -m quran_muaalem.data.prepare_common_voice
```

After this step you should have ~49,601 samples / 17.4 h split 70 / 15 / 15.

---

## 4. Step 3 — Adapt the Pre-Trained Model

The pre-trained checkpoint outputs 43 Quranic phonemes. We resize it to 31 MSA phonemes once, then train.

```bash
python3.14 -m uv run python -c "from src.quran_muaalem.modeling.adapt_model_for_msa import adapt_model_for_msa; adapt_model_for_msa()"
```

This produces `checkpoints/msa_model_adapted/`. Details of the resize are in [MODEL.md §3](MODEL.md). You only need to run this once.

---

## 5. Step 4 — Run Training

The entry point is [train_msa_simple.py](train_msa_simple.py), which delegates to `quran_muaalem.training.train_msa.main`.

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` | `datasets/msa_speech/manifest.json` | Path to the manifest from step 3b. |
| `--model_name` | `obadx/muaalem-model-v3_2` | Pre-trained model or local checkpoint dir. **Use `checkpoints/msa_model_adapted` after step 4.** |
| `--output_dir` | `checkpoints/msa_model_v1` | Where checkpoints and history are written. |
| `--epochs` | `20` | Total epochs. |
| `--batch_size` | `4` | Per-device batch size. |
| `--lr` | `1e-4` | Initial learning rate (cosine decay over `--epochs`). |
| `--accumulation_steps` | `1` | Gradient accumulation. |
| `--device` | `cuda` | `cuda` or `cpu`. Auto-falls back to CPU if no GPU. |
| `--num_workers` | `0` | DataLoader workers. Keep `0` on Windows. |
| `--max_samples` | `None` | Cap the train/val sets. Useful for quick smoke tests. |

### Recommended GPU command

```bash
# 4 GB GPU (e.g. RTX 2050) — keep activations small, accumulate gradients.
python3.14 -m uv run python train_msa_simple.py --model_name checkpoints/msa_model_adapted --device cuda --epochs 20 --batch_size 1 --accumulation_steps 4 --lr 1e-4 --output_dir checkpoints/msa_model_v1

# 8+ GB GPU — drop the accumulation, raise batch size.
python3.14 -m uv run python train_msa_simple.py --model_name checkpoints/msa_model_adapted --device cuda --epochs 20 --batch_size 4 --lr 1e-4 --output_dir checkpoints/msa_model_v1
```

Because the encoder is frozen, the GPU memory footprint is dominated by **activations** during the forward pass, not by optimizer state. `batch_size 1 + accumulation_steps 4` keeps activations small while preserving an effective batch size of 4.

### Quick CPU smoke test (5 minutes)

```bash
python3.14 -m uv run python train_msa_simple.py --model_name checkpoints/msa_model_adapted --device cpu --epochs 1 --batch_size 1 --max_samples 100
```

This validates the whole pipeline end-to-end without committing to a real training run.

---

## 6. What Training Actually Does

Inside `CTCTrainer` (see [src/quran_muaalem/training/train_msa.py](src/quran_muaalem/training/train_msa.py)):

1. **Freeze**: `load_model_for_msa()` sets `requires_grad=False` on every parameter, then re-enables only `level_to_lm_head["phonemes"]`. The encoder and unused heads stay frozen. `AdamW` is given only the trainable subset, so no optimizer state is allocated for the ~605 M frozen params.
2. **Forward**: `_forward_loss(batch)` casts inputs to `model.dtype` (so bf16 weights on CUDA don't meet fp32 activations), runs the model, and pulls `outputs.logits["phonemes"]` of shape `(batch, T_enc, 31)`. The same helper is used for both training and validation, so the two paths never drift.
3. **Lengths**: `input_lengths` = the actual logits time dimension `T_enc` (NOT the attention-mask sum — the encoder downsamples ~2×). `target_lengths` = count of non-pad tokens per sample.
4. **Loss**: log-probs are explicitly upcast to fp32 before `nn.CTCLoss(blank=0, zero_infinity=True)` — CTC doesn't support fp16/bf16.
5. **Backward**: gradient accumulation (`--accumulation_steps`), `clip_grad_norm_(max_norm=1.0)`, `AdamW(weight_decay=0.01)`, `CosineAnnealingLR` step at end of epoch.
6. **Checkpointing**: every epoch writes `checkpoint_epoch_N/`. Whenever `val_loss` improves, also writes `best_model/`. Each save includes the **feature extractor** (`preprocessor_config.json`) alongside `config.json` and `model.safetensors`, so the resulting directory can be loaded directly by `AutoFeatureExtractor.from_pretrained` with no extra setup. Final `training_history.json` records per-epoch train/val loss and learning rate.

The dataset class [src/quran_muaalem/data/msa_dataset.py](src/quran_muaalem/data/msa_dataset.py) handles audio loading, feature extraction with `AutoFeatureExtractor`, padding to a fixed `max_features`, and tokenization via `MSATokenizer`. Labels are padded to length 256 with zeros for batching. If `model_name` looks like a local path that doesn't exist, the dataset fails fast with a clear `FileNotFoundError` instead of letting transformers misinterpret it as a HuggingFace repo id.

---

## 7. Expected Numbers

Targets after a full 20-epoch run on the adapted model with the full ~17 h MSA set:

| Metric | Start | After 20 epochs |
|---|---|---|
| Train CTC loss | ~40 | 5–10 |
| Val CTC loss | ~35 | 10–20 |
| Phoneme accuracy (test) | — | 80–85 % |

### Rough wall-clock

With the encoder frozen, the **forward pass** still runs over all 605 M parameters (you still need its features), but **backward + optimizer** only touch the 32 K-param head. Per-step time is dominated by the encoder forward, not the optimizer.

| Hardware | Per epoch | 20 epochs |
|---|---|---|
| RTX 2050 (4 GB) | 15–20 min | ~5–6 h |
| Modern CPU (10 cores) | 2–3 h | 40–60 h |
| Modest CPU (4 cores) | 5–8 h | 100+ h |

CPU is fine for smoke tests but impractical for the full run. First batch on CUDA always pays a 30–90 s warm-up tax (kernel JIT + cuDNN autotune); subsequent batches are 5–10× faster.

---

## 8. Outputs

After training:

```
checkpoints/msa_model_v1/
├── best_model/                # checkpoint with lowest val loss
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
├── ...
└── training_history.json      # {"train_loss": [...], "val_loss": [...], "lr": [...]}
```

`best_model/` is what you'd plug back into the engine — see [RUNNING.md](RUNNING.md) for swapping checkpoints.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA not available, falling back to CPU` | NVIDIA driver / CUDA toolkit not installed. | Install CUDA 12.1 + matching driver, or accept CPU. |
| `Expected input_lengths to have value at most N, but got value M` | Training script desync — CTC `input_lengths` must equal logits time dim, not attention mask sum. | Already fixed in `train_msa.py`; pull latest if you're seeing this. |
| `stack expects each tensor to be equal size` in DataLoader | Variable feature lengths across the batch. | `MSAPhonemeDataset.__getitem__` pads to `max_features`; check that `max_duration` is consistent. |
| `RuntimeError: expected scalar type Float but found BFloat16` | Inputs reaching the model in fp32 while the model is bf16. | Already fixed: `_forward_loss` casts inputs to `model.dtype`. If you write a custom forward path, do the same. |
| `RuntimeError: CUDA error: out of memory` after first batch finishes | Likely AdamW lazy-allocating optimizer state on the first `.step()`. | The trainer now passes only trainable params to AdamW (encoder is frozen), so this should be gone. If it returns, lower `--batch_size`. |
| `Repository Not Found … 401 Client Error … checkpoints/msa_model_adapted` | The local path doesn't exist; transformers tried it as a HF repo id. | The dataset and adapter both validate the path now. Re-run `adapt_model_for_msa()` to recreate the directory. |
| `Can't load … preprocessor_config.json` when serving a checkpoint | Older checkpoint saved without the feature extractor. | Re-save with the current trainer (which writes `preprocessor_config.json` on every save), or copy it from `obadx/muaalem-model-v3_2`. |
| Loss stays at ~40, never drops | LR too high or label/blank mismatch. | Try `--lr 1e-5`. Verify `[PAD]=0` matches the CTC `blank=0`. |
| OOM on GPU | Batch too large for 4 GB. | `--batch_size 1 --accumulation_steps 4`. |
| Validation loss climbs while training drops | Overfitting on small data subset. | Train on the full set, lower LR, or stop earlier (use `best_model/`). |

---

## 10. File Map

| File | Role |
|---|---|
| [src/quran_muaalem/data/prepare_common_voice.py](src/quran_muaalem/data/prepare_common_voice.py) | TSV → WAV + phoneme manifest. |
| [src/quran_muaalem/data/msa_dataset.py](src/quran_muaalem/data/msa_dataset.py) | `MSAPhonemeDataset` and `get_data_loaders`. |
| [src/quran_muaalem/modeling/adapt_model_for_msa.py](src/quran_muaalem/modeling/adapt_model_for_msa.py) | One-shot head resize: 43 → 31. |
| [src/quran_muaalem/training/train_msa.py](src/quran_muaalem/training/train_msa.py) | `CTCTrainer` + CLI `main()`. |
| [train_msa_simple.py](train_msa_simple.py) | Thin wrapper that calls `train_msa.main`. |
