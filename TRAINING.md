# TRAINING — MSA Fine-Tuning Process

This document covers the full training pipeline: from raw Common Voice download to a fine-tuned MSA phoneme recognizer. For the model itself, see [MODEL.md](MODEL.md). For serving, see [RUNNING.md](RUNNING.md).

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

### 3a. Download Common Voice Arabic

1. Go to https://commonvoice.mozilla.org/en/datasets
2. Pick the **Arabic** locale and download a release (e.g. Common Voice 17 Arabic).
3. Extract the archive into `datasets/common_voice_ar/`. You should end up with:

```
datasets/common_voice_ar/
├── clips/                     # *.mp3 audio files
├── train.tsv
├── dev.tsv
├── test.tsv
├── validated.tsv
└── ...
```

### 3b. Convert to WAV + phoneme manifest

```bash
python3.14 -m uv run python -m quran_muaalem.data.prepare_common_voice
```

What this does (see [src/quran_muaalem/data/prepare_common_voice.py](src/quran_muaalem/data/prepare_common_voice.py)):

- Reads `train.tsv`, `dev.tsv`, `test.tsv`.
- Loads each MP3 with `librosa` and resamples to **16 kHz mono**.
- Writes `audio_NNNNNN.wav` into `datasets/msa_speech/{train,val,test}/`.
- Maps each Arabic character to its MSA phoneme via `ArabicToPhonemes.PHONEME_MAP`.
- Filters out clips shorter than 0.5s or longer than 30s.
- Writes a single `datasets/msa_speech/manifest.json` with three splits:

```json
{
  "train": [{"audio": "datasets/msa_speech/train/audio_000000.wav", "phonemes": "د ر س"}, ...],
  "val":   [...],
  "test":  [...]
}
```

After this step, expect roughly **49,601 samples / 17.4 hours** (depending on which CV release you picked) split 70 / 15 / 15.

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
python3.14 -m uv run python train_msa_simple.py \
    --model_name checkpoints/msa_model_adapted \
    --device cuda \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir checkpoints/msa_model_v1
```

### Quick CPU smoke test (5 minutes)

```bash
python3.14 -m uv run python train_msa_simple.py --model_name checkpoints/msa_model_adapted --device cpu --epochs 1 --batch_size 1 --max_samples 100
```

This validates the whole pipeline end-to-end without committing to a real training run.

---

## 6. What Training Actually Does

Inside `CTCTrainer` (see [src/quran_muaalem/training/train_msa.py](src/quran_muaalem/training/train_msa.py)):

1. **Forward**: feed the audio batch through the (frozen) encoder and the (trainable) phoneme head. Take `outputs.logits["phonemes"]` of shape `(batch, T_enc, 31)`.
2. **Lengths**: `input_lengths` is set to the actual logits time dimension `T_enc` (not the attention mask) to satisfy CTC's `input_length >= target_length` requirement. `target_lengths` is the count of non-pad tokens per sample.
3. **Loss**: `log_softmax` over the vocab axis, transpose to `(T_enc, batch, 31)`, then `nn.CTCLoss(blank=0, zero_infinity=True)`.
4. **Backward**: gradient accumulation (`--accumulation_steps`), `clip_grad_norm_(max_norm=1.0)`, `AdamW(weight_decay=0.01)` step, `CosineAnnealingLR` step at the end of each epoch.
5. **Validation**: same forward + loss on the validation split, no gradient.
6. **Checkpointing**: every epoch writes `checkpoint_epoch_N/`. Whenever `val_loss` improves, also writes `best_model/`. Final `training_history.json` records per-epoch train/val loss and learning rate.

The dataset class [src/quran_muaalem/data/msa_dataset.py](src/quran_muaalem/data/msa_dataset.py) handles audio loading, feature extraction with `AutoFeatureExtractor`, padding to a fixed `max_features`, and tokenization via `MSATokenizer`. Labels are padded to length 256 with zeros for batching.

---

## 7. Expected Numbers

Targets after a full 20-epoch run on the adapted model with the full ~17 h MSA set:

| Metric | Start | After 20 epochs |
|---|---|---|
| Train CTC loss | ~40 | 5–10 |
| Val CTC loss | ~35 | 10–20 |
| Phoneme accuracy (test) | — | 80–85 % |

### Rough wall-clock

| Hardware | Per epoch | 20 epochs |
|---|---|---|
| RTX 2050 (4 GB) | 15–20 min | ~5–6 h |
| Modern CPU (10 cores) | 2–3 h | 40–60 h |
| Modest CPU (4 cores) | 5–8 h | 100+ h |

CPU is fine for smoke tests but impractical for the full run.

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
| Loss stays at ~40, never drops | LR too high or label/blank mismatch. | Try `--lr 1e-5`. Verify `[PAD]=0` matches the CTC `blank=0`. |
| OOM on GPU | Batch too large for 4 GB. | `--batch_size 2 --accumulation_steps 2`. |
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
