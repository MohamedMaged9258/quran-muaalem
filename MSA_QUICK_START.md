# MSA Fine-Tuning Quick Start (For You)

You have RTX 2050 + Common Voice Arabic. Here's your exact path:

---

## Step 1: Download Common Voice Arabic (30 minutes)

1. Go to: https://mozilladatacollective.com/datasets/cmn2g7uu701fqo1072r5na25l
2. Scroll to **"Arabic"**
3. Click **"Download"** button
4. Wait for ~10GB download

File structure after download:
```
Downloads/
└── cv-corpus-17.0-2024-03-20/
    └── cv-corpus-17.0-2024-03-20/
        └── ar/
            ├── clips/          (audio MP3 files)
            ├── validated.tsv   (good quality transcriptions)
            └── test.tsv
```

---

## Step 2: Move to Your Project (10 minutes)

```bash
# In Windows PowerShell or Git Bash
cd C:\Users\moham\Projects\muaalem

# Move the extracted Common Voice folder
mv C:\Users\moham\Downloads\cv-corpus-17.0-2024-03-20\cv-corpus-17.0-2024-03-20\ar datasets/common_voice_ar

# Verify it worked
ls datasets/common_voice_ar/
# Should show: clips/, validated.tsv, test.tsv
```

---

## Step 3: Process Dataset (2-3 hours)

This converts MP3s to WAV and text to phonemes:

```bash
cd C:\Users\moham\Projects\muaalem

# Run the conversion script
python3.14 -m uv run python src/quran_muaalem/data/prepare_common_voice.py
```

**What happens:**
- Reads `validated.tsv` (20k+ recordings)
- Converts MP3 → WAV @ 16kHz
- Converts Arabic text → phonemes
- Creates 70% train / 15% val / 15% test split
- Saves to `datasets/msa_speech/`

**Output structure:**
```
datasets/msa_speech/
├── train/           (14k+ audio files)
├── val/             (3k+ audio files)
├── test/            (3k+ audio files)
└── manifest.json    (JSON with audio paths + phonemes)
```

---

## Step 4: Install Training Dependencies (5 minutes)

```bash
# Add training-specific dependencies
python3.14 -m uv sync --extra engine --extra training
```

(We'll need to add these to pyproject.toml)

---

## Step 5: Train Model on RTX 2050 (1-2 weeks)

```bash
# Run training script
python3.14 -m uv run python -m quran_muaalem.training.msa_trainer \
    --dataset datasets/msa_speech/manifest.json \
    --output checkpoints/msa_model \
    --epochs 20 \
    --batch_size 8 \
    --device cuda
```

**Training time on RTX 2050:**
- ~20k samples = 15 minutes per epoch
- 20 epochs = 5 hours total
- With validation = 1 day of training

**GPU memory:** RTX 2050 has 4GB
- Batch size 8 = ~3.2GB
- Batch size 4 = ~1.8GB (safer)

---

## Step 6: Evaluate Results

```bash
# Test on unseen data
python3.14 -m uv run python -m quran_muaalem.evaluate \
    --model checkpoints/msa_model/best_model \
    --test_data datasets/msa_speech/manifest.json \
    --device cuda
```

Expected output:
```
Phoneme Error Rate (PER): 15-20%
Phoneme Accuracy: 80-85%
```

---

## Complete Workflow (Copy-Paste Ready)

```bash
# 1. Move to project
cd C:\Users\moham\Projects\muaalem

# 2. Prepare data (assumes Common Voice already downloaded to Downloads)
mv C:\Users\moham\Downloads\cv-corpus-*/cv-corpus-*/ar datasets/common_voice_ar
python3.14 -m uv run python src/quran_muaalem/data/prepare_common_voice.py

# 3. Install training dependencies
python3.14 -m uv sync --extra engine --extra training

# 4. Train (this will take hours - can leave running)
python3.14 -m uv run python -m quran_muaalem.training.msa_trainer \
    --dataset datasets/msa_speech/manifest.json \
    --output checkpoints/msa_model \
    --epochs 20 \
    --batch_size 8 \
    --device cuda

# 5. Evaluate
python3.14 -m uv run python -m quran_muaalem.evaluate \
    --model checkpoints/msa_model/best_model \
    --test_data datasets/msa_speech/manifest.json
```

---

## What "Annotation" Means (For You)

You're using **Common Voice** which is already annotated:

```
Common Voice provides:
  ✅ Audio (clips/audio_001.mp3)
  ✅ Text annotation (validated.tsv)

Our script does:
  1. Reads the Arabic text from validated.tsv
  2. Converts to phonemes: "درس" → "د َ ر َ س"
  3. Saves ready-to-train format

You don't manually type anything - it's automated!
```

---

## Timeline

- **Step 1-2:** Download + move data = 1 hour
- **Step 3:** Process dataset = 2-3 hours (can do overnight)
- **Step 4:** Install deps = 5 minutes
- **Step 5:** Training = 1 day (overnight)
- **Step 6:** Evaluation = 1 hour

**Total time: 2-3 days of work**

---

## Troubleshooting

### Issue: "File not found: datasets/common_voice_ar"
**Fix:** Make sure you've completed Step 2 (move the folder)

### Issue: "ModuleNotFoundError: librosa"
**Fix:** Run `python3.14 -m uv sync --extra engine`

### Issue: "CUDA out of memory"
**Fix:** Reduce batch_size from 8 to 4 in training command

### Issue: "Training is too slow"
**Check:** 
- GPU usage: Open Task Manager → Performance → GPU
- Should show RTX 2050 at 90%+ usage
- If low, try closing other apps

---

## Next After Training

Once training completes and accuracy is good (>80%):

1. **Deploy for inference:** Use the fine-tuned model to recognize MSA
2. **Expand to more data:** Collect 200+ hours for 95%+ accuracy
3. **Build application:** Create API/UI for real users
4. **Fine-tune further:** Add linguistic properties (multi-level approach)

---

Ready to start? Begin with **Step 1: Download Common Voice**!
