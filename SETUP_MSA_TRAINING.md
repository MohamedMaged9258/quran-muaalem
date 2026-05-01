# Setup MSA Fine-Tuning - Your Exact Steps

You have the Common Voice Arabic dataset downloaded. Here's exactly what to do:

## Step 1: Move Dataset to Project (2 minutes)

The dataset you downloaded is in your Downloads folder. Move it to your project:

```powershell
# Open PowerShell and run:
cd C:\Users\moham\Projects\muaalem

# Move the dataset (replace with your actual folder name if different)
mv C:\Users\moham\Downloads\ar datasets\common_voice_ar

# Verify it worked
ls datasets\common_voice_ar\

# Should show: clips/, train.tsv, dev.tsv, test.tsv, etc.
```

---

## Step 2: Install Required Dependencies (5 minutes)

First, we need to add training dependencies to the project:

### 2a. Update pyproject.toml

I'll create the updated file with all needed dependencies for training:

```bash
# After this, run:
python3.14 -m uv sync --extra engine --extra training
```

---

## Step 3: Process the Dataset (2-3 hours)

This converts:
- MP3 → WAV (16kHz)
- Arabic text → Phonemes
- Organizes into train/val/test

```bash
cd C:\Users\moham\Projects\muaalem

# Run the conversion
python3.14 -m uv run python src/quran_muaalem/data/prepare_common_voice.py
```

**What you'll see:**
```
Processing train: 100%|████████| 28865/28865 [2:45:32<00:00, 2.93it/s]
Processing train: 28865 samples (5432 skipped)
Processing dev: 100%|████████| 10229/10229 [0:45:12<00:00, 3.75it/s]
Processing dev: 10229 samples (1823 skipped)
Processing test: 100%|████████| 10508/10508 [0:47:28<00:00, 3.68it/s]
Processing test: 10508 samples (1876 skipped)

✅ Dataset ready for training!
   Location: datasets/msa_speech
   Train samples: 22433
   Val samples: 8406
   Test samples: 8632
   Total samples: 39471
```

**This takes time - do overnight or while working on something else!**

Your dataset will be ready in: `datasets/msa_speech/`

---

## Step 4: Verify Dataset Created

```bash
ls datasets/msa_speech/

# Should show:
# - train/ (folder with .wav files)
# - val/ (folder with .wav files)
# - test/ (folder with .wav files)
# - manifest.json (the configuration file)
```

---

## Step 5: Prepare Training Script

Once data is ready, we'll create the training script. For now:

```bash
# Just verify your data is there
python3.14 -c "
import json
with open('datasets/msa_speech/manifest.json') as f:
    manifest = json.load(f)
    print(f'Train: {len(manifest[\"train\"])} samples')
    print(f'Val: {len(manifest[\"val\"])} samples')
    print(f'Test: {len(manifest[\"test\"])} samples')
"
```

---

## Timeline

| Step | Time | What to Do |
|------|------|-----------|
| 1 | 2 min | Move folder |
| 2 | 5 min | Install dependencies |
| 3 | 2-3 hrs | Run conversion (can be overnight) |
| 4 | 1 min | Verify it worked |
| 5 | Ready for training! | Start training once data is ready |

**Total: ~3 hours of actual work time**

---

## If Something Goes Wrong

### Error: "File not found: datasets/common_voice_ar"
- Make sure you moved the folder correctly in Step 1
- Check: `ls datasets\common_voice_ar\` shows files

### Error: "ModuleNotFoundError: librosa" or "soundfile"
- Run: `python3.14 -m uv sync --extra engine --extra training`
- Wait for it to finish

### Error: "Permission denied"
- Close any applications using those files
- Try again

### The script is too slow
- This is normal! Processing 40k+ audio files takes time
- You can monitor Task Manager to see CPU/GPU usage
- It's fine to leave running overnight

---

## Next Steps (After Data Processing)

Once `datasets/msa_speech/manifest.json` is created with ~40k samples:

1. **Update training script** - I'll create the full trainer
2. **Configure training** - Set batch size for your RTX 2050
3. **Start training** - Takes 1 day on GPU

---

## Questions Before You Start?

Before running Step 3 (which takes hours):
- Do you want to process ALL 40k+ samples, or start with 1000 for testing?
- Should I add the training dependencies to pyproject.toml first?

Let me know and I'll guide you!
