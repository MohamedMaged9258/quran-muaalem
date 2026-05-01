# MSA Fine-Tuning Guide: Phoneme-Only Approach

This guide explains how to fine-tune the Quran Muaalem model on Modern Standard Arabic (MSA) using the **phoneme-only approach**.

## Quick Overview

**What you're doing:**
- Taking the pre-trained Wav2Vec2-BERT encoder (frozen)
- Replacing the Quranic CTC head with an MSA CTC head
- Fine-tuning on MSA audio + phoneme transcriptions
- Output: phoneme predictions only (no tajweed/linguistic properties)

**Requirements:**
- 50-100 hours of MSA speech audio
- Phonetic transcriptions (space-separated phonemes)
- 2-4 weeks of work (including data collection/annotation)

---

## Data Format

### 1. Audio Files
- **Format:** WAV or MP3
- **Sampling rate:** 16kHz (critical!)
- **Duration:** Variable (5 seconds to 1 minute per file)
- **Quality:** Clean speech (minimal background noise)
- **Organization:**
  ```
  data/audio/
  ├── train/
  │   ├── audio_001.wav
  │   ├── audio_002.wav
  │   └── ...
  ├── val/
  │   └── ...
  └── test/
      └── ...
  ```

### 2. Phonetic Transcriptions

Each audio file needs a corresponding `.txt` file with the same name containing **space-separated phonemes**.

**Format:** `phoneme1 phoneme2 phoneme3 ...`

**Example:**
```
File: audio_001.wav
Transcription: د َ ر َ س

File: audio_002.wav
Transcription: ك َ ت َ ب َ

File: audio_003.wav
Transcription: م َ د َ ر َ س َ ة
```

**Rules:**
- Use the phoneme characters from `msa_vocab.py` exactly
- Separate each phoneme with a space
- Include vowels and diacritics (َ، ُ، ِ، ْ)
- One transcription per file

### 3. Dataset Structure

Recommended directory layout:
```
datasets/msa_speech/
├── train/
│   ├── audio_001.wav
│   ├── audio_001.txt
│   ├── audio_002.wav
│   ├── audio_002.txt
│   └── ... (50-80 hours)
├── val/
│   ├── audio_val_001.wav
│   ├── audio_val_001.txt
│   └── ... (5-10 hours)
└── test/
    ├── audio_test_001.wav
    ├── audio_test_001.txt
    └── ... (5-10 hours)
```

**Data split:**
- **Train:** 70-80% (50-80 hours)
- **Val:** 10-15% (5-10 hours)
- **Test:** 5-15% (5-10 hours)

---

## Step 1: Prepare Your Data

### 1.1 Convert to 16kHz WAV (if needed)

If your audio is at a different sampling rate, convert it:

```bash
# Convert single file
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Batch convert all MP3s
for f in *.mp3; do
    ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
done
```

### 1.2 Create Phonetic Transcriptions

You have three options:

**Option A: Manual Annotation (Most accurate)**
- Listen to each audio file
- Write phonemes in MSA script (د َ ر َ س)
- Best quality but time-consuming
- Recommended for first 50 files to validate process

**Option B: Semi-automatic with Whisper (Fast)**
```python
import whisper
from arabic_text_to_phonemes import arabic_to_phonemes

# 1. Get text from Whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav", language="ar")
text = result["text"]  # "درس"

# 2. Convert to phonemes
phonemes = arabic_to_phonemes(text)
# → "د َ ر َ س"

# 3. Manual review and fix (20% of files)
```

**Option C: Use existing Arabic speech datasets**
- CommonVoice Arabic: https://commonvoice.mozilla.org/en/datasets
- ArabSpeech: https://huggingface.co/datasets/arabic_speech_corpus
- QCRI Arabic: https://www.qcri.org/

These already have transcriptions - convert text to phonemes.

### 1.3 Create Dataset Manifest (JSON)

Create a `manifest.json` file listing all audio files:

```json
{
  "train": [
    {
      "audio": "data/train/audio_001.wav",
      "phonemes": "د َ ر َ س"
    },
    {
      "audio": "data/train/audio_002.wav",
      "phonemes": "ك َ ت َ ب َ"
    }
  ],
  "val": [
    {
      "audio": "data/val/audio_val_001.wav",
      "phonemes": "ق ِ ر َ ء َ ة"
    }
  ],
  "test": [
    {
      "audio": "data/test/audio_test_001.wav",
      "phonemes": "ع َ ل َ م"
    }
  ]
}
```

---

## Step 2: Create PyTorch Dataset

Create `src/quran_muaalem/data/msa_dataset.py`:

```python
import json
from pathlib import Path
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor
from ..modeling.msa_tokenizer import MSATokenizer


class MSAPhonemeDataset(Dataset):
    """Dataset for MSA phoneme recognition."""

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 15.0,
        model_name: str = "obadx/muaalem-model-v3_2",
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.samples = manifest[split]

        # Setup feature extractor and tokenizer
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = MSATokenizer()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["audio"]
        phonetic_script = sample["phonemes"]

        # Load audio
        audio_array, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            duration=self.max_duration,
        )

        # Extract features
        features = self.processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=3000,  # Adjust based on max_duration
        )

        # Tokenize phonemes
        token_ids = self.tokenizer.encode(phonetic_script)

        return {
            "input_features": features["input_features"].squeeze(0),
            "attention_mask": features["attention_mask"].squeeze(0),
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }


def get_data_loaders(
    manifest_path: str,
    batch_size: int = 4,
    num_workers: int = 0,
):
    """Create train/val/test dataloaders."""

    train_dataset = MSAPhonemeDataset(manifest_path, split="train")
    val_dataset = MSAPhonemeDataset(manifest_path, split="val")
    test_dataset = MSAPhonemeDataset(manifest_path, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
```

---

## Step 3: Adapt the Model

The model architecture doesn't change - only the output layer vocabulary size.

**Key changes needed:**

1. **Input:** Same (Wav2Vec2-BERT encoder frozen)
2. **Output:** CTC head output size changes:
   - Quranic: ~100 tokens (phonemes + properties)
   - MSA: ~35 tokens (phonemes only)

### Minimal Code Change

In `src/quran_muaalem/modeling/modeling_multi_level_ctc.py`:

```python
# Add MSA support
def create_msa_model(pretrained_model_path: str, device: str = "cpu"):
    """Create MSA model by resizing CTC output layers."""
    from transformers import AutoConfig
    from .modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
    from .msa_tokenizer import MSATokenizer

    # Load base model
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    # Resize phoneme output layer to MSA vocabulary size (35 tokens)
    msa_tokenizer = MSATokenizer()
    msa_vocab_size = msa_tokenizer.get_vocab_size()

    # Only keep phoneme CTC head, resize output
    model.ctc_phonemes = torch.nn.Linear(
        model.ctc_phonemes.in_features,
        msa_vocab_size
    )

    # Remove other level heads (tajweed, sifat) - not needed for MSA
    if hasattr(model, 'ctc_tajweed'):
        delattr(model, 'ctc_tajweed')
    # ... remove other heads

    return model.to(device)
```

---

## Step 4: Training Script

Create `src/quran_muaalem/training/msa_trainer.py`:

```python
import torch
from torch.optim import AdamW
from torch.nn import CTCLoss
from tqdm import tqdm
import json


class MSAPhonemeTrainer:
    """Trainer for MSA phoneme recognition model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: str = "cpu",
        lr: float = 1e-4,
        num_epochs: int = 20,
        save_dir: str = "./checkpoints/msa_model",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.ctc_loss = CTCLoss(blank=0, zero_infinity=True)
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            input_features = batch["input_features"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_features,
                attention_mask=attention_mask,
                return_dict=True,
            )

            # CTC loss on phoneme output
            logits = outputs.logits  # Shape: (batch, time, vocab_size)
            
            # Compute input lengths for CTC
            input_lengths = (attention_mask.sum(dim=1)).cpu()
            target_lengths = (labels != 0).sum(dim=1).cpu()

            loss = self.ctc_loss(
                logits.transpose(0, 1).log_softmax(2),
                labels,
                input_lengths,
                target_lengths,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                input_features = batch["input_features"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_features,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                logits = outputs.logits
                input_lengths = (attention_mask.sum(dim=1)).cpu()
                target_lengths = (labels != 0).sum(dim=1).cpu()

                loss = self.ctc_loss(
                    logits.transpose(0, 1).log_softmax(2),
                    labels,
                    input_lengths,
                    target_lengths,
                )

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        """Full training loop."""
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_pretrained(self.save_dir / "best_model")
                print(f"✅ Best model saved (val_loss: {val_loss:.4f})")

            # Save checkpoint
            self.model.save_pretrained(self.save_dir / f"checkpoint_epoch_{epoch+1}")

        # Save history
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f)

        print(f"\n✅ Training complete! Model saved to {self.save_dir}")
```

**Usage:**

```python
from transformers import AutoFeatureExtractor
from quran_muaalem.data.msa_dataset import get_data_loaders
from quran_muaalem.modeling.modeling_multi_level_ctc import create_msa_model
from quran_muaalem.training.msa_trainer import MSAPhonemeTrainer

# Load data
train_loader, val_loader, test_loader = get_data_loaders(
    "data/manifest.json",
    batch_size=4,
)

# Create model
model = create_msa_model("obadx/muaalem-model-v3_2", device="cuda")

# Train
trainer = MSAPhonemeTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    lr=1e-4,
    num_epochs=20,
)
trainer.train()
```

---

## Step 5: Evaluation

After training, evaluate on test set:

```python
from jiwer import cer, wer

def evaluate_phoneme_error_rate(model, test_loader, tokenizer, device="cpu"):
    """Calculate Phoneme Error Rate (PER)."""
    model.eval()
    references = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch["input_features"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = outputs.logits

            # Decode predictions
            pred_ids = logits.argmax(dim=-1)
            for pred_id_seq in pred_ids:
                pred_phonemes = tokenizer.decode(pred_id_seq.cpu().tolist())
                predictions.append(pred_phonemes)

            # Decode references
            for label_seq in batch["labels"]:
                ref_phonemes = tokenizer.decode(label_seq.cpu().tolist())
                references.append(ref_phonemes)

    # Calculate metrics
    per = cer(references, predictions)
    print(f"Phoneme Error Rate (PER): {per:.2%}")
    print(f"Phoneme Accuracy: {(1-per):.2%}")
```

---

## Minimum Data Requirements

**Absolute minimum:** 20 hours
- Risky, likely overfitting
- Model may not generalize

**Recommended:** 50-100 hours
- Good balance of coverage and training time
- Should get ~85-90% phoneme accuracy

**Ideal:** 200+ hours
- Robust model
- Can reach 95%+ accuracy
- Takes 1-2 weeks to train on GPU

---

## Common Issues & Solutions

### Issue 1: Audio Format Mismatch
```
Error: "Expected 1D input"
```
**Solution:** Ensure all audio is mono, 16kHz WAV/MP3

### Issue 2: Phoneme Encoding Errors
```
Error: "Key not in vocabulary"
```
**Solution:** Check transcriptions use exact phoneme characters from `msa_vocab.py`

### Issue 3: Out of Memory
```
Error: "CUDA out of memory"
```
**Solution:** Reduce batch_size from 4 to 2 or 1

### Issue 4: Model Overfits
```
Train loss: 0.1, Val loss: 5.0
```
**Solution:** 
- Add early stopping (stop if val_loss doesn't improve for 3 epochs)
- Use dropout
- Collect more data

---

## Next Steps

1. **Collect/Prepare 50+ hours of MSA audio**
2. **Create phonetic transcriptions** (manually or semi-automatic)
3. **Build manifest.json** with all audio + phoneme pairs
4. **Run training script** (start with 20 epochs on GPU)
5. **Evaluate on test set** and measure PER
6. **If accuracy good:** Deploy for inference
7. **If accuracy bad:** Add more data or adjust hyperparameters

---

## Resources

- **MSA Phoneme Inventory:** `src/quran_muaalem/modeling/msa_vocab.py`
- **MSA Tokenizer:** `src/quran_muaalem/modeling/msa_tokenizer.py`
- **Example Dataset:** See section "Data Format" above
- **Training Code:** `src/quran_muaalem/training/msa_trainer.py`

Good luck! This is a solid foundation for MSA fine-tuning. Start with a small dataset (20 files) to validate the pipeline works, then scale up.
