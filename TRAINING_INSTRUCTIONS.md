# Training Instructions - MSA Fine-Tuning

Your data is ready! You have **49,601 samples**. Now let's train.

---

## Step 1: Verify Everything is Installed

```bash
cd C:\Users\moham\Projects\muaalem

# Check dependencies
python3.14 -m uv sync --extra engine --extra training
```

---

## Step 2: Start Training

### **Option A: Simple (Recommended)**

```bash
cd C:\Users\moham\Projects\muaalem
python3.14 -m uv run python train_msa_simple.py
```

**Default settings:**
- Batch size: 4 (safe for RTX 2050)
- Epochs: 20
- Learning rate: 1e-4
- Output: `checkpoints/msa_model_v1/`

### **Option B: With Custom Settings**

```bash
python3.14 -m uv run python train_msa_simple.py \
    --epochs 25 \
    --batch_size 4 \
    --output_dir checkpoints/msa_model_v2
```

**Available options:**
```
--manifest PATH              # Path to manifest.json (default: datasets/msa_speech/manifest.json)
--model_name NAME            # Pre-trained model (default: obadx/muaalem-model-v3_2)
--output_dir DIR             # Save location (default: checkpoints/msa_model_v1)
--epochs N                   # Number of epochs (default: 20)
--batch_size N               # Batch size (default: 4)
--lr LR                      # Learning rate (default: 1e-4)
--accumulation_steps N       # Gradient accumulation (default: 1)
--device DEVICE              # cuda or cpu (default: cuda)
--num_workers N              # Data loading workers (default: 2)
```

---

## Expected Output

When training starts, you'll see:

```
======================================================================
Starting MSA Fine-Tuning Training
======================================================================
Device: cuda
Epochs: 20
Train samples: 28864
Val samples: 10229
======================================================================

Train Epoch 1: 100%|████████| 7216/7216 [15:42<00:00, 7.66it/s]
Validating: 100%|████████| 1279/1279 [01:45<00:00, 12.11it/s]

Epoch 1/20
  Train Loss: 45.2341
  Val Loss:   38.5612
  LR:         0.000100
  Saved best model (val_loss: 38.5612)
```

---

## Training Timeline on RTX 2050

| Epoch | Time per Epoch | Cumulative Time |
|-------|----------------|-----------------|
| 1-5   | ~17 min each   | 1.5 hours       |
| 6-10  | ~17 min each   | 3 hours         |
| 11-15 | ~17 min each   | 4.5 hours       |
| 16-20 | ~17 min each   | 6 hours         |

**Total: ~6 hours for 20 epochs**

You can safely leave training running overnight!

---

## Monitoring Training

### Option 1: Watch Live Output
Just run the script - see loss decreasing in real-time

### Option 2: Check GPU Usage
Open Task Manager:
1. Right-click taskbar → Task Manager
2. Performance tab → GPU
3. Look for RTX 2050 at 70-90% usage

### Option 3: Check Saved Models
```bash
# See checkpoints being created
ls checkpoints/msa_model_v1/

# Should show:
# - checkpoint_epoch_1/
# - checkpoint_epoch_2/
# - ... checkpoint_epoch_20/
# - best_model/
# - training_history.json
```

---

## What's Happening During Training

```
For each epoch:
  1. Load ~7200 training samples in batches of 4
  2. Pass audio through Wav2Vec2-BERT encoder
  3. Get CTC logits for phoneme prediction
  4. Compare predicted phonemes vs. ground truth
  5. Calculate loss (CTC loss)
  6. Backpropagate and update weights
  
  Then:
  7. Validate on ~1300 validation samples
  8. Save checkpoint if validation improved
  9. Adjust learning rate (cosine annealing)
  10. Repeat for next epoch
```

---

## After Training Completes

Once training finishes, you'll have:

```
checkpoints/msa_model_v1/
├── best_model/                    # Best checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── pytorch_model.bin
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
├── ... (all 20 checkpoints)
└── training_history.json          # Loss over time
```

### **Next: Evaluate on Test Set**

```bash
python3.14 -c "
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from quran_muaalem.data.msa_dataset import MSAPhonemeDataset
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from quran_muaalem.modeling.msa_tokenizer import MSATokenizer

# Load best model
model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
    'checkpoints/msa_model_v1/best_model',
    torch_dtype=torch.float32
)
model.eval()

# Load test data
test_dataset = MSAPhonemeDataset('datasets/msa_speech/manifest.json', split='test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Simple evaluation
tokenizer = MSATokenizer()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['input_features'], batch['attention_mask'])
        logits = outputs.logits
        pred_ids = logits.argmax(dim=-1)
        
        # Simple phoneme comparison
        for pred_seq, label_seq in zip(pred_ids, batch['labels']):
            pred_phonemes = tokenizer.decode(pred_seq.tolist())
            label_phonemes = tokenizer.decode(label_seq.tolist())
            
            pred_list = pred_phonemes.split()
            label_list = label_phonemes.split()
            
            for p, l in zip(pred_list, label_list):
                total += 1
                if p == l:
                    correct += 1

accuracy = (correct / total * 100) if total > 0 else 0
print(f'Phoneme Accuracy: {accuracy:.2f}%')
"
```

Expected accuracy: **80-85%** for MSA (first training run)

---

## Troubleshooting

### Issue: "Out of Memory" Error
**Solution:** Reduce batch size
```bash
python3.14 -m uv run python train_msa_simple.py --batch_size 2
```

### Issue: Training is very slow
**Check:**
```bash
# GPU utilization
nvidia-smi

# Should show >80% GPU usage
# If <20%, close other apps
```

### Issue: Loss not decreasing
**Check:**
1. Is learning rate too high? Try `--lr 1e-5`
2. Are labels correct? Sample a few from manifest.json
3. Is model stuck? Try restarting training

### Issue: "CUDA out of memory"
```bash
# Use gradient accumulation
python3.14 -m uv run python train_msa_simple.py \
    --batch_size 2 \
    --accumulation_steps 2
```

---

## Advanced: Training Configuration

### For RTX 2050 (4GB VRAM) - Safe
```bash
--batch_size 4 --accumulation_steps 1
```

### For RTX 2050 (4GB VRAM) - Push it
```bash
--batch_size 8 --accumulation_steps 1
```

### For RTX 2050 (4GB VRAM) - Cautious
```bash
--batch_size 2 --accumulation_steps 2
```

---

## Save Space

Training generates lots of checkpoints. To save space:

```bash
# Keep only best model
rm -r checkpoints/msa_model_v1/checkpoint_epoch_*
# Keeps ~500MB instead of 5GB
```

---

## Ready to Train?

```bash
cd C:\Users\moham\Projects\muaalem
python3.14 -m uv run python train_msa_simple.py
```

**Expected runtime: 6-8 hours on RTX 2050**

Go ahead and start! Let me know when it finishes and we'll evaluate the results.
