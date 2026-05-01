#!/usr/bin/env python3
"""Quick test to diagnose training setup."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader

print("Step 1: Testing dataset loading...")
from quran_muaalem.data.msa_dataset import MSAPhonemeDataset

try:
    dataset = MSAPhonemeDataset(
        "datasets/msa_speech/manifest.json",
        split="train",
        model_name="obadx/muaalem-model-v3_2",
    )
    print(f"Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    sys.exit(1)

print("\nStep 2: Testing single sample loading...")
try:
    sample = dataset[0]
    print(f"Sample loaded successfully")
    print(f"  - Audio shape: {sample['input_features'].shape}")
    print(f"  - Labels shape: {sample['labels'].shape}")
except Exception as e:
    print(f"ERROR loading sample: {e}")
    sys.exit(1)

print("\nStep 3: Testing dataloader with batch_size=1...")
try:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"DataLoader created")

    print("Loading first batch...")
    batch = next(iter(loader))
    print(f"Batch loaded successfully")
    print(f"  - input_features: {batch['input_features'].shape}")
    print(f"  - labels: {batch['labels'].shape}")
except Exception as e:
    print(f"ERROR with dataloader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing model loading...")
try:
    from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC

    print("Loading model (this may take a minute on CPU)...")
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        "obadx/muaalem-model-v3_2",
        torch_dtype=torch.float32,
    )
    print(f"Model loaded successfully")
    model.eval()
except Exception as e:
    print(f"ERROR loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing forward pass...")
try:
    with torch.no_grad():
        print("Running forward pass on batch (this may take 30-60 seconds on CPU)...")
        outputs = model(
            batch['input_features'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )
    print(f"Forward pass successful")
    if isinstance(outputs, dict):
        phonemes_logits = outputs.get('phonemes', outputs)
        print(f"  - phonemes logits shape: {phonemes_logits[0].shape if isinstance(phonemes_logits, (list, tuple)) else phonemes_logits.shape}")
    else:
        print(f"  - outputs type: {type(outputs)}")
except Exception as e:
    print(f"ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("All tests passed! Training setup is working.")
print("="*70)
print("\nYour setup is ready. To train:")
print("  python3.14 -m uv run python train_msa_simple.py --device cpu --batch_size 1 --epochs 1")
print("\nNote: Training on CPU is VERY SLOW. Install CUDA drivers for GPU support.")
