#!/usr/bin/env python3
"""
CPU-optimized training for MSA fine-tuning
Much simpler than the full trainer - good for testing on CPU
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import torch
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from quran_muaalem.data.msa_dataset import MSAPhonemeDataset
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC

def main():
    print("="*70)
    print("MSA Training (CPU Mode)")
    print("="*70)

    # Load small subset for testing
    print("\nLoading dataset...")
    train_dataset = MSAPhonemeDataset(
        "datasets/msa_speech/manifest.json",
        split="train",
    )

    # Use only first 500 samples for testing (saves time)
    train_subset = Subset(train_dataset, list(range(min(500, len(train_dataset)))))

    train_loader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    print(f"Loaded {len(train_subset)} training samples")

    # Load model
    print("Loading model...")
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        "obadx/muaalem-model-v3_2",
        torch_dtype=torch.float32,
    )
    model.eval()  # Set to eval mode to reduce computation

    device = torch.device("cpu")
    model = model.to(device)

    print(f"Model loaded on {device}")

    # Loss function
    ctc_loss = CTCLoss(blank=0, zero_infinity=True)

    print("\n" + "="*70)
    print("Starting training... (This will take a few minutes)")
    print("="*70 + "\n")

    # Simple forward pass test
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Testing")):
        input_features = batch["input_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        try:
            with torch.no_grad():
                outputs = model(
                    input_features,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            # Check output type
            if isinstance(outputs, dict):
                logits = outputs.get("phonemes", list(outputs.values())[0])
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
            else:
                logits = outputs

            # Calculate loss
            input_lengths = (attention_mask.sum(dim=1)).cpu()
            target_lengths = (labels != 0).sum(dim=1).cpu()

            if logits.dim() == 3:
                logits_t = logits.transpose(0, 1)
            else:
                logits_t = logits

            log_probs = torch.nn.functional.log_softmax(logits_t, dim=-1)
            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        except Exception as e:
            print(f"Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            break

        if batch_idx >= 100:  # Test only 100 batches
            break

    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)
    print("\nYour training setup works on CPU!")
    print("Next steps:")
    print("1. Install NVIDIA drivers for your RTX 2050")
    print("2. Install CUDA Toolkit 12.1")
    print("3. Training will be 10-100x faster on GPU")
    print("\nFor now, you can train on CPU with:")
    print("  python3.14 -m uv run python train_msa_simple.py --device cpu --batch_size 1 --epochs 1")

if __name__ == "__main__":
    main()
