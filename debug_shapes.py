#!/usr/bin/env python3
"""Debug shapes to find the mismatch."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from quran_muaalem.data.msa_dataset import MSAPhonemeDataset
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC

print("Loading dataset...")
train_dataset = MSAPhonemeDataset("datasets/msa_speech/manifest.json", split="train")

# Just first sample
small_subset = Subset(train_dataset, [0])
train_loader = DataLoader(small_subset, batch_size=1, shuffle=False, num_workers=0)

print("Loading model...")
model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
    "obadx/muaalem-model-v3_2",
    torch_dtype=torch.float32,
)
model.eval()
device = torch.device("cpu")
model = model.to(device)

print("\nAnalyzing shapes...")
for batch in train_loader:
    input_features = batch["input_features"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print(f"Input features shape: {input_features.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Attention mask sum (per sample): {attention_mask.sum(dim=1)}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels (non-zero count): {(labels != 0).sum(dim=1)}")

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )

    logits_dict = outputs.logits

    print(f"\nAvailable levels in outputs: {list(logits_dict.keys())}")
    for level, logits_level in logits_dict.items():
        print(f"  {level}: {logits_level.shape}")

    logits = logits_dict["phonemes"]

    print(f"\nPhoneme logits shape: {logits.shape}")
    print(f"Logits time dimension: {logits.shape[1]}")
    print(f"Logits vocab dimension: {logits.shape[2]}")

    # What CTC needs
    input_lengths = (attention_mask.sum(dim=1)).cpu()
    target_lengths = (labels != 0).sum(dim=1).cpu()

    print(f"\nCTC input_lengths: {input_lengths}")
    print(f"CTC target_lengths: {target_lengths}")
    print(f"Actual logits time: {logits.shape[1]}")

    print(f"\nMismatch: logits time ({logits.shape[1]}) vs input_lengths ({input_lengths.item()})")
    print(f"Need: input_lengths <= logits_time AND target_lengths <= logits_time")

    # Calculate proper max_label_length
    proper_max_label = logits.shape[1] - 5  # Leave margin
    print(f"Proper max label length should be: {proper_max_label}")
