"""
MSA Dataset for PyTorch training
Loads audio + phoneme pairs from manifest.json
"""

import json
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor

from ..modeling.msa_tokenizer import MSATokenizer


class MSAPhonemeDataset(Dataset):
    """Dataset for MSA phoneme recognition."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 15.0,
        model_name: str = "obadx/muaalem-model-v3_2",
    ):
        """
        Initialize MSA dataset.

        Args:
            manifest_path: Path to manifest.json
            split: "train", "val", or "test"
            sample_rate: Audio sample rate (must be 16000)
            max_duration: Max audio duration in seconds
            model_name: Pre-trained model for feature extraction
        """
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Load manifest
        with open(self.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.samples = manifest[split]
        print(f"Loaded {len(self.samples)} samples for {split} split")

        # Setup feature extractor and tokenizer.
        # If model_name looks like a local path that doesn't exist, fail fast with
        # a useful message instead of letting transformers retry it as a HF repo
        # id and surface a misleading 401. We treat anything starting with '.', '/',
        # or 'checkpoints' (or with a backslash / drive letter) as a local path —
        # plain HF ids like "obadx/muaalem-model-v3_2" are passed through.
        looks_local = (
            "\\" in model_name
            or model_name.startswith((".", "/", "checkpoints"))
            or (len(model_name) >= 2 and model_name[1] == ":")
        )
        if looks_local and not Path(model_name).exists():
            raise FileNotFoundError(
                f"Model path not found: {model_name}\n"
                f"  - run adapt_model_for_msa() to recreate checkpoints/msa_model_adapted/\n"
                f"  - or pass an existing checkpoint dir / HF repo id."
            )
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = MSATokenizer()

        # Calculate max features length
        self.max_features = int(
            (self.sample_rate * self.max_duration - 400) / (160 * 2)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.samples[idx]
        audio_path = sample["audio"]
        phonetic_script = sample["phonemes"]

        # Load audio
        try:
            audio_array, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                duration=self.max_duration,
            )
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence if load fails
            audio_array = np.zeros(self.sample_rate * 3)

        # Extract features with consistent padding
        features = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_features,
        )

        # Ensure consistent shape (batch, max_features, 160)
        # Pad if necessary
        if features["input_features"].shape[1] < self.max_features:
            pad_size = self.max_features - features["input_features"].shape[1]
            features["input_features"] = torch.nn.functional.pad(
                features["input_features"],
                (0, 0, 0, pad_size),  # pad last 2 dims: (left, right, top, bottom)
                mode='constant',
                value=0
            )
            features["attention_mask"] = torch.nn.functional.pad(
                features["attention_mask"],
                (0, pad_size),
                mode='constant',
                value=0
            )

        # Tokenize phonemes
        token_ids = self.tokenizer.encode(phonetic_script)

        # CTC: target_length must be <= input_length (logits time dim).
        # features["input_features"] is (batch=1, T_feat, 160) here — squeeze
        # happens below — so the time axis is shape[1], NOT shape[0]. The
        # encoder downsamples by ~2, so logits time dim ≈ T_feat / 2.
        input_time_steps = features["input_features"].shape[1] // 2
        max_label_len = max(1, input_time_steps - 5)

        if len(token_ids) > max_label_len:
            token_ids = token_ids[:max_label_len]

        # Pad to fixed length for batch processing (use actual length)
        if len(token_ids) < 256:
            token_ids = token_ids + [0] * (256 - len(token_ids))

        return {
            "input_features": features["input_features"].squeeze(0),
            "attention_mask": features["attention_mask"].squeeze(0),
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }


def get_data_loaders(
    manifest_path: str | Path,
    batch_size: int = 4,
    num_workers: int = 2,
    model_name: str = "obadx/muaalem-model-v3_2",
):
    """
    Create train/val/test dataloaders.

    Args:
        manifest_path: Path to manifest.json
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        model_name: Pre-trained model name

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MSAPhonemeDataset(
        manifest_path,
        split="train",
        model_name=model_name,
    )
    val_dataset = MSAPhonemeDataset(
        manifest_path,
        split="val",
        model_name=model_name,
    )
    test_dataset = MSAPhonemeDataset(
        manifest_path,
        split="test",
        model_name=model_name,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
