"""
Adapt pre-trained Quran Muaalem model for MSA phoneme recognition.

Problem: Model outputs 43 phoneme classes (Quranic) but we want 35 (MSA).
Solution: Resize the CTC phoneme head to output 35 classes instead.

The new size is read from `MSA_PHONEME_COUNT` so the layer always matches
the active vocabulary (28 consonants + 5 vowels/diacritics + [PAD] + [UNK]).

This creates a new model checkpoint with the adapted architecture.
"""

import torch
from pathlib import Path
from transformers import AutoFeatureExtractor
from .modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from .msa_vocab import MSA_PHONEME_COUNT


def adapt_model_for_msa(
    pretrained_model_path: str = "obadx/muaalem-model-v3_2",
    output_path: str = "checkpoints/msa_model_adapted",
    device: str = "cpu",
) -> Wav2Vec2BertForMultilevelCTC:
    """
    Adapt pre-trained Quranic model for MSA.

    Steps:
    1. Load pre-trained model (with 43 phoneme classes)
    2. Resize phoneme CTC head to MSA_PHONEME_COUNT classes
    3. Warm-start the new layer from the overlap rows of the old head
    4. Save the adapted model + feature extractor

    Args:
        pretrained_model_path: HuggingFace model ID or local path
        output_path: Where to save adapted model
        device: Device to load model on (cpu or cuda)

    Returns:
        Adapted model ready for MSA training
    """

    print("=" * 70)
    print("Adapting Model for MSA Phoneme Recognition")
    print("=" * 70)

    # Step 1: Load pre-trained model
    print("\n1. Loading pre-trained Quranic model...")
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    model = model.to(device)
    print(f"   Loaded from: {pretrained_model_path}")

    # Step 2: Check current phoneme output size
    print("\n2. Analyzing model structure...")
    if "phonemes" not in model.level_to_lm_head:
        raise ValueError("Model doesn't have 'phonemes' output head")
    old_phoneme_head = model.level_to_lm_head["phonemes"]

    input_dim = old_phoneme_head.in_features
    old_output_dim = old_phoneme_head.out_features
    new_output_dim = MSA_PHONEME_COUNT

    print(f"   Input dimension: {input_dim}")
    print(f"   Old output dimension (Quranic): {old_output_dim}")
    print(f"   New output dimension (MSA): {new_output_dim}")

    # Step 3: Resize phoneme head
    print("\n3. Resizing phoneme CTC head...")
    # Create new linear layer with MSA vocabulary size
    new_phoneme_head = torch.nn.Linear(input_dim, new_output_dim)

    # Initialize with small random values (or copy overlap from old)
    torch.nn.init.normal_(new_phoneme_head.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(new_phoneme_head.bias)

    # Warm-start: copy overlap rows from the old head into the new one.
    # The first `new_output_dim` rows of the old (43-class) Quranic head share
    # row positions with the new MSA head, so this gives the optimizer a head
    # start over a pure random init.
    if old_output_dim >= new_output_dim:
        print(f"   Copying overlap weights ({new_output_dim}/{old_output_dim})")
        with torch.no_grad():
            new_phoneme_head.weight[:new_output_dim] = old_phoneme_head.weight[:new_output_dim]
            new_phoneme_head.bias[:new_output_dim] = old_phoneme_head.bias[:new_output_dim]

    # Replace the old head with new one
    model.level_to_lm_head["phonemes"] = new_phoneme_head
    print(f"   Phoneme head resized: {old_output_dim} -> {new_output_dim}")

    # Step 4: Update model config
    print("\n4. Updating model configuration...")
    model.config.level_to_vocab_size["phonemes"] = new_output_dim
    print(f"   Updated vocab size in config")

    # Step 5: Save adapted model (weights + config + feature extractor)
    print("\n5. Saving adapted model...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # Also copy the feature extractor / preprocessor config so the
    # checkpoint is self-contained for AutoFeatureExtractor.from_pretrained.
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_path)
    feature_extractor.save_pretrained(output_path)
    print(f"   Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Model Adaptation Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Use adapted model for training:")
    print(f"   python3.14 -m uv run python train_msa_simple.py \\")
    print(f"     --model_name {output_path} \\")
    print(f"     --device cuda --epochs 20")
    print(f"\n2. The model now outputs {new_output_dim} MSA phonemes instead of 43 Quranic")
    print(f"3. Training will adapt these {new_output_dim} classes to your MSA data")

    return model


if __name__ == "__main__":
    # Example usage
    model = adapt_model_for_msa(
        pretrained_model_path="obadx/muaalem-model-v3_2",
        output_path="checkpoints/msa_model_adapted",
        device="cpu",
    )
    print("\nModel ready for MSA training!")
