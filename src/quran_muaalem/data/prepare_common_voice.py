"""
Prepare Common Voice Arabic dataset for MSA fine-tuning.
Converts text transcriptions to phonemes and creates manifest.json
"""

import json
import csv
from pathlib import Path
from typing import Optional
import librosa
import soundfile as sf
from tqdm import tqdm


class ArabicToPhonemes:
    """
    Convert Arabic text to phonetic script.

    Maps Arabic characters to phonemes used in MSA.
    Example: "درس" → "د َ ر َ س"
    """

    # Mapping of Arabic letters to phonemes
    # Format: Arabic_char → MSA_phoneme
    PHONEME_MAP = {
        # Consonants
        'ء': 'ء',  # hamza
        'ب': 'ب',  # ba
        'ت': 'ت',  # ta
        'ث': 'ث',  # tha
        'ج': 'ج',  # jeem
        'ح': 'ح',  # ha
        'خ': 'خ',  # khah
        'د': 'د',  # dal
        'ذ': 'ذ',  # dhal
        'ر': 'ر',  # ra
        'ز': 'ز',  # zay
        'س': 'س',  # seen
        'ش': 'ش',  # shin
        'ص': 'ص',  # sad
        'ض': 'ض',  # dad
        'ط': 'ط',  # tah
        'ظ': 'ظ',  # zah
        'ع': 'ع',  # ayn
        'غ': 'غ',  # ghain
        'ف': 'ف',  # fa
        'ق': 'ق',  # qaf
        'ك': 'ك',  # kaf
        'ل': 'ل',  # lam
        'م': 'م',  # meem
        'ن': 'ن',  # nun
        'ه': 'ه',  # hah
        'و': 'و',  # waw
        'ي': 'ي',  # ya
        'ة': 'ة',  # ta_marbuta

        # Vowels/Diacritics (already diacritized in text)
        'َ': 'َ',  # fatha (a)
        'ُ': 'ُ',  # damma (u)
        'ِ': 'ِ',  # kasra (i)
        'ْ': 'ْ',  # sukun (silence)
    }

    @staticmethod
    def text_to_phonemes(text: str) -> str:
        """
        Convert Arabic text to phonetic script.

        Rules:
        - Keep each character separated by space
        - Handle diacritics (fatha, damma, kasra, sukun)
        - Skip spaces in input

        Example:
            Input:  "درس"
            Output: "د َ ر َ س" (if diacritized in input)
            Output: "د ر س" (if not diacritized)
        """
        phonemes = []

        for char in text:
            if char in ArabicToPhonemes.PHONEME_MAP:
                phoneme = ArabicToPhonemes.PHONEME_MAP[char]
                phonemes.append(phoneme)
            elif char == ' ':
                # Skip spaces in input - we'll add our own spacing
                continue
            # Skip unknown characters

        # Join with spaces
        return ' '.join(phonemes)

    @staticmethod
    def add_default_vowels(text: str) -> str:
        """
        Add default vowels to undiacritized Arabic text.

        Most Arabic text is written without diacritics.
        This adds a default vowel (fatha) after consonants.

        WARNING: This is a simplification and may not be 100% accurate.
        For production, you'd want proper diacritization tool.

        Example:
            Input:  "درس"
            Output: "د َ ر َ س" (with default fatha)
        """
        # For now, we'll just return the text as-is
        # In production, use: Farasa, Buckwalter, or similar tools
        return text


class CommonVoiceProcessor:
    """Process Common Voice Arabic dataset for MSA training."""

    def __init__(
        self,
        cv_path: str | Path,
        output_dir: str | Path = "datasets/msa_speech",
        sample_rate: int = 16000,
    ):
        """
        Initialize processor.

        Args:
            cv_path: Path to Common Voice extracted directory
            output_dir: Where to save processed audio + manifest
            sample_rate: Target audio sample rate (16000 for model)
        """
        self.cv_path = Path(cv_path)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

        for d in [self.train_dir, self.val_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def process_cv_dataset(self, splits: list[str] = None, max_samples: Optional[int] = None):
        """
        Process Common Voice dataset.

        Args:
            splits: List of TSV files to process (e.g., ["train", "dev", "test"])
                   If None, uses ["train", "dev", "test"]
            max_samples: Limit number of samples per split (for testing)

        Returns:
            List of samples with audio_path and phonemes
        """
        if splits is None:
            splits = ["train", "dev", "test"]

        samples = []
        skipped_total = 0

        for split in splits:
            samples.extend(
                self._process_single_split(split, max_samples)
            )

        return samples

    def _process_single_split(self, split: str, max_samples: Optional[int] = None) -> list:
        """Process a single split (train/dev/test)."""
        # Handle different possible TSV names
        tsv_path = self.cv_path / f"{split}.tsv"
        if not tsv_path.exists():
            tsv_path = self.cv_path / f"{split}ed.tsv"  # validated.tsv

        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found for split '{split}' in {self.cv_path}")

        clips_dir = self.cv_path / "clips"
        if not clips_dir.exists():
            raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

        samples = []
        skipped = 0
        processed_count = 0

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for idx, row in enumerate(tqdm(reader, desc=f"Processing {split}")):
                if max_samples and processed_count >= max_samples:
                    break

                try:
                    # Get audio file path
                    mp3_file = clips_dir / row['path']
                    if not mp3_file.exists():
                        skipped += 1
                        continue

                    # Get text transcription (different column names possible)
                    text = row.get('sentence') or row.get('text', '').strip()
                    if not text:
                        skipped += 1
                        continue

                    # Convert text to phonemes
                    phonemes = ArabicToPhonemes.text_to_phonemes(text)
                    if not phonemes:
                        skipped += 1
                        continue

                    # Load and convert audio to WAV @ 16kHz
                    try:
                        audio, sr = librosa.load(
                            str(mp3_file),
                            sr=self.sample_rate,
                            mono=True,
                        )

                        # Check duration (filter out very short clips)
                        duration = len(audio) / self.sample_rate
                        if duration < 0.5:  # Skip clips < 0.5 seconds
                            skipped += 1
                            continue
                        if duration > 30:  # Skip clips > 30 seconds
                            skipped += 1
                            continue

                    except Exception as e:
                        skipped += 1
                        continue

                    # Map CV split to our split structure
                    if split == "train":
                        split_dir = self.train_dir
                        split_name = "train"
                    elif split == "dev":
                        split_dir = self.val_dir
                        split_name = "val"
                    elif split == "test":
                        split_dir = self.test_dir
                        split_name = "test"
                    else:
                        skipped += 1
                        continue

                    # Save audio as WAV
                    output_filename = f"audio_{processed_count:06d}.wav"
                    output_path = split_dir / output_filename

                    sf.write(str(output_path), audio, self.sample_rate)

                    # Record sample
                    sample = {
                        "audio": str(output_path),
                        "phonemes": phonemes,
                        "original_text": text,
                        "split": split_name,
                    }
                    samples.append(sample)
                    processed_count += 1

                except Exception as e:
                    skipped += 1
                    continue

        print(f"  Processed {split}: {processed_count} samples ({skipped} skipped)")
        return samples

    def create_manifest(self, samples: list) -> dict:
        """
        Create manifest.json for training.

        Args:
            samples: List of sample dicts

        Returns:
            Manifest dict with train/val/test splits
        """
        manifest = {
            "train": [],
            "val": [],
            "test": [],
        }

        for sample in samples:
            split = sample.pop("split")
            manifest[split].append({
                "audio": sample["audio"],
                "phonemes": sample["phonemes"],
            })

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"  Manifest saved to {manifest_path}")
        print(f"   Train: {len(manifest['train'])} samples")
        print(f"   Val: {len(manifest['val'])} samples")
        print(f"   Test: {len(manifest['test'])} samples")

        return manifest


def main():
    """
    Example: Process Common Voice Arabic dataset.

    Before running:
    1. Download Common Voice Arabic from: https://commonvoice.mozilla.org/en/datasets
    2. Extract to: datasets/common_voice_ar/
    3. Run this script

    This will process train/dev/test splits automatically.
    """

    # Path to extracted Common Voice directory
    cv_path = Path("datasets/common_voice_ar")

    if not cv_path.exists():
        print(f"  ERROR: Common Voice path not found: {cv_path}")
        print("Please download from: https://commonvoice.mozilla.org/en/datasets")
        return

    # Process dataset
    processor = CommonVoiceProcessor(
        cv_path=cv_path,
        output_dir="datasets/msa_speech",
        sample_rate=16000,
    )

    # Convert all splits (train, dev, test)
    print("Processing Common Voice Arabic dataset...")
    print("This may take 2-3 hours depending on your system...\n")

    samples = processor.process_cv_dataset(
        splits=["train", "dev", "test"],
        max_samples=None,  # Process all samples (or set limit for testing)
    )

    # Create manifest
    manifest = processor.create_manifest(samples)

    print(f"\n  Dataset ready for training!")
    print(f"   Location: {processor.output_dir}")
    print(f"   Train samples: {len(manifest['train'])}")
    print(f"   Val samples: {len(manifest['val'])}")
    print(f"   Test samples: {len(manifest['test'])}")
    print(f"   Total samples: {sum(len(v) for v in manifest.values())}")


if __name__ == "__main__":
    main()
