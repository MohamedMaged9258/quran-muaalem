"""
MSA Tokenizer - Maps MSA phonemes to token IDs for CTC training
Simplified version: phoneme-only (no linguistic properties)
"""

import json
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer
from .msa_vocab import MSA_PHONEMES, MSA_PHONEME_COUNT

PAD_TOKEN = "[PAD]"
PAD_TOKEN_IDX = 0
UNK_TOKEN = "[UNK]"
UNK_TOKEN_IDX = 1


def build_msa_vocab_json(output_path: str | Path):
    """
    Build MSA vocabulary JSON file compatible with Wav2Vec2CTCTokenizer.

    Output format:
    {
      "phonemes": {
        "[PAD]": 0,
        "[UNK]": 1,
        "ب": 2,
        "ت": 3,
        ...
      }
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = {}
    vocab[PAD_TOKEN] = PAD_TOKEN_IDX
    vocab[UNK_TOKEN] = UNK_TOKEN_IDX

    idx = 2
    for phoneme in MSA_PHONEMES:
        if phoneme not in [PAD_TOKEN, UNK_TOKEN]:
            vocab[phoneme] = idx
            idx += 1

    # Write vocab.json in the format transformers expects
    vocab_dict = {"phonemes": vocab}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ MSA vocabulary saved to {output_path}")
    print(f"   Total tokens: {len(vocab)}")
    return vocab_dict


class MSATokenizer:
    """
    Tokenizer for MSA phoneme-only model.
    Maps MSA phonetic script (e.g., "د َ ر َ س") to token IDs.
    """

    def __init__(self, vocab_dict: dict = None):
        """
        Initialize MSA tokenizer.

        Args:
            vocab_dict: Optional pre-built vocabulary dict.
                       If None, will build from MSA_PHONEMES.
        """
        if vocab_dict is None:
            vocab_dict = self._build_vocab()

        self.vocab = vocab_dict.get("phonemes", {})
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = PAD_TOKEN_IDX
        self.unk_token_id = UNK_TOKEN_IDX

    def _build_vocab(self) -> dict:
        """Build vocabulary from MSA phoneme list."""
        vocab = {}
        vocab[PAD_TOKEN] = PAD_TOKEN_IDX
        vocab[UNK_TOKEN] = UNK_TOKEN_IDX

        idx = 2
        for phoneme in MSA_PHONEMES:
            if phoneme not in [PAD_TOKEN, UNK_TOKEN]:
                vocab[phoneme] = idx
                idx += 1

        return {"phonemes": vocab}

    def encode(self, phonetic_script: str) -> list[int]:
        """
        Convert phonetic script to token IDs.

        Args:
            phonetic_script: String of space-separated phonemes (e.g., "د َ ر َ س")

        Returns:
            List of token IDs
        """
        phonemes = phonetic_script.split()
        token_ids = []
        for phoneme in phonemes:
            token_ids.append(self.vocab.get(phoneme, self.unk_token_id))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to phonetic script.

        Args:
            token_ids: List of token IDs

        Returns:
            Phonetic script (space-separated phonemes)
        """
        phonemes = []
        for token_id in token_ids:
            phoneme = self.reverse_vocab.get(token_id, UNK_TOKEN)
            if phoneme != PAD_TOKEN and phoneme != UNK_TOKEN:
                phonemes.append(phoneme)
        return " ".join(phonemes)

    def batch_encode(self, phonetic_scripts: list[str]) -> dict:
        """
        Batch encode multiple phonetic scripts.

        Args:
            phonetic_scripts: List of phonetic script strings

        Returns:
            Dict with 'input_ids' (list of token ID lists)
        """
        return {
            "input_ids": [self.encode(script) for script in phonetic_scripts]
        }

    def batch_decode(self, token_ids_list: list[list[int]]) -> list[str]:
        """
        Batch decode multiple token ID lists.

        Args:
            token_ids_list: List of token ID lists

        Returns:
            List of phonetic scripts
        """
        return [self.decode(token_ids) for token_ids in token_ids_list]

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)


# Helper function to load pre-trained tokenizer from huggingface model
def load_msa_tokenizer_from_pretrained(model_name_or_path: str):
    """
    Load MSA tokenizer from a huggingface model directory.
    Expects vocab.json to exist in the model directory.
    """
    from transformers import Wav2Vec2CTCTokenizer
    return Wav2Vec2CTCTokenizer.from_pretrained(
        model_name_or_path,
        pad_token=PAD_TOKEN
    )


if __name__ == "__main__":
    # Example usage
    print("Building MSA vocabulary...")
    vocab = build_msa_vocab_json("./msa_vocab.json")

    print("\nTesting tokenizer...")
    tokenizer = MSATokenizer(vocab)

    # Test encoding
    phonetic_text = "د َ ر َ س"
    token_ids = tokenizer.encode(phonetic_text)
    print(f"Phonetic text: {phonetic_text}")
    print(f"Token IDs: {token_ids}")

    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")
