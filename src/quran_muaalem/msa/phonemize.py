"""Convert Arabic text to a list of MSA phonemes.

Mirrors the mapping used during data preparation
(`quran_muaalem.data.prepare_common_voice.ArabicToPhonemes`),
but exposes it as a small standalone helper that does not pull in librosa /
soundfile / tqdm.

Note: the 31-class MSA inventory does NOT include alif `ا` or alif maksura
`ى` — these are long-vowel markers that were dropped during training, so the
model never predicts them. This phonemizer drops them too so that
`predicted` and `expected` sequences are directly comparable.
"""

# Arabic character -> MSA phoneme. Identity-mapped because we treat each
# letter / diacritic as its own phoneme symbol in the 31-class inventory.
PHONEME_MAP: dict[str, str] = {
    # Consonants
    "ء": "ء", "ب": "ب", "ت": "ت", "ث": "ث", "ج": "ج", "ح": "ح", "خ": "خ",
    "د": "د", "ذ": "ذ", "ر": "ر", "ز": "ز", "س": "س", "ش": "ش", "ص": "ص",
    "ض": "ض", "ط": "ط", "ظ": "ظ", "ع": "ع", "غ": "غ", "ف": "ف", "ق": "ق",
    "ك": "ك", "ل": "ل", "م": "م", "ن": "ن", "ه": "ه", "و": "و", "ي": "ي",
    "ة": "ة",
    # Diacritics
    "َ": "َ", "ُ": "ُ", "ِ": "ِ", "ْ": "ْ",
}


def text_to_phonemes(text: str) -> list[str]:
    """Convert Arabic text to a list of MSA phoneme tokens.

    Unmapped characters (Latin, punctuation, tatweel, etc.) are dropped.
    """
    return [PHONEME_MAP[ch] for ch in text if ch in PHONEME_MAP]
