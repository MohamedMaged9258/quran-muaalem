"""
Modern Standard Arabic (MSA) Phoneme Vocabulary
Reference: Standard Arabic phonetic inventory
"""

# MSA consonants (28 total)
CONSONANTS = {
    # Bilabial
    "ب": "ba",      # voiced bilabial stop
    "ف": "fa",      # voiceless labiodental fricative
    "م": "ma",      # voiced bilabial nasal

    # Alveolar/Dental
    "ت": "ta",      # voiceless alveolar stop
    "ث": "tha",     # voiceless dental fricative
    "د": "dal",     # voiced alveolar stop
    "ذ": "dhal",    # voiced dental fricative
    "ن": "nun",     # voiced alveolar nasal
    "ل": "lam",     # voiced alveolar lateral
    "ر": "ra",      # voiced alveolar trill
    "س": "seen",    # voiceless alveolar fricative
    "ز": "zay",     # voiced alveolar fricative

    # Postalveolar/Palatal
    "ش": "shin",    # voiceless postalveolar fricative
    "ج": "jeem",    # voiced postalveolar affricate
    "ي": "ya",      # voiced palatal approximant

    # Velar
    "ك": "kaf",     # voiceless velar stop
    "ق": "qaf",     # voiceless uvular stop
    "غ": "ghain",   # voiced uvular fricative
    "خ": "khah",    # voiceless velar fricative
    "ع": "ayn",     # voiced pharyngeal fricative
    "ح": "ha",      # voiceless pharyngeal fricative
    "ه": "hah",     # voiceless glottal fricative
    "ء": "hamza",   # glottal stop
    "و": "waw",     # voiced labial-velar approximant
}

# MSA vowels and diacritics (5 total)
VOWELS_DIACRITICS = {
    "َ": "fatha",   # /a/ - short open front vowel
    "ُ": "damma",   # /u/ - short close back vowel
    "ِ": "kasra",   # /i/ - short close front vowel
    "ْ": "sukun",   # silence/no vowel
    "ة": "ta_marbuta",  # feminine marker (alternative ه)
}

# Special tokens for silence/padding
SPECIAL_TOKENS = {
    "[PAD]": "padding",
    "[UNK]": "unknown",
}

# Complete MSA phoneme inventory
MSA_PHONEMES = {**CONSONANTS, **VOWELS_DIACRITICS, **SPECIAL_TOKENS}

# Reverse mapping (for decoding)
MSA_PHONEME_CODES = {v: k for k, v in {**CONSONANTS, **VOWELS_DIACRITICS}.items()}

# Total phoneme count
MSA_PHONEME_COUNT = len(MSA_PHONEMES)

print(f"MSA Phoneme Inventory:")
print(f"  - Consonants: {len(CONSONANTS)}")
print(f"  - Vowels/Diacritics: {len(VOWELS_DIACRITICS)}")
print(f"  - Special tokens: {len(SPECIAL_TOKENS)}")
print(f"  - Total: {MSA_PHONEME_COUNT}")
