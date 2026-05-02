"""Modern Standard Arabic phoneme vocabulary.

Reference: standard MSA phonetic inventory.
The full inventory is 35 tokens: 28 consonants + 5 vowels/diacritics +
2 special ([PAD]=blank, [UNK]). [PAD] doubles as the CTC blank.
"""

# 28 MSA consonants (matches the canonical Arabic alphabet, including the
# 4 emphatics ص ض ط ظ that are distinct phonemes in MSA).
CONSONANTS = {
    # Bilabial / labiodental
    "ب": "ba",      # voiced bilabial stop
    "ف": "fa",      # voiceless labiodental fricative
    "م": "ma",      # voiced bilabial nasal

    # Alveolar / dental
    "ت": "ta",      # voiceless alveolar stop
    "ث": "tha",     # voiceless dental fricative
    "د": "dal",     # voiced alveolar stop
    "ذ": "dhal",    # voiced dental fricative
    "ن": "nun",     # voiced alveolar nasal
    "ل": "lam",     # voiced alveolar lateral
    "ر": "ra",      # voiced alveolar trill
    "س": "seen",    # voiceless alveolar fricative
    "ز": "zay",     # voiced alveolar fricative

    # Emphatic (pharyngealized) consonants — distinct phonemes in MSA.
    "ص": "sad",     # emphatic voiceless alveolar fricative
    "ض": "dad",     # emphatic voiced alveolar stop
    "ط": "tah",     # emphatic voiceless alveolar stop
    "ظ": "zah",     # emphatic voiced dental fricative

    # Postalveolar / palatal
    "ش": "shin",    # voiceless postalveolar fricative
    "ج": "jeem",    # voiced postalveolar affricate
    "ي": "ya",      # voiced palatal approximant

    # Velar / uvular / pharyngeal / glottal
    "ك": "kaf",     # voiceless velar stop
    "ق": "qaf",     # voiceless uvular stop
    "غ": "ghain",   # voiced uvular fricative
    "خ": "khah",    # voiceless velar fricative
    "ع": "ayn",     # voiced pharyngeal fricative
    "ح": "hha",     # voiceless pharyngeal fricative (distinct from ه)
    "ه": "hah",     # voiceless glottal fricative
    "ء": "hamza",   # glottal stop
    "و": "waw",     # voiced labial-velar approximant
}

# 5 vowels / diacritics
VOWELS_DIACRITICS = {
    "َ": "fatha",   # /a/ - short open front vowel
    "ُ": "damma",   # /u/ - short close back vowel
    "ِ": "kasra",   # /i/ - short close front vowel
    "ْ": "sukun",   # silence / no vowel
    "ة": "ta_marbuta",  # feminine marker (alternative ه)
}

# Special tokens. [PAD]=0 also doubles as the CTC blank.
SPECIAL_TOKENS = {
    "[PAD]": "padding",
    "[UNK]": "unknown",
}

MSA_PHONEMES = {**CONSONANTS, **VOWELS_DIACRITICS, **SPECIAL_TOKENS}
MSA_PHONEME_CODES = {v: k for k, v in {**CONSONANTS, **VOWELS_DIACRITICS}.items()}
MSA_PHONEME_COUNT = len(MSA_PHONEMES)
