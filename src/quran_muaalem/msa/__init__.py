"""MSA inference, comparison, and serving stack.

Public surface mirrors the runtime: settings, the MSAInference class, the
text→phoneme helper used at the diff boundary, and the API/UI factories.
"""

from .compare import CompareResult, DiffOp, compare_phonemes
from .inference import MSAInference, PhonemeAlignment
from .phonemize import text_to_phonemes
from .settings import MSASettings

__all__ = [
    "CompareResult",
    "DiffOp",
    "MSAInference",
    "MSASettings",
    "PhonemeAlignment",
    "compare_phonemes",
    "text_to_phonemes",
]
