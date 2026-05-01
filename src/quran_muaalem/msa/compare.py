"""Compare a predicted phoneme sequence to an expected one.

Uses `difflib.SequenceMatcher` to produce a diff that classifies each
position as `match`, `substitute`, `insert` (extra in prediction), or
`delete` (missing from prediction). Also computes Phoneme Error Rate.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class DiffOp:
    kind: str  # "match" | "substitute" | "insert" | "delete"
    expected: str  # "" for inserts
    predicted: str  # "" for deletes


@dataclass
class CompareResult:
    expected_phonemes: list[str]
    predicted_phonemes: list[str]
    ops: list[DiffOp]
    matches: int
    substitutions: int
    insertions: int
    deletions: int
    phoneme_error_rate: float  # (S + D + I) / max(1, len(expected))


def compare_phonemes(expected: list[str], predicted: list[str]) -> CompareResult:
    matcher = SequenceMatcher(a=expected, b=predicted, autojunk=False)
    ops: list[DiffOp] = []
    matches = subs = inserts = deletes = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ops.append(DiffOp("match", expected[i1 + k], predicted[j1 + k]))
            matches += i2 - i1
        elif tag == "replace":
            # Pair up overlapping positions, then any leftovers are pure ins/del.
            overlap = min(i2 - i1, j2 - j1)
            for k in range(overlap):
                ops.append(DiffOp("substitute", expected[i1 + k], predicted[j1 + k]))
                subs += 1
            for k in range(overlap, i2 - i1):
                ops.append(DiffOp("delete", expected[i1 + k], ""))
                deletes += 1
            for k in range(overlap, j2 - j1):
                ops.append(DiffOp("insert", "", predicted[j1 + k]))
                inserts += 1
        elif tag == "delete":
            for k in range(i2 - i1):
                ops.append(DiffOp("delete", expected[i1 + k], ""))
                deletes += 1
        elif tag == "insert":
            for k in range(j2 - j1):
                ops.append(DiffOp("insert", "", predicted[j1 + k]))
                inserts += 1

    per = (subs + deletes + inserts) / max(1, len(expected))
    return CompareResult(
        expected_phonemes=list(expected),
        predicted_phonemes=list(predicted),
        ops=ops,
        matches=matches,
        substitutions=subs,
        insertions=inserts,
        deletions=deletes,
        phoneme_error_rate=per,
    )
