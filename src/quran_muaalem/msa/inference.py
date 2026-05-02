"""MSA inference: load checkpoint, transcribe audio, produce per-phoneme alignments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoFeatureExtractor

from ..modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from ..modeling.msa_tokenizer import MSATokenizer, PAD_TOKEN_IDX

# Wav2Vec2-BERT downsamples audio to ~50 Hz at the encoder output:
# feature extractor produces frames every 10 ms, the conformer adapter
# halves that → one logits step ≈ 20 ms.
ENCODER_HOP_SECONDS = 0.02


@dataclass
class PhonemeAlignment:
    phoneme: str
    token_id: int
    start: float  # seconds
    end: float    # seconds
    confidence: float  # max softmax prob over the spanning frames


class MSAInference:
    """Wrapper around the fine-tuned multi-level CTC model for MSA phonemes only."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        sample_rate: int = 16000,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.sample_rate = sample_rate

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"MSA checkpoint not found: {path.resolve()}\n"
                f"  - run `adapt_model_for_msa()` to create checkpoints/msa_model_adapted/, then\n"
                f"  - run training to produce checkpoints/msa_model_v1/best_model/.\n"
                f"  See TRAINING.md for the full pipeline."
            )
        if not (path / "preprocessor_config.json").exists():
            raise FileNotFoundError(
                f"{path}/preprocessor_config.json is missing. The adapter must save the "
                f"feature extractor alongside the model."
            )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(path))
        self.model = (
            Wav2Vec2BertForMultilevelCTC.from_pretrained(str(path), torch_dtype=torch.float32)
            .to(self.device)
            .eval()
        )
        self.tokenizer = MSATokenizer()

    @torch.no_grad()
    def _logits(self, audio: NDArray[np.float32]) -> torch.Tensor:
        feats = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        feats = {k: v.to(self.device) for k, v in feats.items()}
        out = self.model(**feats, return_dict=True)
        # (batch=1, T_enc, 31) -> (T_enc, 31) on CPU for downstream maths.
        return out.logits["phonemes"][0].float().cpu()

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """Return the predicted MSA phoneme string (space-separated)."""
        ids = [a.token_id for a in self.align(audio)]
        return self.tokenizer.decode(ids)

    def blank_ratio(self, audio: NDArray[np.float32]) -> float:
        """Fraction of frames where the blank/pad token is the argmax — useful for
        diagnosing early-training collapse (value close to 1.0 → model outputs silence)."""
        logits = self._logits(audio)
        ids = logits.argmax(dim=-1)
        return float((ids == PAD_TOKEN_IDX).float().mean().item())

    def align(self, audio: NDArray[np.float32]) -> list[PhonemeAlignment]:
        """Return per-phoneme timestamps and confidences via CTC greedy decoding."""
        logits = self._logits(audio)
        probs = torch.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1).tolist()
        confs = probs.max(dim=-1).values.tolist()

        alignments: list[PhonemeAlignment] = []
        prev_id: Optional[int] = None
        run_start: int = 0
        run_confs: list[float] = []

        def flush(end_frame: int) -> None:
            if prev_id is None or prev_id == PAD_TOKEN_IDX or not run_confs:
                return
            phoneme = self.tokenizer.reverse_vocab.get(prev_id, "[UNK]")
            alignments.append(
                PhonemeAlignment(
                    phoneme=phoneme,
                    token_id=prev_id,
                    start=run_start * ENCODER_HOP_SECONDS,
                    end=end_frame * ENCODER_HOP_SECONDS,
                    confidence=float(max(run_confs)),
                )
            )

        for frame_idx, (tok_id, conf) in enumerate(zip(ids, confs)):
            if tok_id != prev_id:
                flush(frame_idx)
                prev_id = tok_id
                run_start = frame_idx
                run_confs = [conf]
            else:
                run_confs.append(conf)
        flush(len(ids))
        return alignments
