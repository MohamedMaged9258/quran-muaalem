"""FastAPI service for the fine-tuned MSA model."""

from dataclasses import asdict
import io

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .compare import compare_phonemes
from .inference import MSAInference
from .phonemize import text_to_phonemes
from .settings import MSASettings


def create_app(settings: MSASettings | None = None) -> FastAPI:
    settings = settings or MSASettings()
    inference = MSAInference(
        model_path=settings.model_path,
        device=settings.device,
        sample_rate=settings.sample_rate,
    )

    app = FastAPI(
        title="Quran Muaalem MSA API",
        description="Modern Standard Arabic phoneme recognition, alignment, and comparison.",
        version="0.1.0",
    )

    async def _load_audio(upload: UploadFile) -> np.ndarray:
        if upload is None:
            raise HTTPException(status_code=400, detail="audio file is required")
        raw = await upload.read()
        try:
            audio, _ = librosa.load(
                io.BytesIO(raw), sr=settings.sample_rate, mono=True
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}")
        return audio.astype(np.float32, copy=False)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_path": settings.model_path,
            "device": str(inference.device),
        }

    @app.post("/debug")
    async def debug(audio: UploadFile = File(...)) -> dict:
        """Diagnostic endpoint: returns blank_ratio and top-3 phonemes per frame.
        A blank_ratio near 1.0 means the model is mostly predicting silence — normal
        for checkpoints trained < 5 epochs on small datasets."""
        wav = await _load_audio(audio)
        ratio = inference.blank_ratio(wav)
        logits = inference._logits(wav)
        import torch
        probs = torch.softmax(logits, dim=-1)
        top3 = probs.topk(3, dim=-1)
        frames = []
        for i in range(min(10, logits.shape[0])):
            frames.append({
                "frame": i,
                "top_tokens": [
                    {
                        "id": int(idx),
                        "phoneme": inference.tokenizer.reverse_vocab.get(int(idx), "[UNK]"),
                        "prob": float(prob),
                    }
                    for idx, prob in zip(top3.indices[i].tolist(), top3.values[i].tolist())
                ],
            })
        return {"blank_ratio": ratio, "first_10_frames": frames}

    @app.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...)) -> dict:
        wav = await _load_audio(audio)
        return {"phonemes": inference.transcribe(wav)}

    @app.post("/align")
    async def align(audio: UploadFile = File(...)) -> dict:
        wav = await _load_audio(audio)
        alignments = inference.align(wav)
        return {
            "phonemes": " ".join(a.phoneme for a in alignments),
            "alignments": [asdict(a) for a in alignments],
        }

    @app.post("/compare")
    async def compare(
        audio: UploadFile = File(...),
        expected_text: str = Form(...),
    ) -> dict:
        wav = await _load_audio(audio)
        alignments = inference.align(wav)
        predicted_phonemes = [a.phoneme for a in alignments]
        expected_phonemes = text_to_phonemes(expected_text)
        if not expected_phonemes:
            raise HTTPException(
                status_code=400,
                detail="expected_text contained no recognizable Arabic letters",
            )
        result = compare_phonemes(expected_phonemes, predicted_phonemes)
        return {
            "alignments": [asdict(a) for a in alignments],
            "expected_phonemes": result.expected_phonemes,
            "predicted_phonemes": result.predicted_phonemes,
            "ops": [asdict(op) for op in result.ops],
            "matches": result.matches,
            "substitutions": result.substitutions,
            "insertions": result.insertions,
            "deletions": result.deletions,
            "phoneme_error_rate": result.phoneme_error_rate,
        }

    return app


def main() -> None:
    """Console-script entry point: `quran-muaalem-msa-api`."""
    import uvicorn

    settings = MSASettings()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
