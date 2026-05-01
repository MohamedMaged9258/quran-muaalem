"""Gradio UI for the MSA service. Talks to the API over HTTP."""

from __future__ import annotations

import io
import wave

import gradio as gr
import httpx
import numpy as np

from .settings import MSASettings


def _audio_to_wav_bytes(audio: tuple[int, np.ndarray] | None) -> bytes | None:
    """Gradio gives us (sample_rate, np.int16-or-float). Pack into a WAV blob."""
    if audio is None:
        return None
    sr, arr = audio
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def build_ui(settings: MSASettings | None = None) -> gr.Blocks:
    settings = settings or MSASettings()
    api_url = settings.api_url.rstrip("/")
    client = httpx.Client(timeout=120.0)

    def _post(path: str, files: dict, data: dict | None = None) -> dict:
        try:
            resp = client.post(f"{api_url}{path}", files=files, data=data or {})
        except httpx.HTTPError as exc:
            raise gr.Error(f"could not reach API at {api_url}: {exc}")
        if resp.status_code != 200:
            raise gr.Error(f"API error {resp.status_code}: {resp.text}")
        return resp.json()

    def _files(audio) -> dict:
        wav = _audio_to_wav_bytes(audio)
        if wav is None:
            raise gr.Error("please record or upload an audio clip first")
        return {"audio": ("clip.wav", wav, "audio/wav")}

    def transcribe(audio):
        return _post("/transcribe", _files(audio))["phonemes"]

    def align(audio):
        data = _post("/align", _files(audio))
        rows = [
            [a["phoneme"], f"{a['start']:.2f}", f"{a['end']:.2f}", f"{a['confidence']:.2%}"]
            for a in data["alignments"]
        ]
        return data["phonemes"], rows

    def compare(audio, expected_text):
        if not expected_text or not expected_text.strip():
            raise gr.Error("please enter the expected Arabic text")
        data = _post("/compare", _files(audio), {"expected_text": expected_text})
        rows = [[op["kind"], op["expected"], op["predicted"]] for op in data["ops"]]
        summary = (
            f"Matches: {data['matches']}  |  "
            f"Subs: {data['substitutions']}  |  "
            f"Inserts: {data['insertions']}  |  "
            f"Deletes: {data['deletions']}  |  "
            f"PER: {data['phoneme_error_rate']:.1%}"
        )
        return summary, rows

    with gr.Blocks(title="Quran Muaalem — MSA") as demo:
        gr.Markdown("# Quran Muaalem — Modern Standard Arabic")
        gr.Markdown(f"API: `{api_url}` · Model: `{settings.model_path}`")

        with gr.Tab("Transcribe"):
            t_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
            t_btn = gr.Button("Transcribe", variant="primary")
            t_out = gr.Textbox(label="Predicted phonemes", lines=2)
            t_btn.click(transcribe, inputs=t_audio, outputs=t_out)

        with gr.Tab("Align"):
            a_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
            a_btn = gr.Button("Align", variant="primary")
            a_phon = gr.Textbox(label="Predicted phonemes", lines=2)
            a_table = gr.Dataframe(
                headers=["phoneme", "start (s)", "end (s)", "confidence"],
                label="Per-phoneme timing",
                interactive=False,
            )
            a_btn.click(align, inputs=a_audio, outputs=[a_phon, a_table])

        with gr.Tab("Compare"):
            c_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
            c_text = gr.Textbox(label="Expected Arabic text", lines=2, rtl=True)
            c_btn = gr.Button("Compare", variant="primary")
            c_summary = gr.Textbox(label="Summary", lines=1)
            c_table = gr.Dataframe(
                headers=["op", "expected", "predicted"],
                label="Per-position diff",
                interactive=False,
            )
            c_btn.click(compare, inputs=[c_audio, c_text], outputs=[c_summary, c_table])

    return demo


def main() -> None:
    """Console-script entry point: `quran-muaalem-msa-ui`."""
    settings = MSASettings()
    build_ui(settings).launch(server_name=settings.ui_host, server_port=settings.ui_port)


if __name__ == "__main__":
    main()
