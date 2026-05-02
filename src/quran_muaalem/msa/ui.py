"""Gradio UI for the MSA service. Talks to the API over HTTP."""

from __future__ import annotations

import mimetypes
from pathlib import Path

import gradio as gr
import httpx

from .settings import MSASettings


def build_ui(settings: MSASettings | None = None) -> gr.Blocks:
    settings = settings or MSASettings()
    api_url = settings.api_url.rstrip("/")
    # 120 s is plenty for CPU inference on a 15 s clip; CUDA is much faster.
    client = httpx.Client(timeout=120.0)

    def _post(path: str, files: dict, data: dict | None = None) -> dict:
        try:
            resp = client.post(f"{api_url}{path}", files=files, data=data or {})
        except httpx.HTTPError as exc:
            raise gr.Error(f"could not reach API at {api_url}: {exc}")
        if resp.status_code != 200:
            raise gr.Error(f"API error {resp.status_code}: {resp.text}")
        return resp.json()

    def _files(audio_path: str | None) -> dict:
        if not audio_path:
            raise gr.Error("please record or upload an audio clip first")
        path = Path(audio_path)
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise gr.Error(f"could not read audio file: {exc}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return {"audio": (path.name, blob, mime)}

    EMPTY_DIFF = [["—", "—", "—"]]
    EMPTY_ALIGN = [["—", "—", "—", "—"]]

    def analyze(audio, expected_text):
        files = _files(audio)
        expected_text = (expected_text or "").strip()

        if expected_text:
            data = _post("/compare", files, {"expected_text": expected_text})
            phonemes = " ".join(data["predicted_phonemes"]) or "(no phonemes predicted — model may need more training epochs)"
            align_rows = [
                [a["phoneme"], f"{a['start']:.2f}", f"{a['end']:.2f}", f"{a['confidence']:.2%}"]
                for a in data["alignments"]
            ] or EMPTY_ALIGN
            diff_rows = [[op["kind"], op["expected"], op["predicted"]] for op in data["ops"]] or EMPTY_DIFF
            summary = (
                f"Matches {data['matches']}  ·  "
                f"Subs {data['substitutions']}  ·  "
                f"Inserts {data['insertions']}  ·  "
                f"Deletes {data['deletions']}  ·  "
                f"PER {data['phoneme_error_rate']:.1%}"
            )
            return phonemes, align_rows, summary, diff_rows

        data = _post("/align", files)
        align_rows = [
            [a["phoneme"], f"{a['start']:.2f}", f"{a['end']:.2f}", f"{a['confidence']:.2%}"]
            for a in data["alignments"]
        ] or EMPTY_ALIGN
        phonemes = data["phonemes"]
        if not phonemes:
            phonemes = "(no phonemes predicted — model may need more training epochs)"
        return phonemes, align_rows, "(no expected text — comparison skipped)", EMPTY_DIFF

    with gr.Blocks(title="Quran Muaalem — MSA") as demo:
        gr.Markdown("# Quran Muaalem — Modern Standard Arabic")
        gr.Markdown(
            f"API: `{api_url}` · Model: `{settings.model_path}`  \n"
            "Upload or record audio. Optionally provide expected text to get a phoneme-level diff."
        )

        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
            expected = gr.Textbox(
                label="Expected Arabic text (optional)",
                placeholder="اكتب النص المتوقع هنا للحصول على مقارنة",
                lines=3,
                rtl=True,
            )
        run_btn = gr.Button("Analyze", variant="primary")

        phonemes_out = gr.Textbox(label="Predicted phonemes", lines=2)
        align_table = gr.Dataframe(
            headers=["phoneme", "start (s)", "end (s)", "confidence"],
            label="Per-phoneme alignment",
            interactive=False,
        )
        summary_out = gr.Textbox(label="Comparison summary", lines=1)
        diff_table = gr.Dataframe(
            headers=["op", "expected", "predicted"],
            label="Per-position diff",
            interactive=False,
        )

        run_btn.click(
            analyze,
            inputs=[audio, expected],
            outputs=[phonemes_out, align_table, summary_out, diff_table],
        )

    return demo


def main() -> None:
    """Console-script entry point: `quran-muaalem-msa-ui`."""
    settings = MSASettings()
    build_ui(settings).launch(server_name=settings.ui_host, server_port=settings.ui_port)


if __name__ == "__main__":
    main()
