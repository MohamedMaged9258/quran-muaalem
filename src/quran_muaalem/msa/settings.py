"""Pydantic settings for the MSA service (API + UI)."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class MSASettings(BaseSettings):
    """Configuration for the MSA inference service.

    Reads from environment variables with the `MSA_` prefix, e.g.
    `MSA_MODEL_PATH=checkpoints/msa_model_v1/best_model`.
    """

    model_config = {"env_prefix": "MSA_"}

    model_path: str = Field(
        default="checkpoints/msa_model_adapted",
        description=(
            "Path to a local MSA checkpoint. Defaults to the head-resized "
            "(but untrained) adapter so the service can start before training. "
            "Set MSA_MODEL_PATH=checkpoints/msa_model_v1/best_model after training."
        ),
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Inference device. Falls back to CPU automatically if CUDA is unavailable.",
    )
    sample_rate: int = Field(default=16000, description="Audio sample rate (must be 16000).")
    api_host: str = Field(default="0.0.0.0", description="MSA API bind address.")
    api_port: int = Field(default=8010, description="MSA API port (separate from upstream 8000/8001).")
    api_url: str = Field(
        default="http://127.0.0.1:8010",
        description="Where the UI looks for the API.",
    )
    ui_host: str = Field(default="0.0.0.0", description="MSA UI bind address.")
    ui_port: int = Field(default=7870, description="MSA UI port (separate from upstream 7860).")
