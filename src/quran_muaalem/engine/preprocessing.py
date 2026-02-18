import io
import numpy as np
import torch
from typing import Union
from ray import serve
from transformers import AutoFeatureExtractor

AudioInput = Union[bytes, list[float], np.ndarray, torch.Tensor]


def load_audio(audio: AudioInput, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if isinstance(audio, bytes):
        import librosa

        audio, sr = librosa.load(io.BytesIO(audio), sr=target_sr, mono=True)
        return audio, target_sr
    elif isinstance(audio, (list, np.ndarray)):
        return np.array(audio, dtype=np.float32), target_sr
    elif isinstance(audio, torch.Tensor):
        return audio.numpy().astype(np.float32), target_sr
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")


@serve.deployment(
    name="preprocessing",
    ray_actor_options={"num_cpus": 2},
)
class PreprocessingDeployment:
    def __init__(self, model_name_or_path: str = "obadx/muaalem-model-v3_2"):
        self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

    def __call__(self, audio: bytes) -> dict:
        audio_array, sampling_rate = load_audio(audio, target_sr=16000)

        if sampling_rate != 16000:
            raise ValueError(f"Sampling rate must be 16kHz, got {sampling_rate}")

        features = self.processor(
            [audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return {
            "input_features": features["input_features"],
            "attention_mask": features["attention_mask"],
        }
