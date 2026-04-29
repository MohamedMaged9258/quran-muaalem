import torch
import litserve as ls

from .serve import QuranMuaalemAPI
from .settings import EngineSettings


def main():
    engine_settings = EngineSettings()

    # Override accelerator to CPU if CUDA is not available
    accelerator = engine_settings.accelerator
    if accelerator == "cuda" and not torch.cuda.is_available():
        accelerator = "cpu"

    # Instantiate the API with engine_settings
    api = QuranMuaalemAPI(
        model_name_or_path=engine_settings.model_name_or_path,
        dtype=engine_settings.torch_dtype,
        max_audio_seconds=engine_settings.max_audio_seconds,
        max_batch_size=engine_settings.max_batch_size,
        batch_timeout=engine_settings.batch_timeout,
    )

    # Create the LitServer with CPU accelerator if needed
    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=engine_settings.devices,
        timeout=engine_settings.timeout,
        workers_per_device=engine_settings.workers_per_device,
    )

    # Run the server
    server.run(port=engine_settings.port)


if __name__ == "__main__":
    main()
