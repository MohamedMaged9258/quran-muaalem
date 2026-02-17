#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Quran Muaalem model to TorchScript (fp32, fp16, bf16) on GPU and upload."""

import os
import torch
from transformers import AutoFeatureExtractor
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from huggingface_hub import HfApi, create_repo, upload_file, login
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable.")

MODEL_ID = "obadx/muaalem-model-v3_2"
REPO_ID = "obadx/muaalem-v3_2-torchscript"
SAMPLE_RATE = 16000
DURATION_SECONDS = 15
DUMMY_SPEECH = torch.randn(DURATION_SECONDS * SAMPLE_RATE)

TEMP_DIR = Path("./export_temp")
TEMP_DIR.mkdir(exist_ok=True)

# Device for export – use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Export device: {device}")


def main():
    login(token=HF_TOKEN)
    print("Logged in.")

    print("Loading original model...")
    base_model = Wav2Vec2BertForMultilevelCTC.from_pretrained(MODEL_ID)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    dummy_input = processor(
        DUMMY_SPEECH, return_tensors="pt", sampling_rate=SAMPLE_RATE
    )
    print("Processor keys:", dummy_input.keys())

    def forward_wrapper(input_features, attention_mask):
        return base_model(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=False,
        )

    try:
        create_repo(repo_id=REPO_ID, exist_ok=True, token=HF_TOKEN)
        print(f"Repository {REPO_ID} ready.")
    except Exception as e:
        print(f"Repo creation failed: {e}")

    api = HfApi()

    def export_to_torchscript(wrapper_func, dummy_dict, precision, filename, device):
        print(f"\nExporting to {precision} on {device}...")
        # Move tensors to device
        input_features = dummy_dict["input_features"].to(device)
        attention_mask = dummy_dict["attention_mask"].to(device)

        # Move model to device and set dtype
        base_model.to(device)
        if precision == "fp16":
            base_model.half()
            input_features = input_features.half()
        elif precision == "bf16":
            base_model.to(dtype=torch.bfloat16)
            input_features = input_features.to(dtype=torch.bfloat16)
        else:  # fp32
            base_model.float()
            input_features = input_features.float()

        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapper_func, (input_features, attention_mask), strict=False
            )
        traced_model.save(str(filename))
        print(f"Saved to {filename}")
        return filename

    files_to_upload = []

    fp32_file = TEMP_DIR / "model_fp32.pt"
    export_to_torchscript(forward_wrapper, dummy_input, "fp32", fp32_file, device)
    files_to_upload.append(fp32_file)

    fp16_file = TEMP_DIR / "model_fp16.pt"
    export_to_torchscript(forward_wrapper, dummy_input, "fp16", fp16_file, device)
    files_to_upload.append(fp16_file)

    bf16_file = TEMP_DIR / "model_bf16.pt"
    export_to_torchscript(forward_wrapper, dummy_input, "bf16", bf16_file, device)
    files_to_upload.append(bf16_file)

    # Upload processor
    processor.save_pretrained(TEMP_DIR / "processor")
    for file in (TEMP_DIR / "processor").glob("*"):
        if file.is_file():
            upload_file(
                path_or_fileobj=str(file),
                path_in_repo=f"processor/{file.name}",
                repo_id=REPO_ID,
                token=HF_TOKEN,
            )
            print(f"Uploaded processor/{file.name}")

    # Upload models
    for file in files_to_upload:
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=file.name,
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )
        print(f"Uploaded {file.name}")

    # Upload README (content same as before)
    # ... (keep your existing README generation)

    print("\nAll done!")


if __name__ == "__main__":
    main()
