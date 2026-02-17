#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Quran Muaalem model to TorchScript (fp32, fp16, bf16) and upload to HF Hub.
Run on your local machine after installing required packages."""

import os
import torch
from transformers import AutoFeatureExtractor
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from huggingface_hub import HfApi, create_repo, upload_file, login
from pathlib import Path

# ======================= CONFIGURATION =======================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "Please set the environment variable HF_TOKEN with your Hugging Face token."
    )

MODEL_ID = "obadx/muaalem-model-v3_2"
REPO_ID = "obadx/muaalem-v3_2-torchscript"
SAMPLE_RATE = 16000
DURATION_SECONDS = 15
DUMMY_SPEECH = torch.randn(DURATION_SECONDS * SAMPLE_RATE)  # 15 seconds at 16 kHz

TEMP_DIR = Path("./export_temp")
TEMP_DIR.mkdir(exist_ok=True)
# ==============================================================


def main():
    login(token=HF_TOKEN)
    print("Logged in to Hugging Face.")

    print("Loading original model...")
    base_model = Wav2Vec2BertForMultilevelCTC.from_pretrained(MODEL_ID)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    dummy_input = processor(
        DUMMY_SPEECH, return_tensors="pt", sampling_rate=SAMPLE_RATE
    )
    print("Processor output keys:", dummy_input.keys())
    print("Base model loaded.")

    # Forward wrapper that forces return_dict=False
    def forward_wrapper(input_features, attention_mask):
        return base_model(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=False,
        )

    # Create target repository
    try:
        create_repo(repo_id=REPO_ID, exist_ok=True, token=HF_TOKEN)
        print(f"Repository {REPO_ID} ready.")
    except Exception as e:
        print(f"Repo creation failed: {e}")

    api = HfApi()

    def export_to_torchscript(wrapper_func, dummy_dict, precision, filename):
        print(f"\nExporting to {precision}...")
        input_features = dummy_dict["input_features"]
        attention_mask = dummy_dict["attention_mask"]

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
        # Convert Path to string for save()
        traced_model.save(str(filename))
        print(f"Saved to {filename}")
        return filename

    files_to_upload = []

    fp32_file = TEMP_DIR / "model_fp32.pt"
    export_to_torchscript(forward_wrapper, dummy_input, "fp32", fp32_file)
    files_to_upload.append(fp32_file)

    fp16_file = TEMP_DIR / "model_fp16.pt"
    export_to_torchscript(forward_wrapper, dummy_input, "fp16", fp16_file)
    files_to_upload.append(fp16_file)

    bf16_file = TEMP_DIR / "model_bf16.pt"
    export_to_torchscript(forward_wrapper, dummy_input, "bf16", bf16_file)
    files_to_upload.append(bf16_file)

    # Save and upload processor files
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

    # Upload TorchScript models
    for file in files_to_upload:
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=file.name,
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )
        print(f"Uploaded {file.name}")

    # Create and upload README
    readme_content = f"""
# Muaalem Model – TorchScript versions

This repository contains the exported TorchScript versions of the [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) model.

## Files
- `model_fp32.pt` – full precision (float32)
- `model_fp16.pt` – half precision (float16)
- `model_bf16.pt` – bfloat16 precision

## Model Output
The model returns a **tuple** whose first element is a dictionary of logits for each CTC head. The dictionary keys correspond to the heads defined in the configuration:
{list(base_model.config.level_to_vocab_size.keys())}

For example, to get phoneme logits: `output[0]['phonemes']`.

## Loading the model in your code

### Automatic precision selection based on GPU
```python
import torch
from huggingface_hub import hf_hub_download

def load_optimized_model(repo_id="{REPO_ID}"):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere or newer
            try:
                model_file = hf_hub_download(repo_id=repo_id, filename="model_bf16.pt")
                dtype = torch.bfloat16
            except:
                model_file = hf_hub_download(repo_id=repo_id, filename="model_fp16.pt")
                dtype = torch.float16
        else:
            model_file = hf_hub_download(repo_id=repo_id, filename="model_fp16.pt")
            dtype = torch.float16
    else:
        model_file = hf_hub_download(repo_id=repo_id, filename="model_fp32.pt")
        dtype = torch.float32

    model = torch.jit.load(model_file)
    model = model.to(dtype)
    model.eval()
    return model, dtype

# Load processor from the same repo
from transformers import AutoFeatureExtractor
processor = AutoFeatureExtractor.from_pretrained(repo_id, subfolder="processor")
model, dtype = load_optimized_model()
print(f"Loaded model with dtype: {{dtype}}")
"""

    readme_file = TEMP_DIR / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    upload_file(
        path_or_fileobj=str(readme_file),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print("README uploaded.")

    print("\nAll done! Repository:", REPO_ID)


if __name__ == "__main__":
    main()
