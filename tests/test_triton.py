#!/usr/bin/env python3
import argparse
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from transformers import AutoFeatureExtractor
import torch
from librosa import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="./assets/test.wav")
    parser.add_argument("--url", type=str, default="triton:8001")
    parser.add_argument("--model", type=str, default="muaalem")
    args = parser.parse_args()

    # Load processor from Hugging Face repo
    repo_id = "obadx/muaalem-v3_2-torchscript"
    print("Downloading processor...")
    processor = AutoFeatureExtractor.from_pretrained(repo_id, subfolder="processor")

    # Load and resample audio (librosa returns float32)
    wave, _ = load(args.audio, mono=True, sr=16000)

    # Preprocess: returns torch tensors (input_features shape: [1, time])
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt", padding=True)

    # Convert to required dtypes
    input_features = inputs["input_features"].to(
        dtype=torch.bfloat16
    )  # model expects bf16
    attention_mask = inputs["attention_mask"].to(
        dtype=torch.int32
    )  # model expects int32

    # Triton client
    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)

    # Prepare input tensors
    # For BF16, we provide the raw uint16 representation (same bit width)
    inp0 = grpcclient.InferInput(
        "input_features__0", list(input_features.shape), "BF16"
    )
    # Convert bf16 tensor to uint16 numpy array (preserves bit pattern)
    inp0.set_data_from_numpy(input_features.view(torch.uint16).numpy())

    inp1 = grpcclient.InferInput(
        "attention_mask__0", list(attention_mask.shape), "INT32"
    )
    inp1.set_data_from_numpy(attention_mask.numpy())

    # Output names (must match config.pbtxt)
    output_names = [
        "ghonna_logits",
        "hams_or_jahr_logits",
        "istitala_logits",
        "itbaq_logits",
        "phonemes_logits",
        "qalqla_logits",
        "safeer_logits",
        "shidda_or_rakhawa_logits",
        "tafashie_logits",
        "tafkheem_or_taqeeq_logits",
        "tikraar_logits",
    ]
    outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

    # Send inference request
    print("Sending inference request...")
    try:
        results = client.infer(
            model_name=args.model, inputs=[inp0, inp1], outputs=outputs
        )
    except InferenceServerException as e:
        print("Inference failed:", e)
        return

    # Retrieve results (Triton returns BF16 as float32 by default)
    logits_dict = {name: results.as_numpy(name) for name in output_names}
    print("Received logits for heads:")
    for name, arr in logits_dict.items():
        print(f"  {name}: shape {arr.shape}, dtype {arr.dtype}")

    # Example: decode phonemes (greedy)
    phonemes_logits = logits_dict["phonemes_logits"]
    pred_ids = np.argmax(phonemes_logits, axis=-1)
    print(f"Phonemes argmax shape: {pred_ids.shape}")
    print(f"First 10 predicted ids: {pred_ids[0][:10]}")


if __name__ == "__main__":
    main()
