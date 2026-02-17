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
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="Data type for input_features (default: fp16)")
    args = parser.parse_args()

    # Map dtype argument to torch and triton types
    dtype_map = {
        "fp16": (torch.float16, "FP16"),
        "bf16": (torch.bfloat16, "BF16"),
        "fp32": (torch.float32, "FP32"),
    }
    torch_dtype, triton_dtype = dtype_map[args.dtype]

    # Load processor from Hugging Face repo
    repo_id = "obadx/muaalem-v3_2-torchscript"
    print("Downloading processor...")
    processor = AutoFeatureExtractor.from_pretrained(repo_id, subfolder="processor")

    # Load and resample audio (librosa returns float32)
    wave, _ = load(args.audio, mono=True, sr=16000)

    # Preprocess: returns torch tensors (input_features shape: [1, time])
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt", padding=True)

    # Convert input_features to selected dtype
    input_features = inputs["input_features"].to(dtype=torch_dtype)
    attention_mask = inputs["attention_mask"].to(dtype=torch.int32)  # always int32

    # Triton client
    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)

    # Prepare input tensors
    inp0 = grpcclient.InferInput(
        "input_features", list(input_features.shape), triton_dtype
    )

    # Handle BF16 special case: need to send as uint16
    if args.dtype == "bf16":
        # bfloat16 has no direct numpy equivalent; send raw uint16 bits
        inp0.set_data_from_numpy(input_features.view(torch.uint16).numpy())
    else:
        # fp16 and fp32 can be sent directly
        inp0.set_data_from_numpy(input_features.numpy())

    inp1 = grpcclient.InferInput(
        "attention_mask", list(attention_mask.shape), "INT32"
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
