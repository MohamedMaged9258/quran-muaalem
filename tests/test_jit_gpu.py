from numba.tests.npyufunc.test_ufunc import dtype
from librosa.core import load

import torch
from transformers import AutoFeatureExtractor
from huggingface_hub import hf_hub_download

from quran_muaalem.decode import ctc_decode

# Configuration
REPO_ID = "obadx/muaalem-v3_2-torchscript-v1"
MODEL_FILE = "model_bf16.pt"
SAMPLE_RATE = 16000
DURATION = 3  # seconds
model_dtype = torch.bfloat16

# Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print(f"Using device: {device}")

# Download model and processor
print("Downloading model...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
model = torch.jit.load(model_path, map_location=device)
model.eval()

print("Downloading processor...")
processor = AutoFeatureExtractor.from_pretrained(REPO_ID, subfolder="processor")


audio_path = "./assets/test.wav"
wave, _ = load(audio_path, sr=SAMPLE_RATE, mono=True)

# Preprocess
inputs = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
input_features = inputs["input_features"].to(device, dtype=model_dtype)
attention_mask = inputs["attention_mask"].to(device)


# Run inference
print("Running inference...")
with torch.no_grad():
    output = model(input_features, attention_mask)

# The output is a tuple; first element is the logits dict
logits_dict = output[0]
print("\nOutput dictionary keys and shapes:")
for key, tensor in logits_dict.items():
    print(f"  {key}: {tensor.shape}")

ph_logits = logits_dict["phonemes"]
ph_probs = torch.nn.functional.softmax(ph_logits, dim=-1).cpu().to(torch.float32)
ph_probs, ph_ids = ph_probs.topk(1, dim=-1)
