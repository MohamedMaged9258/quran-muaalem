#!/bin/bash
set -e

# Download the model to the correct Triton model folder
/opt/entry-level.sh obadx/muaalem-v3_2-torchscript-v1 /models/muaalem/1 --dtype fp16

# Start Triton server
exec tritonserver --model-repository=/models --log-verbose=1
