#!/bin/bash
set -euo pipefail

# Default values
REPO_ID=""
OUTPUT_DIR="/models/muaalem/1"
USER_DTYPE=""
CONFIG_DIR="/opt/configs"  # <-- RESTORED: where config templates are stored

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dtype)
            USER_DTYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--dtype fp16|bf16|fp32] <repo_id> [output_dir]"
            exit 0
            ;;
        --*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$REPO_ID" ]; then
                REPO_ID="$1"
            elif [ "$OUTPUT_DIR" = "/models/muaalem/1" ]; then
                OUTPUT_DIR="$1"
            else
                echo "Too many arguments"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$REPO_ID" ]; then
    echo "❌ Usage: $0 [--dtype fp16|bf16|fp32] <repo_id> [output_dir]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Detect GPU compute capability and choose best dtype (unless overridden)
detect_best_dtype() {
    if [ -n "$USER_DTYPE" ]; then
        echo "$USER_DTYPE"
        return
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        echo >&2 "⚠️  No GPU detected – falling back to fp32"
        echo "fp32"
        return
    fi
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
    if [ -z "$COMPUTE_CAP" ]; then
        echo >&2 "⚠️  Could not read compute capability – falling back to fp32"
        echo "fp32"
        return
    fi
    MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
    echo >&2 "✅ Detected GPU compute capability: $COMPUTE_CAP"
    if [ "$MAJOR" -ge 8 ]; then
        echo "bf16"
    else
        echo "fp16"
    fi
}

BEST_DTYPE=$(detect_best_dtype)
echo "🎯 Selected dtype: $BEST_DTYPE"

# Determine model filename and config file based on selected dtype
case "$BEST_DTYPE" in
    bf16)
        MODEL_FILE="model_bf16.pt"
        CONFIG_FILE="config_bf16.pbtxt"
        ;;
    fp16)
        MODEL_FILE="model_fp16.pt"
        CONFIG_FILE="config_fp16.pbtxt"
        ;;
    fp32)
        MODEL_FILE="model_fp32.pt"
        CONFIG_FILE="config_fp32.pbtxt"
        ;;
    *)
        echo "❌ Unknown dtype: $BEST_DTYPE"
        exit 1
        ;;
esac

# Download model (with fallback chain if preferred not found)
download_model() {
    local target="$OUTPUT_DIR/model.pt"
    local tried=""
    # Order: preferred → fp16 → fp32 (but skip if already tried)
    for candidate in "$MODEL_FILE" "model_fp16.pt" "model_fp32.pt"; do
        if [[ "$tried" == *"$candidate"* ]]; then
            continue
        fi
        tried+="$candidate "
        echo "⏳ Attempting to download $candidate ..."
        if uvx hf download "$REPO_ID" "$candidate" --local-dir "$OUTPUT_DIR" --quiet 2>/dev/null; then
            mv "$OUTPUT_DIR/$candidate" "$target"
            echo "✅ Downloaded $candidate as $target"
            # Update BEST_DTYPE and CONFIG_FILE to match actual downloaded file
            case "$candidate" in
                model_bf16.pt) BEST_DTYPE="bf16"; CONFIG_FILE="config_bf16.pbtxt" ;;
                model_fp16.pt) BEST_DTYPE="fp16"; CONFIG_FILE="config_fp16.pbtxt" ;;
                model_fp32.pt) BEST_DTYPE="fp32"; CONFIG_FILE="config_fp32.pbtxt" ;;
            esac
            return 0
        else
            echo "⚠️  Failed to download $candidate"
        fi
    done
    echo "❌ Could not download any model file from $REPO_ID"
    exit 1
}

download_model

# Copy the matching config file to the model repository root
echo "📝 Installing config for dtype $BEST_DTYPE"
cp "$CONFIG_DIR/$CONFIG_FILE" "/models/muaalem/config.pbtxt"

echo "🎉 Model and config ready in /models/muaalem"
