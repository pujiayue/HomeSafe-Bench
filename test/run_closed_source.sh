#!/bin/bash

# =================================================================
# Script to run Home Safety Warning System Inference
# =================================================================

# 1. Global Configuration
# The Base URL for the API provider
BASE_URL="https://api2.aigcbest.top/v1"

# Path to the dataset directory
VIDEO_DIR="HomeSafe-Bench/DATASETS_TimeStamp"

# 2. Define Models to Evaluate
# You can add or remove models from this list
MODELS=(
    "gemini-3-pro-preview"
    "gpt-5.1-2025-11-13"
    "claude-opus-4-1-20250805"
)

# 3. Main Loop
for MODEL in "${MODELS[@]}"; do
    
    # Assign API Keys based on the model name
    # NOTE: For reproduction, please replace the placeholders below with actual API keys.
    case "$MODEL" in
        "gemini-3-pro-preview")
            API_KEY="YOUR_GEMINI_KEY_HERE"
            ;;
        "gpt-5.1-2025-11-13")
            API_KEY="YOUR_GPT_KEY_HERE"
            ;;
        "claude-opus-4-1-20250805")
            API_KEY="YOUR_CLAUDE_KEY_HERE"
            ;;
        *)
            echo "Error: No API key configured for $MODEL"
            continue
            ;;
    esac

    echo "----------------------------------------------------------------"
    echo "🚀 Starting Inference for: $MODEL"
    echo "----------------------------------------------------------------"

    # Call the Python script with explicit arguments
    python client_closed_souece.py \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --video_dir "$VIDEO_DIR"

    echo "✅ Finished processing $MODEL."
    
    # Sleep to respect rate limits
    echo "Sleeping for 5 seconds..."
    sleep 5
    echo ""
done