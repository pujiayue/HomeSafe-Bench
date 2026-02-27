#!/bin/bash
# 这里的名称必须与 Python 代码中 MODEL_KEYS 的键（Key）完全一致
MODELS=(
    "gemini"
    "claude"
    "gpt"
)

for MODEL in "${MODELS[@]}"; do
    python local_video_test.py "$MODEL"
    sleep 5 # 每个模型测完休息一下，避免平台限制
done

video_modal_test