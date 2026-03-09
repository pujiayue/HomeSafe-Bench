#!/bin/bash

# =================================================================
# Script to Deploy Qwen3-VL-30B with vLLM and Run Inference
# =================================================================

# ================= Configuration =================
# Set CUDA devices. Please adjust indices based on your hardware setup.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1  # Example: Using 2 GPUs

PORT=8005
MODEL_NAME="Qwen3-VL-30B"

# IMPORTANT: Replace the path below with the actual path to your downloaded model
# e.g., "/path/to/your/local/models/Qwen3-VL-30B"
MODEL_PATH="/path/to/your/local/models/Qwen3-VL-30B"
DATASET_DIR="/path/to/your/MERGED_DATASETS_FINAL_TimeStamp"

# --- 1. Deep Cleanup of Residual Processes (Fix 'Address already in use') ---
echo "🧹 Cleaning up port $PORT and residual processes..."

# 1.1 Force kill process occupying the specific port
PORT_PID=$(lsof -t -i:$PORT)
if [ ! -z "$PORT_PID" ]; then
    echo "Port $PORT is occupied by PID $PORT_PID. Killing it..."
    kill -9 $PORT_PID
fi

# 1.2 Force kill related Python and Ray processes
# Note: This kills processes for the current user to prevent memory leaks
ray stop --force 2>/dev/null
pkill -u $(whoami) -9 -f "vllm.entrypoints.openai.api_server"
pkill -u $(whoami) -9 -f "multiprocessing.resource_tracker"
pkill -u $(whoami) -9 -f "vllm"

# 1.3 Wait for system resource reclamation
sleep 5

# --- 2. Environment Variables Setup ---
# Force client and server to use the same dummy key
export OPENAI_API_KEY="EMPTY"

# Optimization flags for PyTorch and NCCL to improve stability and performance
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Unset proxies to prevent interference with localhost connections
unset http_proxy
unset https_proxy
unset all_proxy

# --- 3. Launch vLLM Server ---
echo "🚀 Starting Qwen3-VL-30B (TP=2, optimized for 48GB GPUs)..."

# Launch vLLM with specific memory and context optimizations
# Note: --limit-mm-per-prompt restricts multimedia inputs per prompt
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --dtype bfloat16 \
    --api-key "EMPTY" \
    --port $PORT \
    --enable-chunked-prefill \
    --limit-mm-per-prompt '{"image": 10, "video": 1}' \
    --enforce-eager \
    > vllm_qwen3.log 2>&1 &

VLLM_PID=$!

# --- 4. Service Readiness Check ---
echo "⏳ Waiting for API service to start (Check logs: tail -f vllm_qwen3.log)..."
READY=0
for i in {1..200}; do
    # Check health endpoint with the dummy header
    if curl -s -H "Authorization: Bearer EMPTY" http://localhost:$PORT/v1/models > /dev/null; then
        echo "✅ Qwen3-VL Service is ready!"
        READY=1
        break
    fi
    
    # Check if process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ Service failed to start! Last 20 lines of log:"
        tail -n 20 vllm_qwen3.log
        exit 1
    fi
    
    if (( $i % 10 == 0 )); then
        echo "   Waited $((i*5)) seconds..."
    fi
    sleep 5
done

# --- 5. Execute Test Script ---
if [ $READY -eq 1 ]; then
    echo "▶️ Starting Hazard Detection Test..."
    # Pass the model path and port to the python client script
    python3 client_test_qwen3.py "$MODEL_PATH" "$PORT" --video_dir "$DATASET_DIR"
fi

# --- 6. Cleanup ---
echo "🛑 Test finished. Shutting down services..."
kill $VLLM_PID 2>/dev/null
ray stop