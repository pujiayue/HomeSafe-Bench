#!/bin/bash

# ================= Configuration Area =================
# Specify GPU ID (vLLM automatically assigns based on visible devices)
WHICH_GPUS="1" 
PORT=8000

# Directory containing your video dataset (Required for the Python client)
VIDEO_DIR="./videos" 
# Directory for output results
OUTPUT_DIR="./results"

# List of models to test
MODELS=(
    "YOUR_LOCAL_MODEL_PATH1"
    "YOUR_LOCAL_MODEL_PATH2"
)

export CUDA_VISIBLE_DEVICES=$WHICH_GPUS
# Enhance multi-card synchronization stability
export NCCL_IB_DISABLE=1
export VLLM_USE_V1=0  # Force use of the more stable legacy engine core

# Force cleanup before loop (Uncomment if needed)
#echo "🧹 Global Cleanup: Killing all residual vLLM processes..."
#pkill -f "vllm.entrypoints.openai.api_server"
#sleep 2

for MODEL_PATH in "${MODELS[@]}"; do
    echo "========================================================="
    echo "🚀 Preparing to launch model: $MODEL_PATH"

    # 1. Reset default variables
    MM_ARGS=""
    EXTRA_VLLM_ARGS=""
    CURRENT_MAX_LEN=32768  
    GPU_MEM=0.8
    TP=1
    
    # Clear previous environment variable interference
    unset VLLM_QWEN_MAX_PIXELS
    unset VLLM_ALLOW_LONG_MAX_MODEL_LEN

    # 2. Model-specific configuration logic
    if [[ "$MODEL_PATH" == *"InternVL"* ]]; then
        export VLLM_INTERNVL_MAX_DYNAMIC_PATCH=1
        MM_ARGS='--mm-processor-kwargs {"max_dynamic_patch":1}'

    # === Qwen 2B Configuration ===
    elif [[ "$MODEL_PATH" == *"Qwen3-VL-2B"* ]]; then
        echo "⚠️  Qwen3-2B detected (Shared GPU mode)..."
        GPU_MEM=0.6
        CURRENT_MAX_LEN=40960
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    
    # === Qwen 4B Configuration ===
    elif [[ "$MODEL_PATH" == *"Qwen3-VL-4B"* ]]; then
        echo "⚠️  Qwen3-4B detected, optimizing visual token count..."
        
        # 1. Keep this limit, otherwise it bounces back to 80k tokens
        export VLLM_QWEN_MAX_PIXELS=602112
        
        # 2. Change 32768 to 40960 (40k) to accommodate 35560 input
        CURRENT_MAX_LEN=40960
        
        # 3. Must enable this flag to allow lengths exceeding model defaults
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        
        # 4. GPU mem 0.8 should suffice; if it crashes, try 0.7
        GPU_MEM=0.8

    elif [[ "$MODEL_PATH" == *"Qwen3-VL-8B"* ]]; then
        echo "🚀 Qwen3-8B special adaptation"
        export VLLM_QWEN_MAX_PIXELS=200704
        CURRENT_MAX_LEN=40960
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

    elif [[ "$MODEL_PATH" == *"llava-hf--LLaVA-NeXT-Video"* ]]; then
        CURRENT_MAX_LEN=8192
        GPU_MEM=0.9
        EXTRA_VLLM_ARGS='--limit-mm-per-prompt {"video":1}'

    elif [[ "$MODEL_PATH" == *"llava-onevision"* ]]; then
        echo "🚀 LLaVA-OneVision deep adaptation"
        CURRENT_MAX_LEN=32768
        
        # [Modified] Removed "image_grid_pinpoints":[[336,336]] which caused crashes
        MM_ARGS='--mm-processor-kwargs {"image_aspect_ratio":"square","max_dynamic_patch":1}'
        
        if [[ "$MODEL_PATH" == *"0.5b"* ]]; then
            # 0.5B model is very small, no need for dual cards or high memory
            GPU_MEM=0.6 
            TP=1
        fi
    fi

    # 3. TP Parallel Configuration (4B is small, TP=1 is enough unless splitting memory)
    if [[ "$MODEL_PATH" == *"30B"* ]] || [[ "$MODEL_PATH" == *"14B"* ]] || [[ "$MODEL_PATH" == *"8B"* ]] || [[ "$MODEL_PATH" == *"7B"* ]]; then
        echo "Medium/Large model detected, enabling TP=2 mode"
        TP=2
        GPU_MEM=0.9
    fi
    
    # 4. Launch vLLM Serve
    echo "⚙️  Launch Command: vllm serve ... (Mem: $GPU_MEM, TP: $TP, MaxLen: $CURRENT_MAX_LEN)"
    vllm serve "$MODEL_PATH" \
        --trust-remote-code \
        --port $PORT \
        --served-model-name "internvl-model-test" \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $CURRENT_MAX_LEN \
        --limit-mm-per-prompt '{"image":40}' \
        --dtype bfloat16 \
        --distributed-executor-backend mp \
        $EXTRA_VLLM_ARGS \
        $MM_ARGS > vllm_server.log 2>&1 &

    VLLM_PID=$!

    # 5. Health Check
    echo "⏳ Waiting for API service to be ready (PID: $VLLM_PID)..."
    READY=0
    for i in {1..150}; do
        if curl -s http://localhost:$PORT/v1/models > /dev/null; then
            echo "✅ Service is ready!"
            READY=1
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "❌ Service crashed. Last 5 lines of log:"
            tail -n 5 vllm_server.log
            break
        fi
        sleep 5
    done

    # 6. Execute Inference Client
    if [ $READY -eq 1 ]; then
        echo "▶️  Starting test script..."
        # Updated to match the new argparse format in python script
        python client_test.py \
            --model_path "$MODEL_PATH" \
            --video_dir "$VIDEO_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --api_base "http://localhost:$PORT/v1"
    fi

    # 7. Auto Cleanup
    echo "🛑 Test finished, automatically shutting down service (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 3
    
    echo "🏁 Task completed. Press Enter to continue to the next model (if any)..."
    read -p "" 
done