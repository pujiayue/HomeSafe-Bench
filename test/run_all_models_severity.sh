#!/bin/bash

export VLLM_TARGET_DEVICE=cuda
export CUDA_VISIBLE_DEVICES="0,1"  
export NCCL_IB_DISABLE=1

PORT=8001

VIDEO_DIR="HomeSafe-Bench/DATASETS_TimeStamp"
OUTPUT_DIR="HomeSafe-Bench/test/results/severity"

MODELS=(
   "/path/to/your/model/InternVL2-8B"
   "/path/to/your/model/LLaVA-NeXT-Video-7B-hf"
   # ...
)


PYTHON_SCRIPT="eval_severity_client.py"



echo "🧹 Cleaning remaining vLLM processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server"
sleep 2

for MODEL_PATH in "${MODELS[@]}"; do
    echo "========================================================="
    echo "🚀 Preparing to start model: $MODEL_PATH"

    PATH_LOWER=$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
    
    CURRENT_MAX_LEN=32768  
    GPU_MEM=0.8
    TP=1
    
    CMD_ARGS=(
        "--trust-remote-code"
        "--port" "$PORT"
        "--served-model-name" "$MODEL_PATH"
        "--dtype" "bfloat16"
    )

    unset VLLM_QWEN_MAX_PIXELS
    unset VLLM_ALLOW_LONG_MAX_MODEL_LEN
    unset VLLM_INTERNVL_MAX_DYNAMIC_PATCH


    # === A. LLaVA-OneVision ===
    if [[ "$PATH_LOWER" == *"llava-onevision"* ]]; then
        echo "⚙️  Mode: LLaVA-OneVision"
        GPU_MEM=0.9   
        TP=2          
        CURRENT_MAX_LEN=92160 # Must be very large to accommodate multiple frames
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        
        CMD_ARGS+=("--enforce-eager")
        # Key: OneVision treats video frames as images, so a large number of image inputs must be allowed
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":64}")
        
        export VLLM_NO_USAGE_STATS=1
        export VLLM_DISABLE_PROFILING=1

    # === B. LLaVA-NeXT-Video / Generic LLaVA-HF ===
    elif [[ "$PATH_LOWER" == *"llava-next-video"* ]] || [[ "$PATH_LOWER" == *"llava-hf"* ]]; then
        echo "⚙️  Mode: LLaVA-NeXT-Video / HF"
        CURRENT_MAX_LEN=8192 
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        TP=1
        GPU_MEM=0.8
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
        
    # === C. VideoLLaMA ===
    elif [[ "$PATH_LOWER" == *"videollama"* ]]; then
        echo "⚙️  Mode: VideoLLaMA"
        TP=2                  
        GPU_MEM=0.9           
        CURRENT_MAX_LEN=8192  
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")

    # === D. InternVL ===
    elif [[ "$PATH_LOWER" == *"internvl"* ]]; then
        echo "⚙️  Mode: InternVL"
        export VLLM_INTERNVL_MAX_DYNAMIC_PATCH=1
        CMD_ARGS+=("--mm-processor-kwargs" '{"max_dynamic_patch":1}')
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
        GPU_MEM=0.7
        # If it is 8B or larger, TP=2 is usually required
        if [[ "$PATH_LOWER" == *"8b"* ]] || [[ "$PATH_LOWER" == *"26b"* ]]; then
            TP=2
        fi

    # === E. Granite Vision ===
    elif [[ "$PATH_LOWER" == *"granite-vision"* ]]; then
        echo "⚙️  Mode: Granite Vision"
        CURRENT_MAX_LEN=8192
        GPU_MEM=0.6
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")

    # === F. MiniCPM ===
    elif [[ "$PATH_LOWER" == *"minicpm"* ]]; then
        echo "⚙️  Mode: MiniCPM"
        TP=2
        GPU_MEM=0.8
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")

    # === G. Generic fallback (determine TP based on size) ===
    elif [[ "$PATH_LOWER" == *"8b"* ]] || [[ "$PATH_LOWER" == *"7b"* ]]; then
        echo "⚙️  Mode: Generic 7B/8B"
        TP=1 # Single GPU is usually enough for 7B/8B; increase if insufficient
        GPU_MEM=0.8
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
    
    else
        echo "⚠️  Unknown model, using default safe configuration"
        CURRENT_MAX_LEN=8192
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
    fi

    
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        "${CMD_ARGS[@]}" \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $CURRENT_MAX_LEN \
        > vllm_server.log 2>&1 &

    VLLM_PID=$!

    echo "⏳ Waiting for API service to be ready (PID: $VLLM_PID)..."
    READY=0
    # Check in a loop 30 times, every 10 seconds (total 5 minutes timeout)
    for i in {1..30}; do
        # Check whether the port responds with HTTP 200
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models)
        
        if [ "$HTTP_CODE" == "200" ]; then
            echo "✅ Service is ready!"
            READY=1
            break
        fi
        
        # Check whether the process is still alive
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "❌ Service has crashed, last 20 lines of log:"
            tail -n 20 vllm_server.log
            break
        fi
        
        echo "   ... waiting ($i/30)"
        sleep 10
    done

    if [ $READY -eq 1 ]; then
        
        python3 "$PYTHON_SCRIPT" \
            --model_path "$MODEL_PATH" \
            --api_base "http://localhost:$PORT/v1" \
            --video_dir "$VIDEO_DIR" \
            --output_dir "$OUTPUT_DIR"

        echo "✅ Evaluation script finished"
    else
        echo "❌ Skipping evaluation (service not started)"
    fi

    echo "🛑 Test finished, automatically shutting down service (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null
    
    # Force ensure the port is released
    sleep 5
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    sleep 5
    
    echo "---------------------------------------------------------"
done

echo "🎉 All model tests completed!"