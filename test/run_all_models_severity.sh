#!/bin/bash

# ================= 1. 全局配置区域 =================
export VLLM_TARGET_DEVICE=cuda
export CUDA_VISIBLE_DEVICES="0,1"  # 指定使用的 GPU ID
export NCCL_IB_DISABLE=1           # 禁用 IB，防止某些环境下的通信错误

PORT=8001

# 数据集与输出路径 (请根据实际情况修改)
VIDEO_DIR="HomeSafe-Bench/DATASETS_TimeStamp"
OUTPUT_DIR="HomeSafe-Bench/test/results/severity"

# 待测试模型列表
MODELS=(
   "/path/to/your/model/InternVL2-8B"
   "/path/to/your/model/LLaVA-NeXT-Video-7B-hf"
)

# Python 脚本名称 (需与上一步保存的文件名一致)
PYTHON_SCRIPT="eval_severity_client.py"

# ================= 2. 主循环逻辑 =================

# 清理残留进程
echo "🧹 清理残留的 vLLM 进程..."
pkill -9 -f "vllm.entrypoints.openai.api_server"
sleep 2

for MODEL_PATH in "${MODELS[@]}"; do
    echo "========================================================="
    echo "🚀 准备启动模型: $MODEL_PATH"

    # 转为小写以便匹配逻辑
    PATH_LOWER=$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
    
    # --- vLLM 默认参数 ---
    CURRENT_MAX_LEN=32768  
    GPU_MEM=0.8
    TP=1
    
    # 初始化基础参数数组
    CMD_ARGS=(
        "--trust-remote-code"
        "--port" "$PORT"
        "--served-model-name" "$MODEL_PATH"  # 显式指定服务名称，防止客户端找不到
        "--dtype" "bfloat16"
    )

    unset VLLM_QWEN_MAX_PIXELS
    unset VLLM_ALLOW_LONG_MAX_MODEL_LEN
    unset VLLM_INTERNVL_MAX_DYNAMIC_PATCH

    # ================= 3. 模型特定配置 (保持原有逻辑) =================

    # === A. LLaVA-OneVision ===
    if [[ "$PATH_LOWER" == *"llava-onevision"* ]]; then
        echo "⚙️  模式: LLaVA-OneVision"
        GPU_MEM=0.9   
        TP=2          
        CURRENT_MAX_LEN=92160 # 必须很大以容纳多帧
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        
        CMD_ARGS+=("--enforce-eager")
        # 关键：OneVision 将视频帧视为图片，必须允许大量图片输入
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":64}")
        
        export VLLM_NO_USAGE_STATS=1
        export VLLM_DISABLE_PROFILING=1

    # === B. LLaVA-NeXT-Video / Generic LLaVA-HF ===
    elif [[ "$PATH_LOWER" == *"llava-next-video"* ]] || [[ "$PATH_LOWER" == *"llava-hf"* ]]; then
        echo "⚙️  模式: LLaVA-NeXT-Video / HF"
        CURRENT_MAX_LEN=8192 
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        TP=1
        GPU_MEM=0.8
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
        
    # === C. VideoLLaMA ===
    elif [[ "$PATH_LOWER" == *"videollama"* ]]; then
        echo "⚙️  模式: VideoLLaMA"
        TP=2                  
        GPU_MEM=0.9           
        CURRENT_MAX_LEN=8192  
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")

    # === D. InternVL ===
    elif [[ "$PATH_LOWER" == *"internvl"* ]]; then
        echo "⚙️  模式: InternVL"
        export VLLM_INTERNVL_MAX_DYNAMIC_PATCH=1
        CMD_ARGS+=("--mm-processor-kwargs" '{"max_dynamic_patch":1}')
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
        GPU_MEM=0.7
        # 如果是 8B 或更大，通常需要 TP=2
        if [[ "$PATH_LOWER" == *"8b"* ]] || [[ "$PATH_LOWER" == *"26b"* ]]; then
            TP=2
        fi

    # === E. Granite Vision ===
    elif [[ "$PATH_LOWER" == *"granite-vision"* ]]; then
        echo "⚙️  模式: Granite Vision"
        CURRENT_MAX_LEN=8192
        GPU_MEM=0.6
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")

    # === F. MiniCPM ===
    elif [[ "$PATH_LOWER" == *"minicpm"* ]]; then
        echo "⚙️  模式: MiniCPM"
        TP=2
        GPU_MEM=0.8
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")

    # === G. 通用兜底 (根据大小判断 TP) ===
    elif [[ "$PATH_LOWER" == *"8b"* ]] || [[ "$PATH_LOWER" == *"7b"* ]]; then
        echo "⚙️  模式: Generic 7B/8B"
        TP=1 # 7B/8B 单卡通常够用，不够再改
        GPU_MEM=0.8
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
    
    else
        echo "⚠️  未知模型，使用默认安全配置"
        CURRENT_MAX_LEN=8192
        CMD_ARGS+=("--limit-mm-per-prompt" "{\"image\":20}")
    fi

    # ================= 4. 启动 vLLM 服务 =================
    
    echo "▶️  执行命令: vllm serve ... (Mem: $GPU_MEM, TP: $TP, MaxLen: $CURRENT_MAX_LEN)"
    
    # 启动 vLLM (后台运行)
    # 注意：这里使用了 "${CMD_ARGS[@]}" 来正确展开数组参数
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        "${CMD_ARGS[@]}" \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $CURRENT_MAX_LEN \
        > vllm_server.log 2>&1 &

    VLLM_PID=$!

    # ================= 5. 健康检查 =================
    echo "⏳ 等待 API 服务就绪 (PID: $VLLM_PID)..."
    READY=0
    # 循环检查 30 次，每次 10 秒 (共 5 分钟超时)
    for i in {1..30}; do
        # 检查端口是否响应 HTTP 200
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models)
        
        if [ "$HTTP_CODE" == "200" ]; then
            echo "✅ 服务已就绪！"
            READY=1
            break
        fi
        
        # 检查进程是否存活
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "❌ 服务已崩溃，最后 20 行日志如下："
            tail -n 20 vllm_server.log
            break
        fi
        
        echo "   ... 等待中 ($i/30)"
        sleep 10
    done

    # ================= 6. 执行 Python 评测脚本 =================
    if [ $READY -eq 1 ]; then
        echo "▶️  开始运行评测脚本..."
        
        # [修改点] 使用新的参数传递方式
        python3 "$PYTHON_SCRIPT" \
            --model_path "$MODEL_PATH" \
            --api_base "http://localhost:$PORT/v1" \
            --video_dir "$VIDEO_DIR" \
            --output_dir "$OUTPUT_DIR"

        echo "✅ 评测脚本执行完毕"
    else
        echo "❌ 跳过评测 (服务未启动)"
    fi

    # ================= 7. 清理与重置 =================
    echo "🛑 正在关闭服务 (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null
    
    # 强制确保端口释放
    sleep 5
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    sleep 5
    
    echo "---------------------------------------------------------"
done

echo "🎉 所有模型测试完成！"