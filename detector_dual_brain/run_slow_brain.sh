#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7


# --- 1. 自动清理残留进程 (防止 "Free memory is less than..." 错误) ---
echo "正在强制清理残留进程..."
ray stop --force 2>/dev/null
pkill -u $(whoami) -9 -f "vllm.entrypoints.openai.api_server"
pkill -u $(whoami) -9 -f "multiprocessing.resource_tracker"
# 等待几秒让操作系统回收显存
sleep 5

# --- 2. 显存检查 ---
echo "当前显存状态 (如果这里显示显存未释放，脚本将无法运行):"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | grep -E "1,|6,"

# --- 3. 设置环境变量 ---
# 使用正确的 PyTorch 内存分配配置
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# --- 4. 启动 Ray ---
echo "正在启动 Ray..."
ray start --head \
    --node-ip-address=127.0.0.1 \
    --port=6379 \
    --num-gpus=2 \
    --disable-usage-stats \
    --include-dashboard=false

if [ $? -ne 0 ]; then
    echo "Ray 启动失败！可能是端口被占用或之前的 Ray 没有关干净。"
    exit 1
fi
sleep 3

# --- 5. 启动 vLLM ---
echo "正在启动 vLLM..."
# 注意：
# 1. gpu-memory-utilization 设为 0.9 (给 PyTorch 留 10% 的缓冲，防止碎片化 OOM)
# 2. max-model-len 设为 4096 (如果显存依然紧张，可以尝试降低这个值)
python3 -m vllm.entrypoints.openai.api_server \
    --model "/new_disk4/jiayue_pu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/d0ed0380729be07a546fdefafbb4fe411f341e92" \
    --served-model-name "Qwen3-VL-30B" \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --trust-remote-code \
    --distributed-executor-backend mp \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.67 \
    --limit-mm-per-prompt '{"image": 5, "video": 1}' \
    --port 8000 \
    --load-format safetensors

# --- 6. 退出清理 ---
echo "服务停止，清理 Ray..."
ray stop