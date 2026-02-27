import sys
import os
import torch
from fastapi import FastAPI, Request
import uvicorn

# 动态导入 VITA 官方推理逻辑
VITA_ROOT = "/new_disk1/jiayue_pu/TEST_vllm/zero_shot/VITA"
sys.path.append(VITA_ROOT)

# 导入核心函数
# 注意：VITA 的 server.py 通常依赖 argparse 的全局变量，我们需要先初始化它
from web_demo.server import generate_response, load_model, args

app = FastAPI()

@app.post("/vllm_chat")
async def vllm_chat(request: Request):
    data = await request.json()
    text = data.get("text", "")
    images = data.get("images", [])
    # 直接调用官方核心推理函数
    res = generate_response(text, images, [])
    return {"response": res}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # 显式指定模型路径，防止它去读取错误的默认路径
    args.model_path = "/new_disk1/jiayue_pu/TEST_vllm/zero_shot/VITA/demo_VITA_ckpt"
    print(f"🚀 Loading model from: {args.model_path}")
    
    # 核心：必须手动调用加载函数
    load_model()
    
    print("✅ Model loaded successfully. Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)