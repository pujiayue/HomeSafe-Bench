import time
import cv2
import base64
import socketio
import threading
import json
import os

# 配置区
SERVER_URL = 'http://127.0.0.1:8081'
TEST_VIDEO = '/new_disk1/jiayue_pu/MERGED_DATASETS_FINAL_TimeStamp/Robot_Cleans_Stovetop_Catches_Fire.mp4'
STREAM_FPS = 2.0  # 每秒向服务端推2帧画面，模拟摄像头

sio = socketio.Client()
is_video_playing = True
start_time_physical = 0.0

@sio.event
def connect():
    print("✅ 已成功连接到 VITA 流式服务端！")

@sio.event
def disconnect():
    print("❌ 与服务端断开连接")

# 🚨 注意：这里的事件名称 ('model_response') 可能需要根据 VITA server.py 里的实际名字微调
@sio.on('model_response') 
def on_model_response(data):
    """接收模型推理结果的回调函数"""
    global start_time_physical
    current_time = time.time()
    
    # 提取服务端返回的文本
    reply = data.get('text', '') if isinstance(data, dict) else data
    
    # 简单过滤空包
    if not reply.strip():
        return

    print(f"[收到响应] {reply}")
    
    if "Danger" in reply or "Verdict:" in reply and "Safe" not in reply:
        latency = current_time - start_time_physical
        print(f"🔥 触发警报！端到端真实延迟: {latency:.3f} 秒")
        # 视情况可以在这里停止推流或触发机器人急停逻辑

def stream_video(video_path):
    global start_time_physical, is_video_playing
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # 计算每次推送的物理时间间隔
    sleep_interval = 1.0 / STREAM_FPS
    
    print(f"🎬 开始推流模拟，帧率设定为 {STREAM_FPS} fps...")
    start_time_physical = time.time()
    frame_count = 0

    while is_video_playing:
        # 跳帧读取，以匹配我们设定的推流 FPS
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * (fps / STREAM_FPS)))
        ret, frame = cap.read()
        
        if not ret:
            print("视频播放完毕。")
            break

        # 将图像压缩并转为 base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # 构造发给服务端的 Payload
        # 这里需要模拟网页端发给后端的格式，通常是图片+空文本或者系统指令
        payload = {
            "image": f"data:image/jpeg;base64,{frame_b64}",
            "text": "Analyze the current frame for safety. Is there any danger?", # 可以视服务端逻辑调整
            "is_first_frame": (frame_count == 0)
        }

        # 🚨 发送事件名称 ('client_message') 也需与 server.py 对齐
        sio.emit('client_message', payload)
        print(f"  ↗️ 已推送第 {frame_count+1} 帧")
        
        frame_count += 1
        
        # 强制休眠，模拟物理时间的真实流逝！这是“真延迟”的核心。
        time.sleep(sleep_interval)

    cap.release()
    sio.disconnect()

if __name__ == '__main__':
    try:
        # 连接服务端
        sio.connect(SERVER_URL)
        # 启动视频推流线程
        stream_thread = threading.Thread(target=stream_video, args=(TEST_VIDEO,))
        stream_thread.start()
        # 阻塞等待
        stream_thread.join()
    except Exception as e:
        print(f"连接或推流发生异常: {e}")
