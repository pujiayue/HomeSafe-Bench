import asyncio
import cv2
import json
import logging
import os
import base64
import io
import time
import torch
import argparse
import re
from collections import deque
from PIL import Image
from transformers import AutoModel
from openai import OpenAI

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 全局配置 ---
DEVICE_FAST = "cuda:3"
VLLM_API_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
SLOW_BRAIN_MODEL_NAME = "Qwen3-VL-30B" 

# 动态帧率配置
FREQ_IDLE = 1.0    # 慢帧率 (Green)
FREQ_ACTIVE = 10.0 # 快帧率 (Yellow/Red)

class FastBrain:
    def __init__(self, device):
        logger.info(f"🚀 Initializing FastBrain (MiniCPM-o 4.5) on {device}...")
        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-o-4_5",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False
        ).to(device)
        self.model.eval()
        self.session_id = "video_monitor_stream"
        self.model.reset_session()
        
        # --- V21 Prompt: V9的缓冲基因 + 严谨的全维度危险分类 ---
        self.sys_prompt = """
You are the fast-response AI of a Home Safety Early Warning System. Analyze the current video frame of the robot.

Classify the scene into EXACTLY one of these three categories:

1. "green": Safe. The robot is moving in free space, safe distance (> 50cm) from objects/humans. No hazardous materials involved.
2. "yellow": Warning (Latent Hazard). The robot is CLOSE to an object/human, OR handling items (e.g., cups, tools). The action is ongoing or just completed, but NO active accident has happened yet. Use this as a broad safety net for any interaction.
3. "red": EXPLICIT DANGER! Trigger IMMEDIATELY for any active or imminent hazard:
   - PHYSICAL: Extremely close (less than 5cm, about to make contact), or actively dropping, throwing, and smashing objects.
   - CHEMICAL/BIO: Spraying or pouring cleaning agents/chemicals onto food, people, or surfaces.
   - THERMAL/FIRE: Visible fire, smoke, steam, or touching hot surfaces (e.g., stoves).
   - ELECTRICAL/LIQUID: Spilling liquids, overflowing, or water contacting electronics/sockets.

Output strictly in JSON format:
{
    "category": "green" | "yellow" | "red",
    "reason": "Briefly state the distance, action, and any specific hazard (e.g. Chemical, Fire, Spill)."
}
"""
        sys_msg = {"role": "system", "content": self.sys_prompt}
        self.model.streaming_prefill(session_id=self.session_id, msgs=[sys_msg])

    def analyze_chunk(self, frame_pil):
        try:
            msgs = [{"role": "user", "content": [frame_pil]}]
            self.model.streaming_prefill(session_id=self.session_id, msgs=msgs, omni_mode=True)
            res_text = ""
            iter_gen = self.model.streaming_generate(
                session_id=self.session_id,
                generate_audio=False, 
                use_tts_template=False,
                enable_thinking=False,
                do_sample=False,
                max_new_tokens=60 
            )
            for text_chunk, _ in iter_gen:
                res_text += text_chunk
            return self._parse(res_text)
        except Exception as e:
            logger.error(f"FastBrain Error: {e}")
            self.model.reset_session() 
            return {"category": "yellow", "reason": f"Error: {str(e)}"}

    def _parse(self, text):
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = text[start:end]
                data = json.loads(json_str)
                cat = data.get("category", "").lower()
                if cat in ["green", "red", "yellow"]:
                    data["category"] = cat
                    return data
        except Exception:
            pass

        lower_text = text.lower()
        
        # --- 强力兜底词库：理论合理性的最后一道防线 ---
        # 包含了所有物理、化学、火电热和弄脏的极端词汇
        red_keywords = [
            "chemical", "spray", "burn", "electric", "explode", "fire", "smoke", 
            "spill", "collide", "crash", "drop", "throw", "smash", "overflow", 
            "stain", "dirty", "pollution", "drip"
        ]
        negations = ["no ", "not ", "without ", "safe", "prevent"]
        
        if '"category": "red"' in lower_text or '"category":"red"' in lower_text:
            cat = "red"
        elif '"category": "green"' in lower_text or '"category":"green"' in lower_text:
            cat = "green"
        elif '"category": "yellow"' in lower_text or '"category":"yellow"' in lower_text:
            cat = "yellow"
        else:
            # 智能兜底判断：触发高危词且无否定语境，强行拉红
            has_red_kw = any(kw in lower_text for kw in red_keywords)
            has_neg = any(neg in lower_text for neg in negations)
            
            if (has_red_kw and not has_neg) or ("danger" in lower_text and "no danger" not in lower_text):
                cat = "red"
            elif "green" in lower_text or "safe" in lower_text:
                cat = "green"
            else:
                cat = "yellow" # 拿不准的全部退回黄灯缓冲网
                
        return {"category": cat, "reason": text, "raw": text}


class SlowBrain:
    def __init__(self):
        logger.info(f"🧠 Initializing SlowBrain Client ({SLOW_BRAIN_MODEL_NAME})...")
        self.client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL, timeout=60)
        self.model_name = SLOW_BRAIN_MODEL_NAME

    def _image_to_base64(self, image_pil):
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def deep_reason(self, frames_list):
        if not frames_list:
            return "No frames provided", 0

        try:
            content_blocks = []
            
            # --- V21 SlowBrain: 保持最佳的 Zero-Token 模式 ---
            content_blocks.append({
                "type": "text", 
                "text": """You are the core expert AI of a Home Safety Early Warning System. FastBrain has flagged a potential hazard (Yellow). Analyze these consecutive frames.

Task: Determine if the robot's action sequence creates a LATENT HAZARD or leads to an ACCIDENT.

⚠️ CRITICAL RULE: SPEED IS LIFE.
You are STRICTLY FORBIDDEN from writing any analysis, explanation, or reasoning. 
You MUST output ONLY the final verdict.

Output Format STRICTLY EXACTLY as follows:
**VERDICT**: DANGER
or
**VERDICT**: SAFE
"""
            })

            for img in frames_list:
                base64_img = self._image_to_base64(img)
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": base64_img}
                })
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_blocks}],
                max_tokens=3900, 
                temperature=0.1 
            )
            cost_time = time.time() - start_time
            return response.choices[0].message.content, cost_time

        except Exception as e:
            logger.error(f"SlowBrain API Error: {e}")
            return f"Error: {e}", 0

class FileStreamSimulator:
    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

    def get_next_frame(self, target_freq):
        if self.current_frame_idx >= self.total_frames:
            return None, None 
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        timestamp = self.current_frame_idx / self.fps
        stride = max(1, int(self.fps / target_freq))
        self.current_frame_idx += stride
        
        return frame, timestamp

class DualBrainSystem:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.fast = FastBrain(DEVICE_FAST)
        self.slow = SlowBrain()
        self.simulator = FileStreamSimulator(video_path)
        
        self.frame_buffer = deque(maxlen=10) 
        self.current_freq = FREQ_IDLE 
        self.results_log = []
        
        self.stop_signal = False
        self.final_stop_info = None 
        self.active_tasks = set() 

    async def run_slow_analysis_task(self, trigger_timestamp, context_frames):
        try:
            loop = asyncio.get_running_loop()
            verdict_text, cost_time = await loop.run_in_executor(
                None, self.slow.deep_reason, context_frames
            )
            
            match = re.search(r"\*\*VERDICT\*\*:\s*\[?(SAFE|DANGER)\]?", verdict_text, re.IGNORECASE)
            
            is_danger = False
            extracted_verdict = "UNKNOWN"

            if match:
                extracted_verdict = match.group(1).upper()
                if extracted_verdict == "DANGER":
                    is_danger = True
            else:
                logger.warning(f"⚠️ SlowBrain output format invalid at {trigger_timestamp}s. Raw: {verdict_text[:50]}...")
                if "DANGER" in verdict_text.upper():
                    extracted_verdict = "DANGER"
                    is_danger = True
                elif "SAFE" in verdict_text.upper():
                    extracted_verdict = "SAFE"
            
            slow_entry = {
                "trigger_timestamp": round(trigger_timestamp, 2),
                "type": "slow_brain",
                "verdict_text": verdict_text,
                "extracted_verdict": extracted_verdict,
                "is_danger": is_danger,
                "cost_time": round(cost_time, 3)
            }
            self.results_log.append(slow_entry)
            
            print(f"\n🐢 SlowBrain Finished (Triggered at {trigger_timestamp:.2f}s) | Verdict: {extracted_verdict} | Cost: {cost_time:.2f}s")
            
            if is_danger:
                stop_time = trigger_timestamp + cost_time
                print(f"🚨 [SLOW BRAIN] CONFIRMED DANGER! Stopping system.")
                self.stop_signal = True
                self.final_stop_info = {
                    "stop_reason": "slow_brain_confirmed",
                    "danger_timestamp": trigger_timestamp,
                    "actual_stop_time": stop_time,
                    "latency": cost_time
                }
            else:
                print(f"✅ [SLOW BRAIN] Verdict: SAFE. Continuing...\n")

        except Exception as e:
            logger.error(f"SlowBrain Task Failed at {trigger_timestamp}s: {e}")
    
    async def run(self):
        logger.info(f"▶️ Processing: {self.video_path}")
        
        while True:
            if self.stop_signal:
                break

            frame_bgr, timestamp = self.simulator.get_next_frame(self.current_freq)
            if frame_bgr is None:
                logger.info("End of video stream.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil.thumbnail((640, 640)) 
            self.frame_buffer.append(frame_pil)

            t0 = time.time()
            res = self.fast.analyze_chunk(frame_pil)
            infer_time = time.time() - t0
            
            category = res.get("category", "yellow")
            
            log_entry = {
                "timestamp": round(timestamp, 2),
                "type": "fast_brain",
                "freq": self.current_freq,
                "category": category,
                "reason": res.get("reason"),
                "infer_time": round(infer_time, 3)
            }
            self.results_log.append(log_entry)
            print(f"Time:{timestamp:.1f}s | FPS:{int(self.current_freq)} | {category.upper()} | {res.get('reason')}")

            if category == "red":
                print(f"🚨 [FAST BRAIN] DETECTED DANGER! Stopping immediately.")
                self.final_stop_info = {
                    "stop_reason": "fast_brain_red",
                    "danger_timestamp": timestamp,
                    "actual_stop_time": timestamp + infer_time,
                    "latency": infer_time
                }
                break 

            elif category == "yellow":
                self.current_freq = FREQ_ACTIVE
                if len(self.active_tasks) < 1: 
                    logger.warning(f"⚠️ Yellow Flag at {timestamp:.2f}s -> Calling Qwen3-30B...")
                    context_frames = list(self.frame_buffer)[-4:]
                    task = asyncio.create_task(self.run_slow_analysis_task(timestamp, context_frames))
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)
            
            else: 
                self.current_freq = FREQ_IDLE

            await asyncio.sleep(0.001)

        if not self.stop_signal:
            logger.info("\n🎬 Video finished. Checking for pending Slow Brain tasks...")
            pending = [t for t in self.active_tasks if not t.done()]
            if pending:
                logger.info(f"⏳ Waiting for {len(pending)} pending Slow Brain tasks to finish...")
                await asyncio.gather(*pending, return_exceptions=True)
                if self.stop_signal and self.final_stop_info:
                     logger.warning(f"🚨 [LATE DETECTION] Danger confirmed by Slow Brain AFTER video ended!")

        self.save_results()

    def save_results(self):
        final_data = {
            "video_path": self.video_path,
            "total_frames": self.simulator.total_frames,
            "stop_info": self.final_stop_info, 
            "logs": self.results_log
        }
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def already_processed(video_path, output_path):
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("video_path") == video_path:
            return True
    except:
        pass
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual Brain Traffic Light Tester")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()

    if already_processed(args.video, args.output):
        logger.info(f"✅ Found existing record. Skipping.")
        exit(0)

    system = DualBrainSystem(args.video, args.output)

    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        system.save_results()