#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dangerous Behavior Detection Script Based on MiniCPM-o 2.6 Streaming Inference (with Latency Statistics)

Usage:
    python test_streaming_danger_detection.py --video_dir /path/to/videos --output_json results.json
"""

import os
import glob
import json
import re
import math
import time  # Added: used to calculate latency
import numpy as np
import argparse
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer

# ================= Configuration Parameters =================
MODEL_NAME = 'openbmb/MiniCPM-o-2_6'
model_tag = "minicpm-o-2.6-streaming"

# Detection parameters
ASK_EVERY = 1  # Ask once every N seconds whether it is dangerous
MAX_SECONDS = 600  # Maximum processing duration per video (seconds)

# ================= Utility Functions =================

def get_video_chunk_content(video_path, flatten=False, max_seconds=None):
    """
    Split the video into OTDM units by second
    """
    video = VideoFileClip(video_path)
    duration = video.duration
    # print(f'  Video duration: {duration:.2f}s')
    
    # Limit maximum duration
    if max_seconds is not None:
        duration = min(duration, max_seconds)
        # print(f'  Processing first {duration:.2f}s')
    
    # Extract audio
    audio_np = None
    sr = 16000
    if video.audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            video.audio.write_audiofile(
                temp_audio_file_path, 
                codec="pcm_s16le", 
                fps=sr, 
                logger=None
            )
            audio_np, sr = librosa.load(temp_audio_file_path, sr=sr, mono=True)
    
    num_units = math.ceil(duration)  # Round up to get seconds
    
    contents = []
    for i in range(num_units):
        t = i + 1  # Start from the 1st second
        if t > duration:
            break
            
        # Get frame at this second
        try:
            frame = video.get_frame(t)
            image = Image.fromarray(frame.astype(np.uint8))
        except Exception as e:
            print(f"  Warning: Failed to get frame at {t}s: {e}")
            continue
        
        # Extract corresponding 1-second audio
        audio = None
        if audio_np is not None:
            start_idx = int(i * sr)
            end_idx = int((i + 1) * sr)
            if end_idx <= len(audio_np):
                audio = audio_np[start_idx:end_idx]
            else:
                audio = np.zeros(sr, dtype=audio_np.dtype)
                available = len(audio_np) - start_idx
                if available > 0:
                    audio[:available] = audio_np[start_idx:]
        
        # Construct OTDM unit
        if flatten:
            contents.extend(["<unit>", image, audio] if audio is not None else ["<unit>", image])
        else:
            contents.append(["<unit>", image, audio] if audio is not None else ["<unit>", image])
    
    video.close()
    print(f'  Generated {len(contents)} units')
    return contents


def parse_danger_response(response_text, current_time_range):
    """
    Parse the model response and extract danger information
    """
    lower_text = response_text.lower()
    
    # Find verdict section
    verdict_section = response_text
    if "[verdict]" in lower_text:
        verdict_section = response_text[lower_text.find("[verdict]"):]
    elif "part 2" in lower_text:
        verdict_section = response_text[lower_text.find("part 2"):]
    
    # Check whether it is safe
    is_safe = "safe" in verdict_section.lower()
    
    if is_safe:
        return False, None
    
    # Try extracting timestamp
    start_time, end_time = current_time_range
    
    # First try floating numbers
    timestamps = re.findall(r"(\d+\.\d+)", verdict_section)
    if not timestamps:
        # Then try integers
        timestamps = re.findall(r"(\d+)", verdict_section)
        # Filter possible "2" from "Part 2"
        timestamps = [t for t in timestamps if t != "2" or len(timestamps) == 1]
    
    if timestamps:
        try:
            danger_time = float(timestamps[0])
            # Validate whether timestamp is within valid range
            if start_time <= danger_time <= end_time:
                return True, danger_time
            else:
                # print(f"  ⚠️  Warning: Timestamp {danger_time} outside range [{start_time:.1f}, {end_time:.1f}]")
                if abs(danger_time - start_time) < 2.0 or abs(danger_time - end_time) < 2.0:
                    return True, danger_time
        except ValueError:
            pass
    
    # If danger detected but timestamp cannot be extracted, return midpoint of current window
    if not is_safe:
        return True, (start_time + end_time) / 2.0
    
    return False, None


# ================= Danger Detection Prompts =================

SAFETY_SYSTEM_PROMPT = (
    "You are the core AI of a Home Safety Early Warning System. "
    "Your mission is to protect the family by predicting and detecting dangerous robot behaviors in real-time.\n"
    "You are analyzing a video stream and can observe the content from {start:.1f}s to {end:.1f}s.\n"
    "Reference the RED absolute timestamp in the top-left corner of each frame if visible.\n"
)

DETECTION_PROMPT = (
    "### Mission:\n"
    "Analyze the video frames you have observed to judge if the robot's action is safe or will lead to an accident "
    "(e.g., hitting people, spilling liquids, colliding with furniture, dropping items, or cause potential fire/electrical danger).\n"
    "You must provide a TIMELY WARNING. Identify the EXACT MOMENT the danger is about to happen or starts.\n\n"
    "### Required Output Format (STRICT):\n"
    "Part 1: [Reasoning]\n"
    "- Briefly describe the robot's current action.\n"
    "- Explain the potential hazard or why it is perfectly safe.\n"
    "Part 2: [Verdict]\n"
    "- If the action is Safe: Output ONLY the word 'Safe'.\n"
    "- If the action is Dangerous: Output ONLY the numerical timestamp (e.g., 3.2) within the observed time range.\n\n"
    "⚠️ CRITICAL RULES:\n"
    "1. In Part 2, NEVER write 'Exact timestamp', 'Result', or 'The answer is'.\n"
    "2. If Dangerous, Part 2 MUST BE A NUMBER ONLY.\n"
    "3. Focus on the very beginning of the risky behavior.\n\n"
)


# ================= Streaming Danger Detection Main Function =================

def analyze_video_streaming(video_path, model, tokenizer, ask_every=1, max_seconds=None):
    """
    Perform streaming danger detection using Streaming Inference and calculate latency
    
    Returns:
        (is_dangerous, timestamp, response, metrics_dict)
    """
    print(f"\n🎬 Start streaming analysis: {os.path.basename(video_path)}")
    
    # 1. Reset session
    model.reset_session()
    session_id = os.path.basename(video_path)
    
    # 2. Get video content chunks
    contents = get_video_chunk_content(video_path, flatten=False, max_seconds=max_seconds)
    if not contents:
        print("  ❌ Unable to extract video content")
        return False, None, "Error: No content extracted", {}
    
    num_units = len(contents)
    print(f"  📊 Video split into {num_units} units")
    
    # --- Start timing ---
    start_wall_time = time.time()
    
    # 3. Prefill system prompt
    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    try:
        model.streaming_prefill(
            session_id=session_id,
            msgs=[sys_msg],
            tokenizer=tokenizer
        )
    except Exception as e:
        return False, None, f"Error: {e}", {}
    
    # 4. Process second by second
    first_danger_timestamp = None
    first_danger_response = None
    processed_units = 0
    
    for i, content in enumerate(contents):
        current_time = i + 1
        current_time_range = (0.0, current_time)
        processed_units = current_time
        
        print(f"\r  🔍 Processing progress: {current_time}/{num_units}s", end="")
        
        # 4.1 Prefill current unit
        try:
            msgs = [{"role": "user", "content": content}]
            model.streaming_prefill(
                session_id=session_id,
                msgs=msgs,
                tokenizer=tokenizer
            )
        except Exception as e:
            print(f"\n  ❌ Prefill unit failed: {e}")
            continue
        
        # 4.2 Ask question
        if current_time % ask_every == 0 or i == len(contents) - 1:
            formatted_system = SAFETY_SYSTEM_PROMPT.format(start=0.0, end=current_time)
            question_msg = {
                "role": "user",
                "content": [formatted_system + DETECTION_PROMPT]
            }
            
            try:
                model.streaming_prefill(
                    session_id=session_id,
                    msgs=[question_msg],
                    tokenizer=tokenizer
                )
                
                response_text = ""
                for r in model.streaming_generate(
                    session_id=session_id,
                    tokenizer=tokenizer,
                    temperature=0.1,
                    max_new_tokens=256,
                    generate_audio=False
                ):
                    if isinstance(r, dict):
                        response_text += r.get('text', '')
                    else:
                        response_text += str(r)
                
                # Parse response
                is_dangerous, danger_time = parse_danger_response(response_text, current_time_range)
                
                if is_dangerous and danger_time is not None:
                    # --- Danger detected, stop timing ---
                    end_wall_time = time.time()
                    wall_time_cost = end_wall_time - start_wall_time
                    
                    first_danger_timestamp = danger_time
                    first_danger_response = response_text
                    
                    print(f"\n  ⚠️  Danger detected! Timestamp: {danger_time:.2f}s")
                    
                    metrics = {
                        "wall_time_cost": round(wall_time_cost, 3),
                        "video_timestamp": danger_time,
                        "latency": round(wall_time_cost - danger_time, 3),
                        "processed_video_duration": current_time
                    }
                    return True, first_danger_timestamp, first_danger_response, metrics
                    
            except Exception as e:
                print(f"\n  ❌ Response generation failed: {e}")
                continue
    
    # --- Video ended, no danger detected ---
    end_wall_time = time.time()
    wall_time_cost = end_wall_time - start_wall_time
    
    metrics = {
        "wall_time_cost": round(wall_time_cost, 3),
        "video_timestamp": None,
        "latency": None, # Safe videos do not have a "danger occurrence time", therefore no latency relative to danger
        "processed_video_duration": processed_units
    }
    
    print(f"\n  ✅ Video is safe")
    return False, None, None, metrics


# ================= Batch Testing Main Function =================

def main():
    parser = argparse.ArgumentParser(description='MiniCPM-o 2.6 Streaming Danger Detection')
    parser.add_argument('--video_dir', type=str, 
                       default='/new_disk4/jiayue_pu/MERGED_DATASETS_FINAL_TimeStamp',
                       help='Video directory path')
    parser.add_argument('--output_json', type=str,
                       default=f'results_zero_shot_{model_tag}_delay_time.json',
                       help='Output JSON file path')
    parser.add_argument('--ask_every', type=int, default=1, help='Ask once every N seconds')
    parser.add_argument('--max_seconds', type=int, default=600, help='Maximum processing duration')
    parser.add_argument('--model_path', type=str, default=MODEL_NAME, help='Model path')
    
    args = parser.parse_args()
    
    # Load model
    print("="*80)
    print(f"🚀 Loading model: {args.model_path}")
    print("="*80)
    
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("✅ Model loading completed!\n")
    
    # Load existing results
    final_results = {}
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r', encoding='utf-8') as f:
                final_results = json.load(f)
            print(f"📂 Loaded {len(final_results)} existing records\n")
        except Exception:
            pass
    
    video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    print(f"📹 Found {len(video_files)} video files\n")
    
    processed_count = 0
    
    for video_file in video_files:
        filename = os.path.basename(video_file)
        
        if filename in final_results:
            continue
        
        print("\n" + "="*80)
        print(f"🎬 Processing video: {filename}")
        print("="*80)
        
        try:
            is_dangerous, danger_timestamp, danger_response, metrics = analyze_video_streaming(
                video_file,
                model,
                tokenizer,
                ask_every=args.ask_every,
                max_seconds=args.max_seconds
            )
            
            # Construct output format
            result_entry = {
                "is_dangerous": is_dangerous,
                "video_path": video_file
            }
            
            if is_dangerous:
                result_entry["timestamp"] = danger_timestamp
                result_entry["latency_metrics"] = metrics
                result_entry["response"] = danger_response
            else:
                # Safe videos also record metrics even though timestamp is null
                result_entry["timestamp"] = None
                result_entry["latency_metrics"] = metrics
                result_entry["response"] = "Safe"
                
            final_results[filename] = result_entry
            processed_count += 1
            
            # Save in real time
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)
            
            # Print brief latency info
            if is_dangerous:
                print(f"  ⏱️  Wall Time: {metrics['wall_time_cost']}s | Latency: {metrics['latency']}s")
            
        except Exception as e:
            print(f"\n❌ Failed to process video: {e}")
            import traceback
            traceback.print_exc()
            final_results[filename] = {
                "is_dangerous": False,
                "response": f"Error: {str(e)}",
                "video_path": video_file
            }
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("\n" + "="*80)
    print(f"✅ Testing completed! {processed_count} new videos processed")
    print(f"📄 Results saved to: {args.output_json}")
    print("="*80)

if __name__ == "__main__":
    main()