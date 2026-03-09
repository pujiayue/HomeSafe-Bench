#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dangerous Behavior Detection Script Based on MiniCPM-o 2.6 Streaming Inference

Usage:
    python test_streaming_danger_detection.py --video_dir /path/to/videos --output_json results.json
"""

import os
import glob
import json
import re
import math
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
    Split the video into OTDM units by second (official example implementation)
    
    Args:
        video_path: Video file path
        flatten: False returns nested list, True returns flattened list
        max_seconds: Maximum processing duration (seconds), None means process the entire video
    
    Returns:
        flatten=False: [["<unit>", img1, audio1], ["<unit>", img2, audio2], ...]
        flatten=True: ["<unit>", img1, audio1, "<unit>", img2, audio2, ...]
    """
    video = VideoFileClip(video_path)
    duration = video.duration
    print(f'  Video duration: {duration:.2f}s')
    
    # Limit maximum duration
    if max_seconds is not None:
        duration = min(duration, max_seconds)
        print(f'  Processing first {duration:.2f}s')
    
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
    
    num_units = math.ceil(duration)  # Round up to obtain seconds
    
    contents = []
    for i in range(num_units):
        t = i + 1  # Start from the 1st second (moviepy index starts from 1)
        if t > duration:
            break
            
        # Get the frame of this second
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
                # The last second may be less than 1 second, pad with zeros
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
    
    Args:
        response_text: Model response text
        current_time_range: (start_time, end_time) current observed time range
    
    Returns:
        (is_dangerous: bool, timestamp: float or None)
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
                print(f"  ⚠️  Warning: Timestamp {danger_time} outside range [{start_time:.1f}, {end_time:.1f}]")
                # If close to the range, still accept it
                if abs(danger_time - start_time) < 2.0 or abs(danger_time - end_time) < 2.0:
                    return True, danger_time
        except ValueError:
            pass
    
    # If danger detected but timestamp cannot be extracted, return midpoint of the current window
    if not is_safe:
        return True, (start_time + end_time) / 2.0
    
    return False, None