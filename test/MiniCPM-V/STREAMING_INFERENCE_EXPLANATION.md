# MiniCPM-o 2.6 Streaming Inference 详解与对齐分析

## 一、官方示例代码解析

### 1.1 代码流程分析

官方 streaming inference 示例的核心流程：

```python
# 1. 重置 session（清除 KV cache）
model.reset_session()

# 2. 获取视频内容块（按秒切分）
contents = get_video_chunk_content(video_path, flatten=False)

# 3. Prefill 系统提示
res = model.streaming_prefill(session_id=session_id, msgs=[sys_msg], tokenizer=tokenizer)

# 4. Prefill 视频/音频块（逐秒）
for content in contents:
    msgs = [{"role":"user", "content": content}]
    res = model.streaming_prefill(session_id=session_id, msgs=msgs, tokenizer=tokenizer)

# 5. 生成响应
res = model.streaming_generate(session_id=session_id, tokenizer=tokenizer, ...)
```

### 1.2 与 OTDM 机制的对齐

**关键对齐点**：

1. **`get_video_chunk_content()` 函数的作用**：
   - 将视频按 **1 秒**切分成多个单元
   - 每个单元格式：`["<unit>", image_frame, audio_1s]`
   - 这就是 **OTDM 机制**的具体实现

2. **`flatten=False` 的含义**：
   - `flatten=False`：返回 `List[List]`，每个元素是一个 unit
   - `flatten=True`：返回 `List`，所有 units 展平成一个列表
   - 对于 streaming，使用 `flatten=False` 以便逐秒 prefill

3. **Streaming Prefill 的作用**：
   - 每次 `streaming_prefill` 将一个 `<unit>` 存入 KV cache
   - 累积时间记忆，形成连续的视频理解上下文

4. **Streaming Generate 的作用**：
   - 基于所有已 prefill 的 units 生成响应
   - 模型可以访问整个视频的时间上下文

### 1.3 与之前解释的对应关系

| 官方示例代码 | 之前的解释 | 对应关系 |
|------------|----------|---------|
| `get_video_chunk_content()` | OTDM 单元构造 | ✅ 完全对应 |
| `streaming_prefill()` | 将单元存入 KV cache | ✅ 完全对应 |
| `streaming_generate()` | 基于累积上下文生成 | ✅ 完全对应 |
| `session_id` | Session 管理 | ✅ 完全对应 |

---

## 二、`get_video_chunk_content()` 函数实现

根据官方文档，这个函数的实现应该是：

```python
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf

def get_video_chunk_content(video_path, flatten=True):
    """
    将视频按秒切分成 OTDM 单元
    
    Args:
        video_path: 视频文件路径
        flatten: 
            - True: 返回展平的列表，所有 units 在一个列表中
            - False: 返回嵌套列表，每个元素是一个 unit
    
    Returns:
        flatten=True: ["<unit>", img1, audio1, "<unit>", img2, audio2, ...]
        flatten=False: [["<unit>", img1, audio1], ["<unit>", img2, audio2], ...]
    """
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    # 提取音频
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    
    num_units = math.ceil(video.duration)  # 向上取整，得到秒数
    
    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        # 获取第 i+1 秒的帧（从 1 开始索引）
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        
        # 提取对应 1 秒的音频
        audio = audio_np[sr*i:sr*(i+1)]  # sr=16000，即 1 秒
        
        if flatten:
            # 展平模式：所有 units 在一个列表中
            contents.extend(["<unit>", image, audio])
        else:
            # 嵌套模式：每个 unit 是一个子列表
            contents.append(["<unit>", image, audio])
    
    return contents
```

**关键点**：
- **1 fps 采样**：每秒取 1 帧（`video.get_frame(i+1)`）
- **1 秒音频**：`audio_np[sr*i:sr*(i+1)]`，正好 16000 个采样点
- **OTDM 单元**：`["<unit>", image, audio]` 格式

---

## 三、能否达到流式视频理解功能？

### ✅ 完全可以！

官方示例代码**完全支持**流式视频理解，原因：

1. **OTDM 机制已实现**：
   - `get_video_chunk_content()` 构造了标准的 OTDM 单元
   - 每个单元包含 1 秒的视频帧和音频

2. **时间记忆保持**：
   - 通过 `streaming_prefill` 累积所有 units 到 KV cache
   - 模型可以访问整个视频的时间上下文

3. **实时交互能力**：
   - 可以在任意时刻调用 `streaming_generate` 询问问题
   - 模型基于已累积的时间记忆回答

### ⚠️ 注意事项

1. **性能说明**：
   - 官方提到 "audio encoding is not global"，会有轻微性能下降
   - 这是正常的，因为流式处理需要增量编码

2. **使用场景**：
   - 适合：需要时间记忆的任务（如"球在哪个杯子里？"）
   - 适合：实时视频理解
   - 不适合：只需要单帧理解的简单任务

---

## 四、代码对齐关系总结

### 4.1 官方示例 vs 之前的解释

| 概念 | 官方示例代码 | 之前的解释 | 对齐状态 |
|------|------------|----------|---------|
| **OTDM 单元构造** | `get_video_chunk_content()` 返回 `["<unit>", img, audio]` | 每 1 秒构造一个 `<unit>` | ✅ 完全对齐 |
| **1 fps 采样** | `video.get_frame(i+1)` 每秒 1 帧 | 采样率为 1 fps 的帧序列 | ✅ 完全对齐 |
| **1 秒音频** | `audio_np[sr*i:sr*(i+1)]` (16000 采样点) | 每 1 秒的音频块 | ✅ 完全对齐 |
| **Session 管理** | `session_id = '123'` | 通过 session_id 管理 KV cache | ✅ 完全对齐 |
| **KV Cache 累积** | `streaming_prefill()` 逐单元存入 | 将单元存入 KV cache，累积时间记忆 | ✅ 完全对齐 |
| **流式生成** | `streaming_generate()` 基于累积上下文 | 基于已累积的 KV cache 生成 | ✅ 完全对齐 |
| **时间记忆** | 所有 units 都在同一个 session | 通过 KV cache 保持完整时间记忆 | ✅ 完全对齐 |

### 4.2 关键差异说明

**官方示例的 `flatten` 参数**：

- **`flatten=False`**（用于 streaming）：
  ```python
  contents = [
      ["<unit>", img1, audio1],  # 第 1 秒
      ["<unit>", img2, audio2],  # 第 2 秒
      ...
  ]
  ```
  - 每个元素是一个独立的 unit
  - 可以逐秒 `streaming_prefill`
  - **适合流式处理**

- **`flatten=True`**（用于一次性处理）：
  ```python
  contents = ["<unit>", img1, audio1, "<unit>", img2, audio2, ...]
  ```
  - 所有 units 展平在一个列表中
  - 一次性 `chat()` 处理
  - **适合非流式处理**

### 4.3 性能说明对齐

官方文档提到：
> "The streaming inference has a slight performance degradation because the audio encoding is not global."

**原因**：
- 流式处理中，音频是**增量编码**的（每 1 秒编码一次）
- 非流式处理中，音频是**全局编码**的（一次性编码整个音频）
- 增量编码会有轻微的性能损失，但换来了**实时性**和**时间记忆能力**

---

## 五、如何编写危险检测测试脚本

基于官方示例，我已经创建了完整的危险检测脚本：`test_streaming_danger_detection.py`

### 5.1 脚本特点

1. **完全基于官方 streaming inference API**
2. **支持实时危险检测**：每 N 秒询问一次
3. **断点续传**：支持从 JSON 文件恢复
4. **结果格式兼容**：与你的现有测试脚本格式一致

### 5.2 使用方法

```bash
# 基本使用
python test_streaming_danger_detection.py \
    --video_dir /path/to/videos \
    --output_json results.json

# 自定义参数
python test_streaming_danger_detection.py \
    --video_dir /path/to/videos \
    --output_json results.json \
    --ask_every 2 \          # 每 2 秒询问一次
    --max_seconds 300 \      # 每个视频最多处理 300 秒
    --model_path openbmb/MiniCPM-o-2_6
```

### 5.3 核心流程

脚本的核心流程完全遵循官方示例：

```python
# 1. 重置 session
model.reset_session()

# 2. 获取视频内容块（按秒切分）
contents = get_video_chunk_content(video_path, flatten=False)

# 3. Prefill 系统提示
model.streaming_prefill(session_id, [sys_msg], tokenizer)

# 4. 逐秒 Prefill + 定期询问
for i, content in enumerate(contents):
    # Prefill 当前单元
    model.streaming_prefill(session_id, [{"role":"user", "content": content}], tokenizer)
    
    # 每 N 秒询问一次
    if (i+1) % ask_every == 0:
        # Prefill 问题
        model.streaming_prefill(session_id, [question_msg], tokenizer)
        
        # 生成响应
        for r in model.streaming_generate(session_id, tokenizer, ...):
            response += r['text']
        
        # 解析并判断
        if is_dangerous:
            break  # 发现危险，停止
```

### 5.4 与你的现有脚本对比

| 特性 | 现有脚本 (`test_minicpm_o_danger_detection.py`) | 新脚本 (`test_streaming_danger_detection.py`) |
|------|-----------------------------------------------|--------------------------------------------|
| **处理方式** | 分块处理（10秒块，重叠0.5秒） | 流式处理（每秒 1 单元） |
| **时间记忆** | 每个块独立处理 | 累积时间记忆 |
| **API** | `model.chat()` | `model.streaming_prefill()` + `streaming_generate()` |
| **实时性** | 需要等待块处理完成 | 可以边输入边理解 |
| **OTDM 机制** | ❌ 未使用 | ✅ 完全使用 |

---

## 六、总结

### ✅ 官方示例完全支持流式视频理解

1. **OTDM 机制已实现**：`get_video_chunk_content()` 构造标准 OTDM 单元
2. **时间记忆保持**：通过 `streaming_prefill` 累积到 KV cache
3. **实时交互能力**：可以在任意时刻询问问题

### 📝 使用建议

1. **对于危险检测任务**：
   - 使用新脚本 `test_streaming_danger_detection.py`
   - 设置 `ask_every=1` 实现每秒检测
   - 发现危险后立即停止，节省计算资源

2. **对于需要完整视频理解的任务**：
   - 先 prefill 所有 units
   - 最后一次性 generate
   - 模型可以访问整个视频的时间上下文

3. **性能优化**：
   - 如果视频很长，可以设置 `max_seconds` 限制
   - 如果不需要实时性，可以增大 `ask_every`
   - 如果不需要音频，可以只传图像（但会失去 OTDM 的优势）

### 🎯 关键理解

**流式视频理解的核心**：
- ✅ 使用 `flatten=False` 的 `get_video_chunk_content()`
- ✅ 逐秒 `streaming_prefill` 累积时间记忆
- ✅ 在需要时 `streaming_generate` 获取响应
- ✅ 所有操作共享同一个 `session_id`，保持上下文连续性

  