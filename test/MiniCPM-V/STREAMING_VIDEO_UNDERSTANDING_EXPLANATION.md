# MiniCPM-o 2.6 流式视频理解实现详解

## 一、核心架构概览

MiniCPM-o 2.6 通过 **OTDM（全模态时分复用）机制** 实现流式视频理解，核心思想是：
1. **时间对齐**：将视频帧和音频按共享时间轴同步
2. **单元化组织**：每 1 秒组成一个多模态单元 `<unit>`
3. **增量处理**：通过 `streaming_prefill` 和 `streaming_generate` 实现边输入边理解

---

## 二、流式视觉理解实现

### 2.1 视频帧采样策略

根据官方文档和代码实现，视频流被表示为 **采样率为 1 fps 的帧序列**：

```python
# 从 model_server.py 中的实现逻辑可以看出：
# 每秒采样 1 帧，使用 moviepy 提取
frame = video.get_frame(t)  # t 为秒级时间戳
image = Image.fromarray(frame.astype(np.uint8))
```

**关键点**：
- 采样率固定为 **1 fps**（每秒 1 帧）
- 每帧对应视频中的 1 秒时间窗口

### 2.2 混合分辨率编码策略

代码中体现了混合分辨率编码的思想（虽然具体实现可能在模型内部）：

```python
# 从 omnilmm.py 的 get_vision_embedding 可以看到：
# - 使用 EVA02 作为视觉编码器
# - Resampler 将视觉特征压缩到固定 token 数
vision_embedding = vision_tower.forward_features(pixel_values)
res = self.resampler(vision_embedding)  # 压缩到 num_query 个 tokens
```

**实现原理**：
- **最后一帧**（用户查询结束前）：高分辨率编码（如 1344 x 896）
- **其他帧**：中等分辨率编码（如 448 x 448）
- 通过 `Resampler` 模块将所有帧统一压缩到固定数量的视觉 tokens

**为什么这样设计**：
- 平衡效率和性能
- 最后一帧通常包含用户最关心的信息，需要更高分辨率
- 中间帧主要用于时序理解，中等分辨率足够

---

## 三、OTDM（全模态时分复用）机制详解

### 3.1 时间对齐与单元构造

从 `model_server.py` 的 `prefill` 方法可以看到 OTDM 的核心实现：

```python
# model_server.py:416-471
def prefill(self, audio, image, is_end):
    # 1. 收集 1 秒的音频数据（16000 采样率 = 1 秒）
    if len(audio_np) > 16000:
        audio_np = audio_np[:16000]  # 截取 1 秒
    
    # 2. 构造多模态单元
    if self.image_prefill is not None:
        cnts = ["<unit>", self.image_prefill, audio_np]  # 关键：<unit> 标记
    else:
        cnts = [audio_np]
    
    # 3. 通过 streaming_prefill 添加到 session
    msg = {"role":"user", "content": cnts}
    msgs = [msg]
    res = self.minicpmo_model.streaming_prefill(
        session_id=str(self.session_id),
        msgs=msgs, 
        tokenizer=self.minicpmo_tokenizer,
        max_slice_nums=slice_nums,
    )
```

**OTDM 单元结构**：
```
<unit> = ["<unit>", image_frame, audio_1s]
```

其中：
- `"<unit>"`：特殊 token，标记多模态单元的起始
- `image_frame`：该秒对应的视频帧（PIL Image）
- `audio_1s`：该秒对应的音频数据（numpy array，16000 采样率，单声道）

### 3.2 时间轴同步机制

**关键代码位置**：`model_server.py:416-471`

```python
# 视频处理流程（从官方文档示例推断）：
# 1. 视频按秒切分
for i in range(num_units):  # num_units = ceil(video.duration)
    frame = video.get_frame(i+1)  # 第 i+1 秒的帧
    image = Image.fromarray(frame.astype(np.uint8))
    
    # 2. 提取对应时间段的音频
    audio = audio_np[sr*i:sr*(i+1)]  # sr=16000，即 1 秒音频
    
    # 3. 组合成单元
    content = ["<unit>", image, audio]
```

**同步原理**：
- 视频帧和音频按**相同的 1 秒时间窗口**对齐
- 每个 `<unit>` 代表同一时刻的视觉和听觉信息
- LLM 按时间顺序处理这些单元，形成连续的时间记忆

### 3.3 流式处理流程

完整的流式处理流程如下：

```python
# 1. 初始化 session
model.reset_session()
session_id = 'unique_session_id'

# 2. Prefill 系统提示（可选）
sys_msg = model.get_sys_prompt(mode='omni', language='en')
model.streaming_prefill(
    session_id=session_id,
    msgs=[sys_msg],
    tokenizer=tokenizer
)

# 3. 逐秒 Prefill 视频/音频单元
for t in range(video_duration):
    frame = video.get_frame(t+1)
    audio = audio_np[sr*t:sr*(t+1)]
    unit = ["<unit>", frame, audio]
    
    model.streaming_prefill(
        session_id=session_id,
        msgs=[{"role":"user", "content": unit}],
        tokenizer=tokenizer
    )

# 4. 流式生成响应
for r in model.streaming_generate(
    session_id=session_id,
    tokenizer=tokenizer,
    generate_audio=True
):
    text = r['text']
    audio_wav = r['audio_wav']  # 如果启用 TTS
    # 处理流式输出...
```

---

## 四、关键技术细节

### 4.1 Session 管理与 KV Cache

从代码可以看出，流式处理依赖 **session-based KV cache**：

```python
# model_server.py:300-302
if self.minicpmo_model.llm_past_key_values[0][0].shape[2]>8192:
    self.session_id += 1  # 清除 KV cache
```

**机制**：
- 每个 `session_id` 维护独立的 KV cache
- `streaming_prefill` 将多模态单元编码后存入 KV cache
- `streaming_generate` 基于累积的 KV cache 生成响应
- 当 cache 过大时，通过切换 `session_id` 重置

### 4.2 视觉编码流程

从 `omnilmm.py` 可以看到视觉编码的完整流程：

```python
# omnilmm.py:108-121
def get_vision_embedding(self, pixel_values):
    # 1. EVA02 视觉编码器提取特征
    vision_embedding = vision_tower.forward_features(
        pixel_values.type(dtype))
    
    # 2. 移除 prefix tokens（如果有）
    if hasattr(vision_tower, 'num_prefix_tokens') and vision_tower.num_prefix_tokens > 0:
        vision_embedding = vision_embedding[:, vision_tower.num_prefix_tokens:]
    
    # 3. Resampler 压缩到固定 token 数
    res = self.resampler(vision_embedding)
    return res
```

**Resampler 的作用**（`resampler.py:96-172`）：
- 使用 **Perceiver-style 的 cross-attention** 机制
- 将任意数量的视觉 tokens 压缩到固定数量（如 64 或 256）
- 通过可学习的 query 和 2D 位置编码实现

### 4.3 音频处理

音频处理的关键点：

```python
# model_server.py:445-449
audio_np, sr = librosa.load(input_audio_path, sr=16000, mono=True)
if len(audio_np) > 16000:
    audio_np = audio_np[:16000]  # 确保正好 1 秒
```

**音频规格**：
- 采样率：**16000 Hz**（单声道）
- 每个 unit 的音频长度：**1 秒**（16000 个采样点）
- 格式：numpy array (float32)

---

## 五、与现有模型的区别

### 5.1 传统多模态模型 vs MiniCPM-o 2.6

| 特性 | 传统模型 | MiniCPM-o 2.6 |
|------|---------|---------------|
| **输入方式** | 一次性输入所有帧 | 流式输入，逐秒处理 |
| **时间记忆** | 无或有限 | 通过 KV cache 保持完整时间记忆 |
| **实时性** | 需要等待完整视频 | 可以边输入边理解 |
| **查询时机** | 仅在查询时获取帧 | 独立于查询持续感知流 |

### 5.2 关键创新点

1. **独立感知能力**：
   - 模型可以**独立于用户查询持续感知**多模态输入流
   - 这与现有模型（仅在查询时获取帧）有本质区别

2. **时间记忆与推理**：
   - 通过累积的 KV cache，模型可以回答需要时间记忆的问题
   - 例如："球在哪个杯子里？"、"我刚才擦掉了哪些词？"

3. **原生多轮实时对话**：
   - 支持在流式输入过程中进行多轮交互
   - 每次交互都基于当前已累积的时间上下文

---

## 六、实际应用示例

### 6.1 危险行为监测场景

基于你的测试需求，流式视频理解的典型应用流程：

```python
# 伪代码示例
model.reset_session()
session_id = video_filename

# 系统提示
sys_msg = model.get_sys_prompt(mode='omni', language='en')
model.streaming_prefill(session_id, [sys_msg], tokenizer)

# 逐秒处理视频
for t in range(0, video_duration):
    # 1. 提取 1 秒单元
    frame = video.get_frame(t+1)
    audio = audio_np[16000*t:16000*(t+1)]
    unit = ["<unit>", frame, audio]
    
    # 2. Prefill 单元
    model.streaming_prefill(
        session_id,
        [{"role":"user", "content": unit}],
        tokenizer
    )
    
    # 3. 每 N 秒询问一次（实时预警）
    if t % ask_every == 0:
        question = f"分析 0.0s 到 {t+1:.1f}s，判断是否有危险行为？"
        model.streaming_prefill(
            session_id,
            [{"role":"user", "content": [question]}],
            tokenizer
        )
        
        # 4. 生成响应
        response = ""
        for r in model.streaming_generate(session_id, tokenizer):
            response += r['text']
        
        # 5. 解析并判断
        if "Safe" not in response:
            danger_timestamp = extract_timestamp(response)
            break  # 发现危险，停止处理
```

---

## 七、总结

MiniCPM-o 2.6 的流式视频理解通过以下机制实现：

1. **1 fps 采样 + 混合分辨率编码**：平衡效率和性能
2. **OTDM 机制**：将多模态流按时间轴同步，组织成 `<unit>` 单元
3. **Session-based KV Cache**：维护时间记忆，支持增量处理
4. **Streaming API**：`streaming_prefill` 和 `streaming_generate` 实现边输入边理解

这种设计使得模型能够：
- ✅ 实时处理视频流
- ✅ 保持完整的时间记忆
- ✅ 支持需要时序推理的任务
- ✅ 原生支持多轮实时对话

---

## 参考资料

- 核心代码文件：
  - `omnilmm/model/omnilmm.py`：模型架构
  - `omnilmm/model/resampler.py`：视觉特征压缩
  - `web_demos/minicpm-o_2.6/model_server.py`：流式处理实现
  - `test_minicpm_o_danger_detection.py`：应用示例

- 官方文档：
  - README.md 中的 "Multimodal Live Streaming" 章节
  - "Chat with Video" 章节（3D-Resampler 说明）

