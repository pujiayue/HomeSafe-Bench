import cv2
import os

def extract_keyframes(video_path, keyframes, output_dir="extracted_frames"):
    """
    从视频中提取指定的关键帧并保存为PNG图片。
    
    Args:
        video_path (str): 视频文件的路径。
        keyframes (dict): 包含关键帧名称和对应时间戳（秒）的字典。
        output_dir (str): 保存图片的输出目录。
    """
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 -> {video_path}")
        return

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("错误: 无法打开视频文件。")
        return

    # 获取视频的帧率 (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频加载成功: {video_path}")
    print(f"FPS: {fps}, 总帧数: {total_frames}")

    # 遍历关键帧字典进行提取
    for name, timestamp in keyframes.items():
        # 计算目标帧的索引 (时间戳 * FPS)
        target_frame_index = int(timestamp * fps)
        
        # 检查帧索引是否超出范围
        if target_frame_index >= total_frames:
            print(f"警告: 关键帧 '{name}' (时间: {timestamp}s, 帧: {target_frame_index}) 超出视频总长度，跳过。")
            continue

        # 设置视频读取位置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        
        # 读取帧
        ret, frame = cap.read()
        
        if ret:
            # 构建输出文件名
            output_filename = f"{name}_{timestamp}s.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存为PNG
            cv2.imwrite(output_path, frame)
            print(f"已保存: {output_path}")
        else:
            print(f"错误: 无法读取关键帧 '{name}' (时间: {timestamp}s)。")

    # 释放资源
    cap.release()
    print("处理完成。")

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 视频文件路径 (根据你的描述，这是源文件路径)
    # 请确保这个路径相对于脚本是正确的，或者使用绝对路径
    video_file_path = "/Users/pujiayue/EmbodiedSafety/MERGED_DATASETS_FINAL_TimeStamp/0__zzl_7.mp4"
    
    # 你提供的数据
    data = {
        "keyframes": {
          "intent_onset": 2.44,
          "pnr": 4.08,
          "intervention_deadline": 2.86,
          "impact_outcome": 6.65,
          "action_end": 6.65
        }
    }

    # --- 执行提取 ---
    extract_keyframes(video_file_path, data["keyframes"])