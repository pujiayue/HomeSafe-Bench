import json

def filter_json(input_path, output_path):
    try:
        # 读取原始 JSON 文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_data = {}
        
        for key, value in data.items():
            # 提取目标字段，如果字段不存在则设置默认值
            new_data[key] = {
                "is_dangerous": value.get("is_dangerous", False),
                "timestamp": value.get("timestamp", 0.0),
                "response": value.get("response", "")
            }
        
        # 将结果保存到新的 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
            
        print(f"成功！提取后的数据已保存至: {output_path}")

    except Exception as e:
        print(f"处理过程中出错: {e}")

# 执行转换
input_file = '/new_disk4/jiayue_pu/MiniCPM-o/results_minicpm-o-4.5-streaming_delay_time.json'
output_file = '/new_disk4/jiayue_pu/MiniCPM-o/results_minicpm-o-4.5-streaming.json'

filter_json(input_file, output_file)