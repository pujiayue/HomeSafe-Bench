## 代码目录
- `HomeSafe-Bench/detector_dual_brain` 为代码根目录
- `HomeSafe-Bench/detector_dual_brain/run_slow_brain.sh` 在 138 服务器的 6、7 卡上双卡启动慢脑模型（通过vllm框架）
- `HomeSafe-Bench/detector_dual_brain/dual_brain.py`  为双脑架构检测器主代码（包括minicpm o 4.5快脑模型和通过vllm架构 api 调用qwen3-30B慢脑模型）
- `HomeSafe-Bench/detector_dual_brain/run_slow_brain.sh` 为使用`dual_brain(_xxx).py` 批量测试视频case脚本
- `HomeSafe-Bench/detector_dual_brain/evaluate.py` 是评估脚本
- `HomeSafe-Bench/detector_dual_brain/anno_truth_final.json` 是数据集 ground truth
- `HomeSafe-Bench/detector_dual_brain/dual_brain_latency_eval.csv` 是最终评估结果
- 数据目录在 `HomeSafe-Bench/DATASETS_TimeStamp`

## 启动说明
- tmux 在后台开一个会话窗口，`conda activate qwen3-vl`， 打开 vllm qwen3环境 运行`run_slow_brain_new.sh` 开启 vllm 慢脑 api
- 在另一个窗口，`conda activate minicpm_o_4.6`，运行`run_slow_brain_new.sh` 即可
