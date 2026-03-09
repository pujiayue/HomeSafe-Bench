# HomeSafe-Bench

This repository provides the implementation details for the paper:

HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios

# Abstract
The rapid evolution of embodied agents has accelerated the deployment of household robots in real-world environments. However, unlike structured industrial settings, household spaces introduce unpredictable safety risks, where system limitations such as perception latency and lack of common sense knowledge can lead to dangerous errors. Current safety evaluations, often restricted to static images, text, or general hazards, fail to adequately benchmark dynamic unsafe action detection in these specific contexts. To bridge this gap, we introduce **HomeSafe-Bench**, a challenging benchmark designed to evaluate Vision-Language Models (VLMs) on unsafe action detection in household scenarios. HomeSafe-Bench is contrusted via a hybrid pipeline combining physical simulation with advanced video generation and features 438 diverse cases across six functional areas with fine-grained multidimensional annotations. Beyond benchmarking, we propose **Hierarchical Dual-Brain Guard for Household Safety (HD-Guard)**, a hierarchical streaming architecture for real-time safety monitoring. HD-Guard coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning, effectively balancing inference efficiency with detection accuracy. Evaluations demonstrate that HD-Guard achieves a superior trade-off between latency and performance, while our analysis identifies critical bottlenecks in current VLM-based safety detection.

# Overview
- HomeSafe-Bench: A benchmark for evaluating unsafe action detection in household scenarios.
- HD-Guard: A dual-brain system for real-time safety monitoring:
    - FastBrain: Lightweight, high-frequency screening.
    - SlowBrain: Asynchronous, deep multimodal reasoning.

# Figures
- Overview of HomeSafe-Bench and HD-Guard:![](pics/abcd_00.png)

# Data
You can download the data from [Google Drive](https://drive.google.com/drive/folders/1mMTKtmGmu-dBdylRZUoPt3QRKmDe_gk4?usp=sharing).


# Requirements
1. **Evaluation on Homesafe-Bench:**

Set up the environment:
```bash
conda create -n vllm python=3.10 -y
pip install -r requirements_vllm.txt
```

2. **Running HD-Guard:**

The HD-Guard architecture requires two separate environments to run the SlowBrain (vLLM/Qwen) and FastBrain (MiniCPM) concurrently without dependency conflicts. We recommend using Python 3.10 for both environments.

**(a) SlowBrain Environment (vLLM / Qwen3)**
```bash
conda create -n qwen3-vl python=3.10 -y
conda activate qwen3-vl
pip install -r requirements_qwen3-vl.txt
```

**(b) FastBrain Environment (MiniCPM-o 4.5)**
```bash
conda create -n minicpm_o_4.5 python=3.10 -y
conda activate minicpm_o_4.5
pip install -r requirements_minicpm_o_4.5.txt
```


# Getting Started

1. Evaluation on Homesafe-Bench:
```bash
cd test
./run_all_models.sh               # Test open-source models
./run_closed_source.sh            # Test closed-source models
./run_all_models_severity.sh      # Test models for severity assessment
python evaluate.py                # Evaluate test results
```
For test and latency calculations of models not supported by the vLLM framework, please refer to the code.


2. **Running HD-Guard (Dual-Brain Architecture)**

To run the dual-brain detector, follow these steps:

**Step 1: Start the SlowBrain (vLLM API)**
```bash
tmux new -s slow_brain
conda activate qwen3-vl
# Navigate to the dual brain directory and start the vLLM backend
cd detector_dual_brain
bash run_slow_brain.sh
```

**Step 2: Start the FastBrain & Main Detector**
```bash
conda activate minicpm_o_4.6
cd detector_dual_brain
bash run_batch_test.sh
```