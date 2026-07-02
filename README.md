<p align="center">
<h1 align="center"><img src="log.png" align="center" width="6.5%"><strong>Building-MLLM: From Geometric Labels to Language Understanding for Indoor Building Components</strong></h1>

<p align="center">
  Shuju Jing<sup>1</sup>, Chao Yin<sup>2†</sup>
  <br>
  <sup>1</sup>Shandong University &emsp;
  <sup>2</sup>The Hong Kong University of Science and Technology
  <br>
</p>

</p>


## 📌 Overview
<p align="center">
  <img src="Graphical Abstract.jpg" alt="Overview Image" width="1000"/>
</p>

<p align="justify">
Point cloud-based understanding has become an important enabler for facility operation and maintenance workflows involving indoor building components. However, existing methods output only discrete labels without explaining component functions or natural language interactions. We propose Building-MLLM, a point cloud-centered multimodal large language model (MLLM) for indoor components, which explicitly models point clouds and instructions to generate responses across Simple Recognition, Complex Captioning, and Multi-Engineering Question Answering tasks. Building-MLLM addresses semantic concentration through four domain-specific mechanisms: Point Information Enhancer for task-relevant semantics, Geometry-Preserving Regularization preventing geometric erosion, fixed textual prefix for domain stabilization, and multi-dimensional LoRA balancing recognition with reasoning. We further develop a multi-constraint progressive instruction-generation engine that compiles a synthetic point cloud–text dataset comprising 4,198 objects, 37,782 instruction-following pairs, and 47 indoor component categories. Experimental results show that Building-MLLM achieves 88.00%, 65.10%, and 68.14% on the three task types, respectively, demonstrating its superiority for indoor component language understanding.
</p>

## ⚙️ Environment Setup

1. Clone this repository to your server:

```bash
git clone git@github.com:JingShuju/Building-MLLM.git
```
2. Navigate to the project directory:
```
cd Building-MLLM
```
3. Set up the environment:
```
conda create -n buildingmllm python=3.10 -y
conda activate buildingmllm
pip install --upgrade pip
pip install -e .
pip install ninja
pip install flash-attn
```

4. Environment used in our experiments:
```
Driver 535.230.02 
CUDA 12.2
Python 3.10.13
PyTorch 2.7.1 
Transformers 4.40.2 
```

5. For better compatibility, we provide a more robust environment configuration tested across multiple setups. Researchers are encouraged to refer to:

```
Building-MLLM/environment.yml
```


## 📎 Pre-trained Baseline Model Preparation
Our work is developed based on the pre-trained PointLLM_7B model.

1. Download the pre-trained baseline model:
```
https://huggingface.co/RunsenXu/PointLLM_7B_v1.2/tree/main
```
2. Place it under the following directory:
```
Building-MLLM/RunsenXu/PointLLM_7B_v1.2
```

## 📂 Data Preparation

1. Download our point cloud–text instruction-following dataset for indoor building components:
```
# Google Drive
https://drive.google.com/file/d/1aNTcDwRTN2jaTLmNT6ATEiBJeHzWpSsY/view
```
or
```
# Baidu Drive
https://pan.baidu.com/s/1nLsEuqer4stGPd1sqLvV9w?pwd=vtt4
```
2. Place point cloud data in:
```
Building-MLLM/data
```
3. Place text instruction annotations in:
```
Building-MLLM/data/anno_data
```




## 🧠 Building-MLLM Method
Two-stage Training

Stage 1: Modality alignment training using Simple Recognition point cloud–text instruction data:
```
CUDA_VISIBLE_DEVICES=0,1 bash Building-MLLM/scripts/JING_stage1.sh
```
Stage 2: Instruction tuning using Simple Recognition, Complex Captioning, and Multi-Engineering QA point cloud–text instruction data:
```
CUDA_VISIBLE_DEVICES=0,1 bash Building-MLLM/scripts/JING_stage2_lora.sh
```


## 🚀 Inference


1. Simple Recognition
```
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/eval_1.py \
--model_name outputs/JING_train_stage2/JING_stage2_lora \
--task_type classification \
--prompt_index 0 \
--data_path \
--anno_path
```
2. Complex Captioning
```
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/eval_2.py \
--model_name outputs/JING_train_stage2/JING_stage2_lora \
--task_type captioning \
--prompt_index 2 \
--data_path \
--anno_path
```
3. Multi-Engineering QA

```
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/eval_3.py \
--model_name outputs/JING_train_stage2/JING_stage2_lora \
--conversation_types 2_Knowledge_Capability 3_Commonsense_Reasoning 4_Advanced_Engineering_Reasoning 5_Functional_Comparison 6_Constraint_Reasoning 7_Spatial_Relationship 8_Embodied_Interaction \
--data_path \
--anno_path
```



## 📊 Evaluation


1. GPT-4-based Simple Recognition Evaluation
```
export OPENAI_API_KEY

CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator.py \
--results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation \
--model_type gpt-4-0613 \
--eval_type open-free-form-classification \
--parallel --num_workers 4
```

2. GPT-4-based Complex Captioning Evaluation
```
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator.py \
--results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation \
--model_type gpt-4-0613 \
--eval_type object-captioning \
--parallel --num_workers 4
```
3. Traditional Metric-based Captioning Evaluation
```
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/traditional_evaluator.py \
--results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation
```
4. GPT-4-based Multi-Engineering QA Evaluation
```
# Knowledge Capability
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "knowledge capability" 

# Commonsense Reasoning
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "commonsense reasoning" 

# Advanced Engineering Reasoning
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "advanced engineering reasoning" 

# Functional Comparison
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "functional comparison"

# Constraint Reasoning
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "constraint reasoning"

# Spatial Relationship
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "spatial relationship"

# Embodied Interaction
CUDA_VISIBLE_DEVICES=0 python buildingmllm/eval/evaluator_7task.py --results_path outputs/JING_train_stage2/JING_stage2_lora/evaluation --model_type gpt-4-0613 --parallel --num_workers 4 --task_type "embodied interaction"
```

## ⚡ Acknowledgements
We would like to thank the authors of *PointLLM: Empowering Large Language Models to Understand Point Clouds* for their inspiring work and open-source code:
https://github.com/InternRobotics/PointLLM
