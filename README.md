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



## ⚡ Acknowledgements
We would like to thank the authors of *PointLLM: Empowering Large Language Models to Understand Point Clouds* for their inspiring work and open-source code:
https://github.com/InternRobotics/PointLLM
