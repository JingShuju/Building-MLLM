#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#在mem中加入pointllm/model/pointllm.py 加入pointllm/model/pointbert/point_encoder.py 
 #最开始在这里加载模型

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List
import yaml
import torch
import transformers
from buildingmllm.train.pointllm_trainer import PointLLMTrainer #在这里加载训练器
from transformers import TrainerCallback
from buildingmllm import conversation as conversation_lib   #对话
from buildingmllm.model import *
from buildingmllm.data import make_object_point_data_module   #加载点云
import torch.nn as nn
# * logger
from buildingmllm.utils import build_logger
from peft import PeftModel
IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]" #填充 保持长度一致 
DEFAULT_EOS_TOKEN = "</s>" #结束位置
DEFAULT_BOS_TOKEN = "</s>" #开始位置
DEFAULT_UNK_TOKEN = "<unk>" #用于识别罕见词
torch.autograd.set_detect_anomaly(True)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")  #这里是大语言模型
    version: Optional[str] = field(default="v1")
    tune_layer: int = field(default=0)
@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})  #只是又单独划分的吗？？？
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments): #训练参数Arguments
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."}) # * whether to load checkpoints at the mo
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(default=True, metadata={"help": "Whether to fix the PointNet."})

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    
    # * for two stage training
    tune_mm_mlp_adapter: bool = field(default=True) # * set True when pre-training, and false when fine-tuning  #我默认两阶段都是训练的 他说不是
    stage_2: bool = field(default=False) # * set True when fine-tuning
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None) # * path to the pre-trained projector & output_embed & input_embed  #这个是在第二阶段联合微调用的？
    detatch_point_token: bool = field(default=False) # * deprecated
    # * point backbone ckpt path
    point_backbone_ckpt: str = field(default=None)  #这是point_bert的backbone

    #lora的尝试
    use_lora: Optional[bool] = field(default=False, metadata={"help": "Whether to enable LoRA injection"})
    use_lora_only: Optional[bool] = field(default=False, metadata={})
    use_ia3_only: Optional[bool] = field(default=False, metadata={})
    use_loha_only: Optional[bool] = field(default=False, metadata={})
    use_adalora_only: Optional[bool] = field(default=False, metadata={})
    use_lokr_only: Optional[bool] = field(default=False, metadata={})  # * whether to use LoKR
    use_mt_prompt_only: Optional[bool] = field(default=False, metadata={})  # * whether to use multitask prompt tuning
    use_prompt_only: Optional[bool] = field(default=False, metadata={})  # * whether to use prompt tuning
    #lora的尝试

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


#lora的尝试
import peft
#'AdaLoraConfig', 'AdaptionPromptConfig', 'IA3Config', 'LoHaConfig', 'LoKrConfig', 'LoraConfig', 'MultitaskPromptTuningConfig', 'PeftConfig', 'PrefixTuningConfig', 'PromptEncoderConfig', 'PromptLearningConfig', 'PromptTuningConfig']
from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig, IA3Config, LoHaConfig, LoKrConfig, MultitaskPromptTuningConfig, PromptTuningConfig
"""import inspect
peft_methods = [
    name for name, obj in inspect.getmembers(peft) 
    if inspect.isclass(obj) and name.endswith("Config")
]
print(sorted(peft_methods))"""
lora_config = LoraConfig(
    r=16,  #8 #16 #32
    lora_alpha=64,#16 #32 #64
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],#"q_proj", "v_proj"  #"q_proj", "v_proj", "k_proj", "o_proj"
    task_type=TaskType.CAUSAL_LM
    #layers_to_transform=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #layers_to_transform=list(range(16, 32))
    #layers_pattern="model.model.layers."
)
#lora的尝试
adalora_config = AdaLoraConfig(
    init_r=32,                  # 初始秩
    target_r=8,                 # 最终秩（会在训练中逐渐降到这个值）
    beta1=0.85, beta2=0.85,     # EMA 参数，控制重要性计算
    tinit=200,                  # 开始降秩的 step
    tfinal=1600,                # 完成降秩的 step
    deltaT=10,                  # 每隔多少步重新计算重要性
    lora_alpha=16,              # LoRA 缩放系数
    lora_dropout=0.1,           # Dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 注入的模块  #"gate_proj", "up_proj", "down_proj"
    task_type="CAUSAL_LM"
)

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,        # 任务类型
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj"], # 注入位置
    feedforward_modules=["up_proj", "down_proj"], # 可选，FFN 层注入
    inference_mode=False                 # False = 训练模式
)

loha_config = LoHaConfig(
    r=16,                              # 低秩分解维度
    alpha=64,                         # scaling 因子
    rank_dropout=0.05,                    # 代替 LoRA 的 lora_dropout
    module_dropout=0.0,                    # 如果不用模块级dropout就设0
    target_modules=["up_proj","down_proj"],
    task_type=TaskType.CAUSAL_LM
)

lokr_config = LoKrConfig(
    r=12,                           # 低秩分解的秩
    alpha=96,                       # scaling 系数
    target_modules=["up_proj","down_proj"],  # 插入位置
    task_type=TaskType.CAUSAL_LM,  # 任务类型
    rank_dropout=0.05,
    module_dropout=0.05
)



def register_grad_probe_on_module(module, tag="module"):
    """
    在 module 上注册反向钩子：反向时打印/统计梯度信息。
    grad_input / grad_output 是 tuple；有时会存在 None。
    """
    def _hook(mod, grad_input, grad_output):
        # 1) 模块输出的梯度（通常更直观）
        if grad_output and grad_output[0] is not None:
            g = grad_output[0]
            if torch.is_tensor(g):
                print(f"[{tag}] grad_out mean={g.abs().mean().item():.6g} "
                      f"max={g.abs().max().item():.6g} shape={tuple(g.shape)}")

        # 2) 也可以查看该模块参数的梯度（此时已写入 param.grad）
        for n, p in mod.named_parameters(recurse=True):
            if p.requires_grad:
                if p.grad is None:
                    print(f"[{tag}.{n}] grad=None")
                else:
                    print(f"[{tag}.{n}] grad mean={p.grad.abs().mean().item():.6g}")

    # 返回句柄，必要时可以 .remove() 取消挂钩
    return module.register_full_backward_hook(_hook)

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train():
    parser = transformers.HfArgumentParser(   #Hf是 hungging face
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info" # * default is passive(warning)
    # * build logger
    logger = build_logger(__name__, training_args.output_dir + '/train.log')

    
    #if not training_args.force_fsdp or torch.cuda.device_count() == 1 or training_args.use_lora:
    if not training_args.force_fsdp:
        print("🚫 禁用 FSDP：因单卡或启用 LoRA")
        training_args.fsdp = []
        training_args.fsdp_transformer_layer_cls_to_wrap = None
        training_args.force_fsdp = False
        logger.info("✅ FSDP 设置已禁用：fsdp = None, fsdp_transformer_layer_cls_to_wrap = None")

    if training_args.model_debug:
        # * do not load checkpoint, load from config
        config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        model = PointLLMLlamaForCausalLM._from_config(config)
    else:

        model = PointLLMLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
           
        )
        #print(model.model.linear_expand)
        #model.linear_expand = nn.Linear(256, 384)
        #init_weights(model.linear_expand)
        if hasattr(model.model, "point_tune"):
            if not training_args.stage_2:
               model.model.point_tune.apply(init_weights)


            


    model.config.use_cache = False



    mt_prompt_config = MultitaskPromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    tokenizer_name_or_path= model_args.model_name_or_path,
    num_tasks=1#,
    #task_ids=["translation", "text_generation", "sentiment"]
    )
    prompt_cfg = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        tokenizer_name_or_path=model_args.model_name_or_path,
        prompt_tuning_init="TEXT",                  # 比 RANDOM 更易收敛
        prompt_tuning_init_text="Instruction: "    # 按你的任务写一小段指令
    )
    
    #lora的尝试
    if training_args.use_lora:
        if training_args.use_lora_only:
            model = get_peft_model(model, lora_config)
        
        if training_args.use_adalora_only:
            model = get_peft_model(model, adalora_config)
            model.set_adapter("adalora")
        if training_args.use_ia3_only:
            model = get_peft_model(model, ia3_config)
        if training_args.use_loha_only:
            model = get_peft_model(model, loha_config)
        if training_args.use_lokr_only:
            model = get_peft_model(model, lokr_config)
        if training_args.use_mt_prompt_only:
            model = get_peft_model(model, mt_prompt_config)
        if training_args.use_prompt_only:
            model = get_peft_model(model, prompt_cfg)
            



        model.print_trainable_parameters()








    if training_args.fix_llm:
        # * This will fix all the parameters
        logger.info("LLM is fixed. Fix_llm flag is set to True")
        # * fix llama, lm_head, pointnet, projection layer here
        model.requires_grad_(False)  #从embedding到head都冻住了

        model.get_model().fix_llm = True #model.get_model() 是PointLLMLlamaForCausalLM

        model.get_model().point_proj.requires_grad_(True) #point_proj和tune_mm_mlp_adapter
        model.get_model().point_backbone.requires_grad_(True) # * set as True for fsdp, use fix_pointnet flag to control
        model.get_model().point_tune.requires_grad_(True)
        model.get_model().linear_expand.requires_grad_(True)
        if training_args.stage_2:
            model.get_model().point_tune.requires_grad_(False)
            model.get_model().linear_expand.requires_grad_(False)
            
        
        if model_args.tune_layer != 0:
            num_layers = len(model.get_model().layers)
            if model_args.tune_layer > 0:
                for i in range(model_args.tune_layer):
                    model.get_model().layers[i].requires_grad_(True)
        
        
        #lora的尝试

        for name, param in model.named_parameters():
            if "lora_" in name or "ia3" in name  or "hada" in name or "lokr" in name:
                param.requires_grad = True
                    #lora的尝试

    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")
    
    #print(f"[确认] 是否启用 FSDP: {training_args.fsdp is not None}")
    #print(f"[确认] 是否启用 FSDP: {bool(training_args.fsdp)}")


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if not training_args.fix_pointnet:
        # * not fix pointnet
        logger.info("Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded.")
        model.get_model().fix_pointnet = False
    else:
        logger.info("Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded.")
        model.get_model().fix_pointnet = True # * use with torch.inference_mode to control, not requires_grad for fsdp for second stage
        logger.info("Set requires_grad of point backbone to False by Jing")
        model.get_model().point_backbone.requires_grad_(False)  #刚加的
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            model.get_model().point_backbone.requires_grad_(False) # * fix pointnet for first stage, need for fsdp in stage2
    
    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_point_backbone_config
        logger.info("Point projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    if not training_args.stage_2:
        # * we assume in stage2, llm, point_backbone, and projection layer can be loaded from the model checkpoint
        """print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)"""
        #print("embed_tokens requires_grad:", model.get_input_embeddings().weight.requires_grad)
        
        #model.load_pretrained_full_model(path="checkpoints/PointLLM_full_trained")
        model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)
    else:
        # * stage2
        #print("✅ point_backbone 可训练状态:", any(p.requires_grad for p in model.get_model().point_backbone.parameters()))
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer,use_lora=training_args.use_lora) 
        #print("✅ point_backbone 可训练状态:", any(p.requires_grad for p in model.get_model().point_backbone.parameters()))

    point_backbone_config = model.get_model().point_backbone_config
    #print("✅ point_backbone 可训练状态:", any(p.requires_grad for p in model.get_model().point_backbone.parameters()))

    """print("\n=== 参数冻结状态检查（🔥 = 可训练，❄️ = 冻结） ===")
    total, trainable = 0, 0
    for name, param in model.named_parameters():
        num_param = param.numel()
        total += num_param
        if param.requires_grad:
            trainable += num_param
            flag = "🔥 Trainable"
        else:
            flag = "❄️ Frozen"
        print(f"{flag:12} | {name:60} | shape: {tuple(param.shape)}")
    print("=========================================================")
    print(f"总参数量：{total:,}，其中可训练参数量：{trainable:,}（{trainable / total * 100:.4f}%）")
    print("=========================================================\n")"""
    # 创建保存目录
    import os
    from datetime import datetime
    save_dir = "/root/nobug/pl/use_paramters"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "param_status.txt")

    # 构建打印与保存内容
    lines = []
    lines.append("\n=== 参数冻结状态检查（🔥 = 可训练，❄️ = 冻结） ===")
    total, trainable = 0, 0

    for name, param in model.named_parameters():
        num_param = param.numel()
        total += num_param
        if param.requires_grad:
            trainable += num_param
            flag = "🔥 Trainable"
        else:
            flag = "❄️ Frozen"
        line = f"{flag:12} | {name:60} | shape: {tuple(param.shape)}"
        print(line)
        lines.append(line)

    summary = f"总参数量：{total:,}，其中可训练参数量：{trainable:,}（{trainable / total * 100:.4f}%）"
    lines.append("=" * 70)
    lines.append(summary)
    lines.append("=" * 70 + "\n")

    print("=" * 70)
    print(summary)
    print("=" * 70)

    # 写入到本地文件
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n✅ 参数冻结状态已保存至: {save_path}  [{timestamp}]")
    

    
    

    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad] #FSDP（Fully Sharded Data Parallel）时 模型参数部分冻结 + 使用 FSDP = 非标准用法，所以打补丁并提示风险
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_point_data_module(tokenizer=tokenizer,
                                                    data_args=data_args) #加载点云数据
    

    class AdaLoRAUpdateCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            m = kwargs["model"]
            try:
                if hasattr(m, "base_model") and hasattr(m.base_model, "update_and_allocate"):
                    m.base_model.update_and_allocate(state.global_step)
                elif hasattr(m, "update_and_allocate"):
                    m.update_and_allocate(state.global_step)
            except Exception:
                pass
            return control

    trainer = PointLLMTrainer(model=model, #用于在训练过程中自定义保存逻辑，额外保存点云投影模块（如 point_proj）的权重，以便多模态微调（tune_mm_mlp_adapter）时单独加载。
                    tokenizer=tokenizer,
                    args=training_args,
                    #callbacks=[AdaLoRAUpdateCallback()], #只为adalora加的
                    **data_module)
    #trainer.add_callback(AdaLoRAUpdateCallback())#替换方案

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        """trainer.create_optimizer()  # 让它先创建 optimizer

        # 下面加你的检查 point_backbone.reduce_dim
        tune_params = [n for n, p in model.named_parameters() if "point_tune" in n]
        print("point_tune params in model:", len(tune_params))

        opt_params = sum([list(g["params"]) for g in trainer.optimizer.param_groups], [])
        opt_param_ids = {id(p) for p in opt_params}

        missing = [n for n, p in model.named_parameters()
                if "point_tune" in n and id(p) not in opt_param_ids]
        print("Missing from optimizer:", missing)"""
        #point_tune #model.model.linear_expand  #point_backbone.linear_expand
        if not training_args.stage_2:
            handle = register_grad_probe_on_module(model.model.point_tune, tag="point_tune")
        else:
            handle = register_grad_probe_on_module(model.base_model.model.model.point_tune, tag="point_tune")


        trainer.train()  #从这里打标 开始训练
    trainer.save_state()  #到这的时候已经训练结束
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)
    
    if getattr(training_args, "use_lora", False):
        print("🧩 [LoRA检测] 启用LoRA，开始合并并保存完整模型...")

        merged_model = trainer.model.merge_and_unload()

        merged_model.save_pretrained(training_args.output_dir, safe_serialization=False)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        print("❄️ [LoRA检测] 未启用LoRA，无需合并保存完整模型。")


if __name__ == "__main__":
    train()
