import argparse
import torch
from torch.utils.data import DataLoader
import os
from buildingmllm.conversation import conv_templates, SeparatorStyle
from buildingmllm.utils import disable_torch_init
from buildingmllm.model import *
from buildingmllm.model.utils import KeywordsStoppingCriteria
from buildingmllm.data import ObjectPointCloudDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from buildingmllm.eval.evaluator import start_evaluation

import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"   # ✅ 修复 decoder-only 模型警告
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.float32).cuda() #torch_dtype=torch.bfloat16
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer, use_lora=False)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None # * load point cloud only
    )
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation(model, tokenizer, conv, dataloader, annos, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    results = {"prompt": "use_human_question"}
    responses = []

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    

    
    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype)  # B, N, C
        object_ids = batch["object_ids"]  # list of string
        batch_prompts = []
        batch_questions = []  # 👈 在 for batch 之前新建列表

        # 🔹 为每个样本构造它自己的 human 提问
        for obj_id in object_ids:
            if obj_id not in annos:
                print(f"[Warning] object_id {obj_id} not found in annotations.")
                continue

            #human_q = annos[obj_id]["conversations"][0]["value"]
            # 去掉 <point>\n 或 <point>
            human_q = annos[obj_id]["conversations"][0]["value"].replace("<point>\n", "").replace("<point>", "").strip()

            batch_questions.append(human_q)  # 👈 新增，保存干净的 human_q

            if mm_use_point_start_end:
                qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + human_q
            else:
                qs = default_point_patch_token * point_token_len + '\n' + human_q
            


            conv_single = conv.copy()
            conv_single.append_message(conv.roles[0], qs)
            conv_single.append_message(conv.roles[1], None)

            batch_prompts.append(conv_single.get_prompt())

        # tokenizer 处理所有 prompt
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()

        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # 🔹 模型生成输出
        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria)

        # 🔹 保存结果
        #for obj_id, output in zip(object_ids, outputs):
        for obj_id, human_q, output in zip(object_ids, batch_questions, outputs):
            responses.append({
                "object_id": obj_id,
                "question": human_q,  # 👈 改成用 batch_questions 里的 human_q
                "ground_truth": annos[obj_id]["conversations"][1]["value"],
                "model_output": output
            })

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name, "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    # * load annotation files 一次性加载
    with open(args.anno_path, 'r') as fp:
        annos_all = json.load(fp)

    # * 遍历多个 conversation_type
    for conv_type in args.conversation_types:  # 👈 循环多个类型
        anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
        output_file = f"{anno_file}_{conv_type}.json"
        output_file_path = os.path.join(args.output_dir, output_file)

        # * First inferencing, then evaluate
        if not os.path.exists(output_file_path):
            # * dataset 按 conv_type 加载
            dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, (conv_type,), args.use_color)
            dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
            
            model, tokenizer, conv = init_model(args)

            # * 过滤 annos
            annos = {
                anno["object_id"]: anno
                for anno in annos_all
                if anno["conversation_type"] == conv_type
            }

            print(f'[INFO] Start generating results for {output_file}.')
            results = start_generation(model, tokenizer, conv, dataloader, annos, args.output_dir, output_file)

            # * release model and tokenizer, and release cuda memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
        else:
            # * directly load the results
            print(f'[INFO] {output_file_path} already exists, directly loading...')
            with open(output_file_path, 'r') as fp:
                results = json.load(fp)

        if args.start_eval:
            evaluated_output_file = output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
            eval_type_mapping = {
                "captioning": "object-captioning",
                "classification": "open-free-form-classification"
            }
            start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="outputs/PointLLM_train_stage2/PointLLM_train_stage2_lora_20250715_2220") 

    # * dataset type
    parser.add_argument("--data_path", type=str, default="data/obset-combined-2", required=False)
    parser.add_argument("--anno_path", type=str, default="data/anno_data/324_complex_test.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=2)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"], help="Type of the task to evaluate.")


    parser.add_argument("--conversation_types", nargs="+", type=str, default=["2_Knowledge_Capability"], help="Which conversation type to use ( e.g. 1_General_Visual_Recognition, 2_Knowledge_Capability, etc. )") 
    ##--conversation_types "1_General_Visual_Recognition" "2_Knowledge_Capability" "3_Commonsense_Reasoning" "4_Advanced_Engineering_Reasoning" "5_Functional_Comparison" 
    #"6_Constraint_Reasoning" "7_Spatial_Relationship" "8_Embodied_Interaction" "9_1-8" "10_2-8"  \
       


    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    """if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError"""

    main(args)