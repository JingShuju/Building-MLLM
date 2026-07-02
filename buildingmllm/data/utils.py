from collections import OrderedDict, defaultdict

import transformers
from buildingmllm import conversation as conversation_lib
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch

import numpy as np
import os

IGNORE_INDEX = -100

# * Sample Usage:
# * from utils import LRUCache
# * cache = LRUCache(capacity, max_access_count)
# if self.cache is None:
#     info_data = self.multiview_scannet[info_index]
# else:
#     info_data = self.cache.get(info_index)
#     if info_data is None or self.cache.get_access_count(info_index) >= self.cache.max_access_count:
#         # If not in cache, or accessed max_access_count times, load it and put it in cache
#         info_data = self.multiview_scannet[info_index]
#         self.cache.put(info_index, info_data)
#         self.cache.reset_access_count(info_index)

class LRUCache:
    def __init__(self, capacity, max_access_count):
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.capacity = capacity
        self.max_access_count = max_access_count

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value  # Put key as the newest one
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:  # Update the value and put it as newest
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:  # If cache is full
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)  # Remove oldest item
            del self.access_count[oldest_key]  # Remove the corresponding access count
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        self.access_count[key] = 0


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy() #这里获得的对话模型是conv_vicuna_v1_1
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}#这里roles = {'human': 'USER', 'gpt': 'ASSISTANT'}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
    #上面的sourse是
    #source = [{'from': 'human', 'value': '<point_start><point_patch><point_patch><point_patch><point_patch><point_patch><point_...tch><point_patch><point_end>\nwhat is this?'}, {'from': 'gpt', 'value': 'rectangular duct segment'}]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    #print(conversations)
    #最终上面生成格式统一、能直接输入 LLM 的 prompt #USER: <point_start>...<point_end> What is this? #ASSISTANT: It's a rectangular duct.</s>
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids   
    #input_ids 内容是tensor([[    1,   319, 13563, ..., 32001,32000, ..., 32002,   825, ...,     2]])#其中前面是问题中间32000重复是点云token后面接着是文本
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " #sep = ' ASSISTANT: ' 是回答的部分
    for conversation, target in zip(conversations, targets): #conversation是完整的单词 #target是对应的数字
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) #700
        #print(conv.sep2) #</s>是终止符
        rounds = conversation.split(conv.sep2)
        #print(rounds) #把分隔符分成了三段对话引号
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):  #取第一段问答 有问有答
            if rou == "":
                break

            parts = rou.split(sep) #将单轮问答 分割成 问和答 
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep #这一步将<assistant>:键入到问中 parts[0] = "<user>: What is AI?<assistant>: "
            round_len = len(tokenizer(rou).input_ids) #单轮问答的总长度
            #print(round_len) #587 单轮问答总长度
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2 #单轮问答中问的长度
            #print(instruction_len) #570 单轮问题总长度
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            #print("Masked target:")
            #print(target.tolist())


            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens
        #target就是三轮问答 以上操作就是把三轮问答中仅保留回答
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    #print(f"input_ids.shape = {input_ids.shape}, labels.shape = {targets.shape}")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
    
def preprocess_multimodal_point_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    point_token_len = point_backbone_config['point_token_len'] #point_token_len = 513 输入点云encoder中的token 512+1 class
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    
    """point_backbone_config = {
    'point_cloud_dim': 6,  # 6维度点云数据，xyzrgb
    'backbone_output_dim': 384, #点云 backbone（如 PointNet++, Point-BERT）输出的 token 特征维度是 384。
    'project_output_dim': 4096, #点云特征投影到跨模态融合时的高维空间维度，通常用于对齐语言或视觉模态（如 LLM 的嵌入空间）。
    'point_token_len': 513, #点云编码后 token 的个数。一般是 512 个 patch token + 1 个 <CLS> token（或 point_start_token）。
    'mm_use_point_start_end': True, #是否在点云 token 序列前后加入 <point_start> 和 <point_end> token，方便多模态融合对齐（如 LLM）。
    'projection_hidden_layer': 2, #点云到跨模态空间的投影网络（如 MLP）使用 2 层隐藏层。
    'use_max_pool': False, #是否使用 max pooling 来聚合 patch token。如果为 True，表示使用全局池化，可能只生成一个 token（即 point_token_len=1）。
    'projection_hidden_dim': [1024, 2048], #是否使用 max pooling 来聚合 patch token。如果为 True，表示使用全局池化，可能只生成一个 token（即 point_token_len=1）。
    'default_point_patch_token': '<point_patch>', # 表示 patch token 的占位符，适用于 prompt 设计（如 <point_patch>...<point_patch>）。
    'point_patch_token': 32000, #<point_patch> token 在 tokenizer 中的 token ID，通常添加到 vocab 的最后面（例如在原始 vocab=32000 后）。
    'default_point_start_token': '<point_start>', #点云 token 序列的起始符号，用于多模态融合和 prompt control。
    'default_point_end_token': '<point_end>', #点云 token 序列的结束符号。
    'point_start_token': 32001, #<point_start> token 在 tokenizer 中的 token ID，通常添加到 vocab 的最后面（例如在原始 vocab=32000 后）。
    'point_end_token': 32002 #<point_end> token 在 tokenizer 中的 token ID，通常添加到 vocab 的最后面（例如在原始 vocab=32000 后）。
    }"""
    #print(sources)
    #[[{'from': 'human', 'value': '<point>\nWhat color is the 3D model of the Mercedes Benz supercar?'}, {'from': 'gpt', 'value': 'The 3D model of the Mercedes Benz supercar is grey.'}, 
    # {'from': 'human', 'value': 'What style is the model designed in?'}, {'from': 'gpt', 'value': 'The model is rendered in a cartoonish style, which distinguishes it from realistic renderings.'}, 
    # {'from': 'human', 'value': 'Despite the cartoonish rendering, does the model retain any typical features of Mercedes Benz supercars?'}, 
    # {'from': 'gpt', 'value': 'Yes, despite its cartoonish design, the model retains recognisable features of the Mercedes Benz brand, including its aerodynamic shape and sleekness usually associated with supercars.'}]]
    for source in sources: #source 就是把上面的一个中括号去掉
        for sentence in source: #sentence就是一个大括号的内容
            replace_token = default_point_patch_token * point_token_len 
            if point_backbone_config['mm_use_point_start_end']:
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

    #print(sources)
    return sources

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    filename = f"{object_id}_{pointnum}.npy"
    point_cloud = np.load(os.path.join(data_path, filename))

    # * normalize
    point_cloud = pc_norm(point_cloud)

    if not use_color:
        point_cloud = point_cloud[:, :3]

    return point_cloud

@dataclass
class DataCollatorForPointTextDataset(object):
    """Collate examples for mixed dataset with text and point cloud data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(  #自动补齐到最大长度
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'point_clouds' in instances[0]:
            point_clouds = [instance['point_clouds'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['point_clouds'] = torch.stack(point_clouds)
            else:
                batch['point_clouds'] = point_clouds # * return as lists

        """# ✅ 调试输出：打印 batch 中各字段的 shape 和部分内容
        print("=== Collated Batch Info ===")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape = {value.shape}") # #input_ids: shape = torch.Size([1, 564]) #labels: shape = torch.Size([1, 564]) #attention_mask: shape = torch.Size([1, 564]) #point_clouds: shape = torch.Size([1, 12000, 6])
                if key == "input_ids":                                          #attention_mask给input_ids展示有效tokens  #564维度包含了系统描述。问题文本、点云toekns、回答文本
                    print("input_ids (first row):", value[0].tolist()[:50])
                if key == "labels":
                    print("labels (first row):", value[0].tolist()[:50])
            elif isinstance(value, list):
                print(f"{key}: list of length {len(value)}")
                if len(value) > 0:
                    print(f"  first item shape = {value[0].shape}")
        print("============================")"""

        return batch

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    """
    pc: Nx3 array
    This functions normalizes a point cloud to fit within a unit sphere.
    It first calculates the centroid of the point cloud and then subtracts
    it from all points before scaling all points to fit within a unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc