#    Copyright 2023 Runsen Xu

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from buildingmllm.utils import *

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import os
#from pointllm.model.loss import *
import numpy as np
# * add logger
import logging
logger = logging.getLogger(__name__)



def stat(t, tag):
    t = t.detach()
    nz = (t != 0).float().mean().item()
    print(f"{tag}: mean={t.float().mean():.3e} std={t.float().std():.3e} "
          f"min={t.min().item():.3e} max={t.max().item():.3e} nonzero={nz:.4f}")
    


class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"

class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            from buildingmllm.model import PointTransformer
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False
        self.fix_llm = False
        


        self.linear_expand = nn.Linear(256, 384)
        from buildingmllm.model import PointTUNE
        self.use_point_tune = getattr(config, "use_point_tune", True)
        if self.use_point_tune:        
            self.point_tune = PointTUNE(config)

            

            if self.point_tune.mae_fp != 2:
                    #print(f"[Debug] self.point_tune.mae_fp = {self.point_tune.mae_fp} (≠ 2)")
                    self.point_tune.mask_token = nn.Parameter(torch.zeros(1, 1, 4096))
                    self.point_tune.pos_embed_mae = False
                    #self.point_tune.pos_embed_type = 0
                    if self.point_tune.pos_embed_type != 0:
                        self.point_tune.pos_embed_mae = True
                        
                        self.point_tune.decoder_pos_embed = nn.Sequential(
                            nn.Linear(3, 128),
                            nn.GELU(),
                            nn.Linear(128, 4096)
                        )

    
    
    # mae #随机遮盖一部分 patch，以训练重建被遮盖点的自编码器（MAE）模型。
    #每个样本独立随机 mask。
    #cls_token 永不被 mask。
    #最终输出一个 (B, G) 的布尔张量，True 表示对应 group 被遮盖。
    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape   #[1, 129, 4096]
        G = G - 1  
        # skip the mask  #[1, 128]
        if noaug or self.point_tune.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.point_tune.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        cls_token = torch.from_numpy(np.zeros([B, 1])).to(torch.bool)
        end_mask = torch.cat([cls_token,overall_mask],dim=1)
        return end_mask.to(center.device) # B G







    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)

    def forward(
        self,
        #position_ids: Optional[torch.LongTensor] = None,  # ✅ 加上这一行
        input_ids: torch.LongTensor = None, ##torch.Size([1, 564])
        attention_mask: Optional[torch.Tensor] = None, #torch.Size([1, 564]) 全是True
        past_key_values: Optional[List[torch.FloatTensor]] = None, #past_key_values = None
        inputs_embeds: Optional[torch.FloatTensor] = None, #inputs_embeds = None
        use_cache: Optional[bool] = None, #use_cache = None
        output_attentions: Optional[bool] = None, #output_attentions = False
        output_hidden_states: Optional[bool] = None, #output_hidden_states = False
        point_clouds: Optional[torch.FloatTensor] = None,  #torch.Size([1, 12000, 6])
        return_dict: Optional[bool] = None,  #return_dict = True
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        #print("orig_embeds_params:", orig_embeds_params)
        #print(f"[DEBUG] position_ids in forward: {position_ids.shape if position_ids is not None else None}")

 
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            #print("Is embed_tokens trainable?:", self.embed_tokens.weight.requires_grad)

            #print(inputs_embeds.shape) #torch.Size([1, 564, 4096]) #torch.Size([4, 714, 4096])
        point_backbone = getattr(self, 'point_backbone', None)
        #print("point_backbone:", type(point_backbone), point_backbone) #将点云[B, N, 6]编码成 [B, num_group, 384]
        #输入通道数	6（通常为 XYZ + RGB）
        #输出维度	384
        #Transformer 层数	12 层
        #MLP hidden dim	1536（= 4×384） Linear (D → 4D)
        #Self-attention	多头 attention（qkv, proj）
        #使用 pos_embed	是（3D → MLP → 384）
        #DropPath / Dropout	有设置（部分 block 为 DropPath）
        point_backbone_config = getattr(self, 'point_backbone_config', None)
        #print("point_backbone_config:", type(point_backbone_config), point_backbone_config)
        #下面是point_backbone_config 点云编码器的配置参数
        #{'point_cloud_dim': 6, 'backbone_output_dim': 384, 'project_output_dim': 4096, 'point_token_len': 513, 'mm_use_point_start_end': True, 'projection_hidden_layer': 2, 
        # 'use_max_pool': False, 'projection_hidden_dim': [1024, 2048], 'default_point_patch_token': '<point_patch>', 'point_patch_token': 32000, 'default_point_start_token': '<point_start>',
        #  'default_point_end_token': '<point_end>', 'point_start_token': 32001, 'point_end_token': 32002}
        #下面表示：当模型处于训练状态（或输入 token 长度大于 1），并且启用了点云主干网络(point_backbone) 且传入了点云数据 (point_clouds) 时，才执行后续点云相关操作。
        point_token_start = []
        xyz = []
        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            # * enter when training or the first generation step of inference
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                if type(point_clouds) is list:
                    # * variable numbers of points
                    point_features = []
                    for point_cloud in point_clouds: # * iterate over batch
                        point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                        point_features.append(point_feature)
                else:
                    #print(point_clouds.shape)  #torch.Size([1, 12000, 6])
                    #point_features = self.point_backbone(point_clouds)
                    point_features, xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb = self.point_backbone(point_clouds)
                    #print(f"✅ xyz shape: {xyz.shape}")             # [B, G, 3]
                    #print(f"✅ lc_x shape: {lc_x.shape}")           # [B, G, C]
                    #print(f"✅ rgb shape: {rgb.shape}")             # [B, G, C_rgb]
                    #print(f"✅ knn_xyz shape: {knn_xyz.shape}")     # [B, G, K, 3]
                    #print(f"✅ knn_x shape: {knn_x.shape}")         # [B, G, K, C]
                    #print(f"✅ knn_rgb shape: {knn_rgb.shape}")                   
                    #print(point_features.shape) #torch.Size([1, 513, 384])
            #从这个位置添加adapter  point_tune
            #stat(knn_x, ">> in: point_tune input <<")
            #stat(self.linear_expand.weight.data, "linear_expand")
            #print(self.linear_expand.weight.requires_grad)
            knn_x = knn_x.requires_grad_()
            knn_x = self.linear_expand(knn_x)
            #stat(knn_x, ">> in: point_tune input <<")

            #print(any(p.requires_grad for p in self.linear_expand.parameters()))
            #print(knn_x.shape)
            if self.use_point_tune: 
                feature_adapter, feature_frozen, feature_plus, knn_xyz, xyz = self.point_tune(point_features, xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb)
                point_features_ori = feature_plus#feature_frozen#feature_plus
                
                """current_dtype = point_features.dtype
                with torch.autocast("cuda", enabled=False):
                    feature_adapter, feature_frozen, feature_plus, knn_xyz, xyz = self.point_tune(
                        point_features.float(),  # 保证输入也是 FP32
                        xyz.float(),
                        lc_x.float(),
                        rgb.float(),
                        knn_xyz.float(),
                        knn_x.float(),
                        knn_rgb.float()
                    )
                
                    # 转回原精度
                feature_adapter = feature_adapter.to(current_dtype)
                feature_frozen  = feature_frozen.to(current_dtype)
                feature_plus    = feature_plus.to(current_dtype)
                knn_xyz         = knn_xyz.to(current_dtype)
                xyz             = xyz.to(current_dtype)
                point_features_ori = feature_plus"""
                
                #feature_adapter , feature_frozen, feature_plus[1, 513, 384] #knn_xyz[1, 512, 32, 3] #xyz[1, 512, 3]
            #print(any(p.requires_grad for p in self.point_tune.parameters()))
            # 假设你的模块叫 my_module
            """for name, param in self.point_tune.named_parameters():
                if param.grad is None:
                    print(f"{name}: ❌ 没有梯度 (grad=None)")
                elif torch.all(param.grad == 0):
                    print(f"{name}: ⚠ 梯度全 0")
                else:
                    print(f"{name}: ✅ 梯度正常, mean={param.grad.abs().mean().item():.6f}")"""

            mae_gts = None  #patch.feature
            recon_gts_points = None   # patch
            recon_gts_features = None  # feature
            bool_masked_pos = None
            point_features_vis = 0
            point_features_mask = 0
            pos_emd_vis = None
            pos_emd_mask = None



            if type(point_clouds) is list:
                point_features_ori = [self.point_proj(point_feature) for point_feature in point_features_ori]
            else:
                point_features_ori = self.point_proj(point_features_ori)
                #print("self.point_proj =", self.point_proj)
                #print(point_features.shape) #torch.Size([1, 513, 4096])
            

            if self.point_tune.mask_dim  == 4096 and (self.point_tune.mask_ratio == 0 or self.point_tune.mask_ratio == 0.3):
                bool_masked_pos = self._mask_center_rand(point_features_ori, noaug = False)
                batch_size, seq_len, _ = point_features_ori.size()
                point_features_vis = point_features_ori[~bool_masked_pos].reshape(batch_size, -1, self.point_tune.mask_dim)
                #print(point_features_vis.shape)#torch.Size([4, 360, 4096])  #掩码掩的是tokens

                recon_gts_points = knn_xyz[~bool_masked_pos[:,1:]].reshape(batch_size, -1, knn_xyz.size()[2], knn_xyz.size()[3])
                #print(recon_gts_points.shape)# torch.Size([1, 90, 81, 3])#未被遮挡点云的patches
                if self.point_tune.pos_embed_mae:
                    pos_emd_vis = self.decoder_pos_embed(xyz[~bool_masked_pos[:,1:]]).reshape(batch_size, -1, self.point_tune.pos_embed_dim)
                    pos_emd_mask = self.decoder_pos_embed(xyz[bool_masked_pos[:,1:]]).reshape(batch_size, -1, self.point_tune.pos_embed_dim)
                
                if self.point_tune.mae_fp==0:
                    mae_gts = knn_xyz[bool_masked_pos[:,1:]].reshape(batch_size, -1, knn_xyz.size()[2], knn_xyz.size()[3])
                elif self.point_tune.mae_fp==1:
                    mae_gts = point_features_ori[bool_masked_pos].reshape(batch_size, -1, self.point_tune.mask_dim)
                    #print( mae_gts.shape) #torch.Size([1, 38, 4096]) 被遮挡的toekns 的真实值
                mask_token = self.point_tune.mask_token.expand(batch_size, mae_gts.size()[1], -1)
                #print(mask_token.shape) #torch.Size([1, 38, 4096]) 被遮挡的toekns 的随机初始值

                if self.training:
                    point_features = torch.cat([point_features_vis,mask_token],dim=1) #torch.Size([1, 91, 4096]) +  torch.Size([1, 38, 4096])  合起来是torch.Size([1, 129, 4096])
                else:
                    point_features = point_features_ori
                #截止到上面 我们获得了point_features[1, 129, 4096]带有遮挡 #mask_token[1, 38, 4096] #mae_gts[1, 38, 4096] 
            # #point_features_vis[1, 91, 4096] #recon_gts_points [1, 90, 81, 3]
            if self.point_tune.recon_pos == 1:
                if bool_masked_pos is not None:
                    batch_size, seq_len, _ = point_features_ori.size()
                    recon_gts_features = point_features_ori[~bool_masked_pos].reshape(batch_size, -1, 4096)
                    #print(recon_gts_features.shape)
                else:
                    recon_gts_features = point_features_ori
            if (self.point_tune.mask_dim == 384) or (self.point_tune.mask_ratio != 0 and self.point_tune.mask_ratio != 0.3):
                point_features = point_features_ori



            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_point_features = self.point_proj(dummy_point_features)


            new_input_embeds = []
            cur_point_idx = 0                                   #input_ids torch.Size([1, 564]) # inputs_embeds torch.Size([1, 564, 4096])
            mask_start = [] 
            full_reconstruction_start = []
            vis_pos_start = []
            mask_pos_start = []
                    #point_features[1, 129, 4096]带有遮挡
                    #mask_token[1, 38, 4096] #mae_gts[1, 38, 4096] 
                    #point_features_vis[1, 91, 4096]   = recon_gts_features[1, 91, 4096]
                    #recon_gts_points [1, 90, 81, 3]



            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C #遍历每个样本的token ID 序列和其对应的token embedding
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0: #检查当前样本是否包含 <point_patch> token #例如 32000）
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue   #只要cur_input_ids含有32000 那么new_input_embeds就是cur_input_embeds #torch.Size([1, 564, 4096])
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device) ##torch.Size([1, 513, 4096]) #把点云和文本数据放到一个仪器上 方便以后拼接
                num_patches = cur_point_features.shape[0] # * number of point tokens  #513
                if point_backbone_config['mm_use_point_start_end']:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0] #找到cur_input_ids点云开始位置的索引 是在34

                    point_token_start.append(point_start_tokens+1+1)
    
                    if isinstance(point_features_vis, int):
                        mask_start.append(point_start_tokens+1)  # mae  
                        vis_pos_start.append(point_start_tokens+2)  
                        mask_pos_start.append(point_start_tokens+2)
                    else: #point_start_tokens是在34
                        mask_start.append(point_start_tokens + point_features_vis.size()[1]+1)  #mask开始的位置
                        vis_pos_start.append(point_start_tokens+2) #真正点云开始的地方
                        mask_pos_start.append(point_start_tokens+1+point_features_vis.size()[1]) #和mask_start一样
                    if self.point_tune.recon_fp == 1:  #这里存疑！！！
                        full_reconstruction_start.append(point_start_tokens + 1) # t->t  full reconstruction
                    else:
                        full_reconstruction_start.append(point_start_tokens + 1 + 1) # t->t  full reconstruction #真正点云开始的地方




                    for point_start_token_pos in point_start_tokens:
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                            #以上通过torch.cat 将 embedding 分段拼接成新输入，确保只有点云特征部分是可训练的，其余 embedding 全部 detach() 冻结，从而实现 “局部可学习 embedding 替换” 的目标。
                            #print(cur_new_input_embeds.shape)  #torch.Size([564, 4096])
                        else:
                            #cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            #print(inputs_embeds)
            #print(inputs_embeds.shape)
            #PointLLMLlamaModel的输出是inputs_embeds=torch.Size([1, 564, 4096]) #将点云和文本融合了564  点云采的是513 后续看看这些数字之间有什么关系
        
#TypeError: forward() got an unexpected keyword argument 'position_ids'
        output =  super(PointLLMLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict#, position_ids=position_ids  # ✅ 加上这个
        )  #这个return 将融合后的文本加点云embedding 送入 LLaMA 编码器，进行完整的 token-to-token 自注意力计算、上下文编码、输出 last_hidden_state。
        if self.training:
            """print("✅ output.shape:", output.shape if hasattr(output, "shape") else type(output))#[4, 181, 4096] LLM的输出
            print("✅ mae_gts.shape:", mae_gts.shape)  #[4, 38, 4096] 遮挡的真实toukens
            print("✅ recon_gts_points.shape:", recon_gts_points.shape)  #[4, 90, 81, 3]  #未遮挡的真实的patch 
            print("✅ recon_gts_features.shape:", recon_gts_features.shape) #[4, 91, 4096] #未遮挡的真实toekns
            print("✅ mask_start:", mask_start) #126
            print("✅ full_reconstruction_start:", full_reconstruction_start) #36"""
            return output,mae_gts,recon_gts_points,recon_gts_features,mask_start,full_reconstruction_start
        #output是
        else:
            return output


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.cd_loss = ChamferDistanceL2().cuda()
        #if hasattr(self.model, "point_tune"):
            #self.point_tune = self.model.point_tune
        # mae 
        if self.model.point_tune.mae_fp ==0:
            self.model.point_tune.mae_predict_head =  nn.Sequential(
                nn.Linear(4096,2048),
                nn.GELU(),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 96) #config.point_pn_params['group_size']*3
            )
        elif self.model.point_tune.mae_fp == 1:  #进行遮挡的损失计算#[4, 38, 4096]
            if self.model.point_tune.mask_dim==4096:
                self.model.point_tune.mae_predict_head =  nn.Sequential(
                    nn.Linear(4096,1024),
                    nn.GELU(),
                    nn.Linear(1024, 4096)
                )
            else:
                self.model.point_tune.mae_predict_head =  nn.Sequential(
                    nn.Linear(4096,self.model.point_tune.mask_dim),
                    nn.GELU(),
                    nn.Linear(self.model.point_tune.mask_dim, self.model.point_tune.mask_dim)
                )

        # recon
        if self.model.point_tune.recon_fp==0:
            self.model.point_tune.recon_predict_head =  nn.Sequential(
                nn.Linear(4096,2048),
                nn.GELU(),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 96)#config.point_pn_params['group_size']*3
            )
        elif self.model.point_tune.recon_fp==1:
            if self.model.point_tune.recon_pos == 0:
                self.recon_predict_head =  nn.Sequential(
                    nn.Linear(4096,2304),
                    nn.GELU(),
                    nn.Linear(2304, 2304)
                )
            elif self.model.point_tune.recon_pos == 1:  
                self.model.point_tune.recon_predict_head =  nn.Sequential(
                    nn.Linear(4096,1024),
                    nn.GELU(),
                    nn.Dropout(p=0.1), #刚加的
                    nn.Linear(1024, 4096)
                )

        # Initialize weights and apply final processing
        self.post_init()




    def get_model(self):
        return self.model
    #outputs = model(**batch)这个的输出就输送到下面
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, # * control whether to return past_key_values
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        #position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #ObjectPointCloudDataset(Dataset)转到DataCollatorForPointTextDataset跳转到下面来的
        #input_ids: shape = torch.Size([4, 564]) #labels: shape = torch.Size([4, 564]) #attention_mask: shape = torch.Size([4, 564]) #point_clouds: shape = torch.Size([4, 8192, 6])
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if self.training:

            outputs,mae_gts,recon_gts_points,recon_gts_features,mask_start,full_reconstruction_start   = self.model(
                input_ids=input_ids,    #torch.Size([1, 564])
                attention_mask=attention_mask, #torch.Size([1, 564]) 全是True
                past_key_values=past_key_values,#past_key_values = None
                inputs_embeds=inputs_embeds, #inputs_embeds = None  #inputs_embeds=torch.Size([1, 564, 4096]) 经过LLM的融合结果
                use_cache=use_cache, #use_cache = None
                output_attentions=output_attentions, #output_attentions = False
                output_hidden_states=output_hidden_states, #output_hidden_states = False
                return_dict=return_dict, #return_dict = True
                point_clouds=point_clouds,#, #torch.Size([1, 12000, 6])
                #position_ids=position_ids
            )
        else:
            outputs = self.model(  # mae 
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                point_clouds=point_clouds,#, #torch.Size([1, 12000, 6])
                #position_ids=position_ids
            )  
        #以上转到PointLLMLlamaModel中
        #print(outputs)
        #print("last_hidden_state shape:", outputs.last_hidden_state.shape) #last_hidden_state shape: torch.Size([1, 564, 4096])
        #print(outputs.shape)
        hidden_states = outputs[0]
        

        if self.training and (self.model.point_tune.mask_ratio==0.3 or self.model.point_tune.mask_ratio==0):
            if isinstance(mae_gts, list):
                gts_point_size = mae_gts[0].size()[1]
            else:
                gts_point_size = mae_gts.size()[1]    #38 遮挡的toekn的个数        # mae 
            mask_tokens_list = []
            for i in range(hidden_states.size()[0]):
                mask_tokens_list.append(hidden_states[i,mask_start[i]:mask_start[i]+gts_point_size,:])
            mask_tokens = torch.stack(mask_tokens_list,dim=0)
            #print("mask_tokens.shape:", mask_tokens.shape) #[4, 38, 4096] mask_tokens是llm输出的遮挡的部分的tokens
            if self.model.point_tune.mae_fp==0:
                B,G,_,_ = mae_gts.shape
                pred_points = self.model.point_tune.mae_predict_head(mask_tokens).reshape(B*G,-1,3).float()
                mae_gts = mae_gts.reshape(B*G,-1,3).float()
                mae_loss = self.cd_loss(pred_points,mae_gts)
            elif self.model.point_tune.mae_fp==1: #进行的是这一个
                B,G,_ = mae_gts.shape #[4, 38, 4096]
                pred_points = self.model.point_tune.mae_predict_head(mask_tokens).reshape(B, G, self.model.point_tune.mask_dim).float()

                #print("pred_points.shape:", pred_points.shape) #[4, 38, 4096] 预测的遮挡的部分的tokens
                mae_gts = mae_gts.reshape(B, G, self.model.point_tune.mask_dim).float()
                #print("mae_gts.shape:", mae_gts.shape) #[4, 38, 4096] 遮挡的真实值
                if self.model.point_tune.mae_feature ==0:
                    mae_loss = (pred_points - mae_gts) ** 2
                    mae_loss = mae_loss.mean(dim=-1)  
                    mae_loss = (mae_loss.sum()) / (B*G)  #这里就是计算的遮挡部分的损失 计算的是LLM输出的遮挡部分的tokens经过映射 与被遮挡真实的tokens 进行损失计算
        else:
            mae_loss = None


        if self.training and (self.model.point_tune.mask_ratio==0.3 or self.model.point_tune.mask_ratio==0):    
            reconstruction_tokens_list = []
            for i in range(hidden_states.size()[0]):
                if self.model.point_tune.recon_fp==0: #是这个
                    reconstruction_tokens_list.append(hidden_states[i,full_reconstruction_start[i]:full_reconstruction_start[i]+recon_gts_points.size()[1],:])
                else: #打印一下下面看看那
                    reconstruction_tokens_list.append(hidden_states[i,full_reconstruction_start[i]:full_reconstruction_start[i]+recon_gts_features.size()[1],:])
            reconstruction_tokens = torch.stack(reconstruction_tokens_list,dim=0)
            #print("reconstruction_tokens.shape:", reconstruction_tokens.shape) #[4, 90, 4096] #未遮挡的tokens

            if self.model.point_tune.recon_fp==0:
                B,G,_,_ = recon_gts_points.shape # patch
                pred_points = self.model.point_tune.recon_predict_head(reconstruction_tokens).reshape(B*G,-1,3).float()
                #print(pred_points.shape)
                recon_gts_points = recon_gts_points.reshape(B*G,-1,3).float()
                #print(recon_gts_points.shape)
                full_reconstruction_loss = self.cd_loss(pred_points,recon_gts_points)
            elif self.model.point_tune.recon_fp==1:
                B,G,_ = recon_gts_features.shape # center
                pred_points = self.model.point_tune.recon_predict_head(reconstruction_tokens).reshape(B, G, -1).float()
                #print(pred_points.shape)
                recon_gts_features = recon_gts_features.reshape(B, G, -1).float()
                #print(recon_gts_features.shape)
                if self.model.point_tune.recon_feature ==0:
                    # 去掉第一个 token，只保留 patch tokens
                    pred_patch = pred_points[:, 1:, :]            # [B, G-1, C]
                    gt_patch = recon_gts_features[:, 1:, :].detach()       # [B, G-1, C]
                    full_reconstruction_loss = (pred_patch - gt_patch) ** 2
                    #full_reconstruction_loss = (pred_points - recon_gts_features) ** 2
                    full_reconstruction_loss = full_reconstruction_loss.mean(dim=-1)  
                    #full_reconstruction_loss = (full_reconstruction_loss.sum()) / (B*G) 
                    full_reconstruction_loss = (full_reconstruction_loss.sum()) / (B * (G - 1)) 
        else:
            full_reconstruction_loss = None

        #print("lm_head requires_grad:", self.lm_head.weight.requires_grad)
        logits = self.lm_head(hidden_states)
        #print("hidden_states.shape:", hidden_states.shape) #hidden_states.shape: torch.Size([1, 564, 4096])
        #print("logits.shape:", logits.shape) #logits.shape: torch.Size([1, 564, 32003])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)  #去掉最后一个token的预测
            shift_labels = labels[..., 1:].contiguous() # * B, L  #去掉第一个toekn
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # # [B*(L-1), V]
            shift_labels = shift_labels.view(-1) #[B*(L-1)]
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            #loss = loss_fct(shift_logits, shift_labels)
            loss_ce = loss_fct(shift_logits, shift_labels)  
            if mae_loss is not None and full_reconstruction_loss is not None:
                #loss = loss_ce + mae_loss + full_reconstruction_loss  #执行的是这个
                loss = loss_ce + 0.05 * full_reconstruction_loss # 0.05~0.2
                #print(f"loss_ce: {loss_ce.item():.4f}, mae_loss: {mae_loss.item():.4f}, full_reconstruction_loss: {full_reconstruction_loss.item():.4f}, total_loss: {loss.item():.4f}")
                #loss_ce: 8.2581, mae_loss: nan, full_reconstruction_loss: 228.1180, total_loss: nan
            else:
                loss = loss_ce
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        #到这里是完成了一次训练
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )






    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer, use_lora):  #第二阶段
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
        
        if use_lora:
            self.get_input_embeddings().weight.requires_grad = True #True
            self.get_output_embeddings().weight.requires_grad = True #True
            print("✅ [LoRA启用] 输入输出 embedding 已解冻")
        else:
            self.get_input_embeddings().weight.requires_grad = False
            self.get_output_embeddings().weight.requires_grad = False
            print("❄️ [LoRA关闭] 输入输出 embedding 已冻结")

        #print("embed_tokens requires_grad:", self.get_input_embeddings().weight.requires_grad)
        
        #print("embed_tokens requires_grad:", self.get_output_embeddings().weight.requires_grad)
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True): #第一阶段

        config = self.config
        #print(config )
        """PointLLMConfig {
        "DEFAULT_POINT_END_TOKEN": "<point_end>",
        "DEFAULT_POINT_PATCH_TOKEN": "<point_patch>",
        "DEFAULT_POINT_START_TOKEN": "<point_start>",
        "_name_or_path": "checkpoints/PointLLM_7B_v1.1_init",
        "architectures": [
            "PointLLMLlamaForCausalLM"
        ],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "mm_use_point_start_end": true,
        "model_type": "pointllm",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "pad_token_id": 0,
        "point_backbone": "PointBERT",
        "point_backbone_ckpt": "",
        "point_backbone_config_name": "PointTransformer_8192point_2layer",
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": false,
        "torch_dtype": "float16",
        "transformers_version": "4.28.0.dev0",
        "use_cache": false,
        "use_color": true,
        "vocab_size": 32000
        }"""
        #print("embed_tokens requires_grad:", self.get_input_embeddings().weight.requires_grad)
        
        #print("embed_tokens requires_grad:", self.get_output_embeddings().weight.requires_grad)
        point_backbone_config = self.get_model().point_backbone_config
        #print(point_backbone_config)
        """{'point_cloud_dim': 6, 'backbone_output_dim': 384, 'project_output_dim': 4096, 'point_token_len': 513, 'mm_use_point_start_end': True, 'projection_hidden_layer': 2, 'use_max_pool': False, 'projection_hidden_dim': [1024, 2048]}"""
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token  #<point_patch>传递给点云框架
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]
        #print("embed_tokens requires_grad:", self.get_input_embeddings().weight.requires_grad)
        #print("embed_tokens requires_grad:", self.get_output_embeddings().weight.requires_grad)
        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token
            #print("embed_tokens requires_grad:", self.get_input_embeddings().weight.requires_grad)
            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
            #print("embed_tokens requires_grad:", self.get_input_embeddings().weight.requires_grad)
            if num_new_tokens == 0:
                input_embeddings = self.get_input_embeddings().weight   #删除了data
                output_embeddings = self.get_output_embeddings().weight  #删除了data
                for p in self.get_input_embeddings().parameters():
                    #print(p)
                    #print(p.shape) #[32003, 4096] 词表大小（vocab size）= 原始 32000 + 你添加的 3 个特殊 token：<point_patch>, <point_start>, <point_end> #这是embed_tokens
                    #print("Requires grad:", p.requires_grad)
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        #print("Requires grad:", p.requires_grad)
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and all input embeddings are trainable (num_new_tokens == {num_new_tokens}).")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")          

            elif num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                #print("是否可训练:", self.get_input_embeddings().weight.requires_grad)
                for p in self.get_input_embeddings().parameters():
                    #print(p)
                    #print(p.shape) #[32003, 4096] 词表大小（vocab size）= 原始 32000 + 你添加的 3 个特殊 token：<point_patch>, <point_start>, <point_end> #这是embed_tokens
                    #print("Requires grad:", p.requires_grad)
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        #print("Requires grad:", p.requires_grad)
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
