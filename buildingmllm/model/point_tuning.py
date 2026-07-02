import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import random
import math

def stat(t, tag):
    t = t.detach()
    nz = (t != 0).float().mean().item()
    print(f"{tag}: mean={t.float().mean():.3e} std={t.float().std():.3e} "
          f"min={t.min().item():.3e} max={t.max().item():.3e} nonzero={nz:.4f}")



class LG(nn.Module):
    def __init__(self, out_dim, alpha, beta, vv, LG_dim):  #dim_expansion, type
        super().__init__() #576, 100, 1000, vv, 2
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta, vv) #3,576, 1, 1, vv, 2
        self.param_linear = True
        if LG_dim == 2:
            self.linear1 = Linear1Layer(out_dim, out_dim, bias=False)
            self.linear2 = []
            self.linear2.append(Linear2Layer(out_dim, bias=True))
            self.linear2 = nn.Sequential(*self.linear2)
        #self.reduce_dim = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=1)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1),
            nn.BatchNorm2d(384),           # 或 LayerNorm
            nn.ReLU(inplace=True)          # 激活函数有助于数值稳定
        )
    def forward(self, lc_xyz, lc_x, lc_rgb, knn_xyz, knn_x, knn_rgb): ##xyz 512,3 #lc_x 512,384 #rgb 512,3 #knn_xyz 512,32,3, knn_x 512 32 384 , knn_rgb 512 32 3
        
        # Normalize x (features) and xyz (coordinates)
        #print(knn_xyz[0, :10, 0, :])
        #print(knn_xyz.shape)
        #print("  Sample knn_x[0, :4, :4, :4]:", knn_x[0, :4, :4, :4])
        mean_x = lc_x.unsqueeze(dim=-2) #mean_x.shape = [4, 512, 1, 384]
        std_x = torch.std(knn_x - mean_x) #knn 邻域点相对中心点（mean_x）的整体标准差
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        #knn_x = lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)###############################################################临时的！！
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        #print("  Sample knn_x[0, :4, :4, :4]:", knn_x[0, :4, :4, :4])
        #print(knn_x.shape) #torch.Size([1, 512, 32, 768])
        knn_x = knn_x.permute(0, 3, 1, 2).contiguous()  #
        #print(knn_x.shape)
        #print("  Sample knn_x[0, :4, :4, :4]:", knn_x[0, :4, :4, :4])
        #print("Mean:", knn_x.mean().item())
        #print("Std :", knn_x.std().item())
        #stat(knn_x, ">> in: point_tune input <<")
        #stat(self.reduce_dim[0].weight.data, "reduce_dim.conv.weight")
        #print("Before linear1: requires_grad =", knn_x.requires_grad)
        knn_x = self.reduce_dim(knn_x)  # shape: [B, 384, G, K]
        #print("After linear1: requires_grad =", knn_x.requires_grad)

        #stat(knn_x, ">> in: point_tune input <<")
        #print(knn_x.shape)
        #print(any(p.requires_grad for p in self.reduce_dim.parameters()))

        #print("  Sample knn_x[0, :4, :4, :4]:", knn_x[0, :4, :4, :4])
        knn_x = knn_x.permute(0, 2, 3, 1).contiguous()  # shape: [B, G, K, 384]
        #print(knn_x.shape) #4 512 32 768
        #print("  Sample knn_x[0, :4, :4, :4]:", knn_x[0, :4, :4, :4])
        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2) #4，512,32,3 变成[4, 3, 512, 32]
        knn_x = knn_x.permute(0, 3, 1, 2) #4 512 32 768   [4, 768, 512, 32]
        knn_rgb = knn_rgb.permute(0, 3, 1, 2) #4，512,32,3 变成[4, 3, 512, 32]
        #print(knn_xyz.shape)
        #print(knn_x.shape)
        #print(knn_x[0, :, 0, 0])
        #print(knn_rgb.shape)
        #print("Min:", knn_x.min().item())
        #print("Max:", knn_x.max().item())
        #print("Mean:", knn_x.mean().item())
        #print("Std :", knn_x.std().item())
        if self.param_linear: #B, G, K, #输入的knn_x [4, 768, 512, 32] #第一个reshape 4, 768, 16384
            #print(knn_x.shape) #输入是输入[4, 384, 512, 32] #B, G, K 4 512 32
            #print("📌 knn_x before Linear1Layer:")
            #print("  Shape       :", knn_x.shape)
            #print("  Min         :", knn_x.min().item())
            #print("  Max         :", knn_x.max().item())
            #print("  Mean        :", knn_x.mean().item())
            #print("  Std         :", knn_x.std().item())
            #print("  Sum         :", knn_x.sum().item())

            # 你也可以查看一小部分实际值
            #print("  Sample knn_x[0, :4, :4, :4]:", knn_x[0, :4, :4, :4])

            knn_x = self.linear1(knn_x.reshape(B, -1, G*K)).reshape(B, -1, G, K) #输入[4, 384, 512, 32]
            #print("✅ linear1 training mode:", self.linear1.training)
            #print("📌 Conv1d weight mean:", self.linear1.net[0].weight.abs().mean())
            #print("📌 Conv1d bias:", self.linear1.net[0].bias)
            #print("📌 BN mean:", self.linear1.net[1].running_mean)
            #print("📌 BN var:", self.linear1.net[1].running_var)
            #print("Min:", knn_x.min().item())
            #print("Max:", knn_x.max().item())
            #print("Mean:", knn_x.mean().item())
            #print("Std :", knn_x.std().item())
            #print(knn_x[0, :, 0, 1])
            #print(knn_x.shape)
        #print("Before linear1: requires_grad =", lc_x.requires_grad)
        knn_x_w = self.geo_extract(knn_xyz, knn_x, lc_x)  #[4, 3, 512, 32] [4, 384, 512, 32]  [4, 3, 512, 32]
        #print("Before linear1: requires_grad =", lc_x.requires_grad)
        #print("  Sample knn_x[0, :4, :4, :4]:", knn_x_w[0, :4, :4, :4])
        #print(knn_x_w.shape)
        if self.param_linear:
            for layer in self.linear2:
                knn_x_w = layer(knn_x_w)
        
        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] 

        # target_dtype = next(self.out_transform.parameters()).dtype
        # # 将lc_x转换为目标dtype
        # lc_x = lc_x.to(target_dtype)

        lc_x = self.out_transform(lc_x)
        return lc_x


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2),  #* 2 #/2
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta, vv):
        super().__init__() #3,384, 1, 1, vv, 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta  #100 #1000
        self.vv = vv
        # 门控模块（使用中心点特征 lc_x）
        self.gate = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 384),
            nn.Sigmoid() #将输出压缩到 [0, 1]
        )
    def forward(self, knn_xyz, knn_x, lc_x):                              ##[4, 3, 512, 32] [4, 384, 512, 32]  
        B, _, G, K = knn_xyz.shape #B=4,G=512,K=32
        feat_dim = self.out_dim // (self.in_dim * 2) #out_dim 384 # feat_dim64
        

        feat_range = torch.arange(feat_dim).float().cuda()    #tensor([0., 1., 2., 3. 64] #是用来构造 不同频率的位置编码基 的
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim) #
        #feat_range / feat_dim = [0/64, 1/64, 2/64, ..., 63/64]≈ tensor([0.0000, 0.0156, 0.0312, ..., 0.9844])
        #torch.pow按元素幂运算   dim_embed = [100^(0/64), 100^(1/64), 100^(2/64), ..., 100^(63/64)] 大概是从1到100
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)
        #knn_xyz是[4, 3, 512, 32]  div_embed是[4, 3, 512, 32, 64]
        #torch.div是逐元素除法 #每个坐标3 被除以64次
        sin_embed = torch.sin(div_embed) #[4, 3, 512, 32, 64] 每个位置变成了正弦值
        cos_embed = torch.cos(div_embed) #每个位置变成了余弦值4, 3, 512, 32, 64]
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4) #[4, 3, 512, 32, 128]
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K) #[4, 3, 32, 128, 512] #[4, 384, 512, 32]
        #  # 变为 [4, 384, 512]
        # 门控
        #lc_x 是 [4, 512, 384]
        """gate = self.gate(lc_x)  # [B, C, G] #[4, 512, 384]  #内容是0到1
        gate = gate.permute(0, 2, 1) # 变为 [4, 384, 512]
        position_embed = position_embed * gate.unsqueeze(-1)  # [4, 384, 512, 32] #[4, 384, 512, 1]"""
        
        
        
        
        # Weigh
        knn_x_w = knn_x + position_embed
        
        return knn_x_w ## [4, 384, 512, 32]
        #return knn_x_w.to(knn_x.dtype)



class En(nn.Module):  ##input_points=1024, num_stages=3, embed_dim=288, k_neighbors=81, alpha=100, beta=1000, vv=torch.randn(1, 5000), LG_dim=[2, 2, 2]
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, vv, LG_dim):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        #self.raw_point_embed = Linear1Layer(6, self.embed_dim, bias=False) #输入1024,6 输出1024,288

        #self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LG_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim  # 初始的out_dim是384
        #group_num = self.input_points#  # 初始的group_num是1024

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            if LG_dim[i] == 2 or LG_dim[i] == 1:
                out_dim = out_dim
                #out_dim = out_dim * 2 # 576
                #group_num = group_num // 2 # 512
            #self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))# 512,81
            self.LG_list.append(LG(out_dim, self.alpha, self.beta, vv, LG_dim[i])) #576, 100, 1000, vv, 2 
            self.Pooling_list.append(Pooling(out_dim))
    
    
    
    def forward(self, xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb):
       
        # Multi-stage Hierarchy
        for i in range(self.num_stages): #xyz 512,3 #lc_x 512,384 #rgb 512,3 #knn_xyz 512,32,3, knn_x 512 32 384 , knn_rgb 512 32 3 
            knn_x_w = self.LG_list[i](xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb) ## [4, 384, 512, 32]
            x = self.Pooling_list[i](knn_x_w) # [4, 384, 512]
            lc_x = x.transpose(1, 2).contiguous() #[4, 512, 384]
        return lc_x, knn_xyz, xyz
    

class PointTUNE(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()

        self.use_max_pool = use_max_pool
        self.config = config
        self.input_points = getattr(config, "input_points", 1024)
        self.num_stages = getattr(config, "num_stages", 3) #这里也要改
        self.embed_dim = getattr(config, "embed_dim", 384) #原本是288
        self.pos_init_dim = getattr(config, "pos_init_dim", 3)
        self.k_neighbors = getattr(config, "group_size", 81)
        self.beta = getattr(config, "beta", 1000)
        self.alpha = getattr(config, "alpha", 100)
        self.LG_dim = getattr(config, "LG_dim",[2, 2, 2]) #[2, 2, 2]
        self.point_dims = getattr(config, "point_dims", 6)
        self.vv = torch.randn(1, 5000)
        self.projection_hidden_dim = getattr(config, "projection_hidden_dim", [1024, 2048])
        self.projection_hidden_layer = getattr(config, "projection_hidden_layer", 2)
        self.trans_dim = getattr(config, "trans_dim", 384)
        self.cls_dim = getattr(config, "cls_dim", 40)
        self.num_heads = getattr(config, "num_heads", 6)
        self.depth = getattr(config, "depth", 12)
        self.drop_path_rate = getattr(config, "drop_path_rate", 0.1)
        self.name = getattr(config, "NAME", "PointTUNE")
        self.encoder_dims = getattr(config, "encoder_dims", 256) 
        self.num_group = getattr(config, "num_group", 512)
        self.group_size = getattr(config, "group_size", 81)
        
        self.recon_fp = getattr(config, "recon_fp", 1) #计算的是tokens
        self.mae_fp = getattr(config, "mae_fp", 1)
        self.mask_dim = getattr(config, "mask_dim", 4096)
        self.mask_ratio = getattr(config, "mask_ratio", 0.1)  #原始是0.3   #第一阶段是0 第二阶段改成0.1  #用正则头的话是0 不用正则头的话是0.1
        self.mae_feature = getattr(config, "mae_feature", 0)
        self.recon_feature = getattr(config, "recon_feature", 0)
        self.pos_embed_mae = getattr(config, "pos_embed_mae", 0)
        self.pos_embed_dim = getattr(config, "pos_embed_dim", 4096)
        self.recon_pos = getattr(config, "recon_pos", 1)
        self.pos_embed_type = getattr(config, "pos_embed_type", 0) #0不使用位置  1使用位置编码

        
        self.norm = nn.LayerNorm(384)
        if self.LG_dim[-1] != 3: #最后一个元素不等于3所以执行本条
            self.out_dim = self.embed_dim   #*(2**self.num_stages)
            #self.out_dim = self.embed_dim*(2**self.num_stages) #self.embed_dim是288, self.num_stages是3，所以288*2^3=2304
        else:
            self.out_dim = self.embed_dim*(2**(self.num_stages-1))
            
        self.En = En(self.input_points, self.num_stages, self.embed_dim, self.k_neighbors, self.alpha, self.beta, self.vv, self.LG_dim)
        #input_points=1024, num_stages=3, embed_dim=288, k_neighbors=81, alpha=100, beta=1000, vv=torch.randn(1, 5000), LGA_dim=[2, 2, 2]
        self.class_embedding = nn.Parameter(torch.randn(1,1,384))


        #gammam函数
        self.my_luck = nn.Parameter(torch.tensor(1e-4))  # 或 1e-6~1e-3 #gammam函数 step1/2
        #self.my_luck = nn.Parameter(torch.full((384,), 1e-4))



        
    def forward(self, point_features, xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb, pos_head_type=None):

        #x = self.reduce_dim(x)
        #print(x.shape) #[4, 512, 384]
        #print(knn_xyz.shape) #[4, 512, 32, 3]
        #print(xyz.shape) #[4, 512, 3]
        
        x, knn_xyz, xyz = self.En(xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb)
        class_embed = self.class_embedding.expand(x.size(0), -1, -1).to(dtype=x.dtype)  #之前是这个
        #class_embed = torch.zeros(x.size(0), 1, x.size(2), dtype=x.dtype, device=x.device)
        feature_adapter = torch.cat([class_embed, x], dim=1)
        feature_adapter = self.norm(feature_adapter)
        feature_frozen = point_features
        #feature_plus = feature_frozen + feature_adapter
        


        #gammam函数
        #print("gamma =", self.my_luck.item()) #gammam函数 step2/2
        #print("gamma =", self.my_luck[:5])
        feature_plus = feature_frozen + self.my_luck* feature_adapter #gammam函数 step2/2
        
        
        
        feature_plus = self.norm(feature_plus)
        
        return feature_adapter, feature_frozen, feature_plus, knn_xyz, xyz  #point_features是冻住的encdoer输出   feature_final是adapter的输出


    @property
    def dtype(self):
        return self.En.raw_point_embed.net[0].weight.dtype

    @property
    def device(self):
        return self.En.raw_point_embed.net[0].weight.device