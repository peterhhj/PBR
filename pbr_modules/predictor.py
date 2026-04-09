import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MemoryEfficientGCNLayer(nn.Module):
    """
    针对 3DGS 超大点云优化的 3D-GCN 卷积层。
    """
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, vertices, feature_map, neighbor_index, chunk_size=4000):
        N, neighbor_num = neighbor_index.shape

        # 全局基础特征计算
        feature_out = torch.matmul(feature_map, self.weights) + self.bias
        feature_center = feature_out[:, :self.out_channel]
        feature_support = feature_out[:, self.out_channel:]

        support_direction_norm = F.normalize(self.directions, dim=0)

        out_features = []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)

            v_chunk = vertices[i:end]
            idx_chunk = neighbor_index[i:end]

            neighbors_coords = vertices[idx_chunk]          
            neighbors_features = feature_support[idx_chunk] 

            neighbor_direction = neighbors_coords - v_chunk.unsqueeze(1)
            neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
            
            theta = torch.matmul(neighbor_direction_norm, support_direction_norm) 
            theta = self.relu(theta)

            activation_support = theta * neighbors_features
            activation_support = activation_support.view(-1, neighbor_num, self.support_num, self.out_channel)
            activation_support = torch.max(activation_support, dim=2)[0]
            activation_support = torch.sum(activation_support, dim=1)

            chunk_out = feature_center[i:end] + activation_support
            out_features.append(chunk_out)
            
            # 手动断开非梯度引用，帮助垃圾回收
            del neighbors_coords, neighbor_direction, neighbor_direction_norm

        return torch.cat(out_features, dim=0)


class PBRMaterialPredictor(nn.Module):
    # 【核心修改】：降低邻居数和支撑数，精简网络结构
    def __init__(self, in_channels=6, support_num=4, neighbor_num=10):
        super().__init__()
        self.neighbor_num = neighbor_num
        
        # 将原来臃肿的 3 层砍到 2 层，并降低隐藏通道数
        self.conv1 = MemoryEfficientGCNLayer(in_channels, 32, support_num=support_num)
        self.conv2 = MemoryEfficientGCNLayer(32, 64, support_num=support_num)
        
        self.head_albedo = nn.Linear(64, 3)
        self.head_roughness = nn.Linear(64, 1)
        self.head_metallic = nn.Linear(64, 1)

    def forward(self, xyz, scaling, neighbor_indices):
        # 截取前 neighbor_num 个邻居 (即使外面传了 20 个进来，我们也只用 10 个，大幅省显存)
        idx = neighbor_indices[:, :self.neighbor_num]
        
        features = torch.cat([xyz, scaling], dim=-1)

        # 减小 chunk_size 保持计算平稳
        x = self.conv1(xyz, features, idx, chunk_size=4000)
        x = F.relu(x)
        x = self.conv2(xyz, x, idx, chunk_size=4000)
        x = F.relu(x)
        
        albedo_logits = self.head_albedo(x)
        roughness_logits = self.head_roughness(x)
        metallic_logits = self.head_metallic(x)
        
        return albedo_logits, roughness_logits, metallic_logits