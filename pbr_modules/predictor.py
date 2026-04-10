import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEfficientGCNLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, support_num: int) -> None:
        super().__init__()
        self.out_channel = out_channel
        self.support_num = support_num

        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.empty(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.empty((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.empty(3, support_num * out_channel))
        self.initialize()

    def initialize(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(
        self,
        vertices: torch.Tensor,
        feature_map: torch.Tensor,
        neighbor_index: torch.Tensor,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        N, neighbor_num = neighbor_index.shape
        feature_out = torch.matmul(feature_map, self.weights) + self.bias
        feature_center = feature_out[:, : self.out_channel]
        feature_support = feature_out[:, self.out_channel :]
        support_direction_norm = F.normalize(self.directions, dim=0)

        outputs = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            v_chunk = vertices[start:end]
            idx_chunk = neighbor_index[start:end]

            neighbors_coords = vertices[idx_chunk]
            neighbors_features = feature_support[idx_chunk]

            neighbor_direction = neighbors_coords - v_chunk.unsqueeze(1)
            neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
            theta = self.relu(torch.matmul(neighbor_direction_norm, support_direction_norm))

            activation_support = theta * neighbors_features
            activation_support = activation_support.view(-1, neighbor_num, self.support_num, self.out_channel)
            activation_support = torch.max(activation_support, dim=2).values
            activation_support = torch.sum(activation_support, dim=1)
            outputs.append(feature_center[start:end] + activation_support)

        return torch.cat(outputs, dim=0)


class PBRMaterialPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        support_num: int = 4,
        neighbor_num: int = 10,
        style_dim: int = 454,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.neighbor_num = neighbor_num
        self.conv1 = MemoryEfficientGCNLayer(in_channels, 64, support_num=support_num)
        self.conv2 = MemoryEfficientGCNLayer(64, hidden_dim, support_num=support_num)

        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head_albedo = nn.Linear(hidden_dim, 3)
        self.head_roughness = nn.Linear(hidden_dim, 1)
        self.head_metallic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        xyz: torch.Tensor,
        scaling: torch.Tensor,
        neighbor_indices: torch.Tensor,
        style_code: torch.Tensor,
    ):
        idx = neighbor_indices[:, : self.neighbor_num]

        xyz_centered = xyz - xyz.mean(dim=0, keepdim=True)
        xyz_norm = xyz_centered / xyz_centered.std(dim=0, keepdim=True).clamp(min=1e-6)
        log_scaling = torch.log(scaling.clamp(min=1e-6))
        features = torch.cat([xyz_norm, log_scaling], dim=-1)

        x = F.relu(self.conv1(xyz, features, idx))
        x = F.relu(self.conv2(xyz, x, idx))

        if style_code.ndim == 1:
            style_code = style_code.unsqueeze(0)
        style_scale, style_bias = self.style_proj(style_code).chunk(2, dim=-1)
        style_scale = style_scale.expand(x.shape[0], -1)
        style_bias = style_bias.expand(x.shape[0], -1)
        x = x * (1.0 + 0.1 * torch.tanh(style_scale)) + style_bias
        x = self.fusion(x)

        return self.head_albedo(x), self.head_roughness(x), self.head_metallic(x)
