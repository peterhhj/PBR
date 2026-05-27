import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        N, neighbor_num = neighbor_index.shape
        chunk_size = max(int(chunk_size), 1)
        weight_center = self.weights[:, : self.out_channel]
        weight_support = self.weights[:, self.out_channel :]
        bias_center = self.bias[: self.out_channel]
        bias_support = self.bias[self.out_channel :]
        support_direction_norm = F.normalize(self.directions, dim=0)

        outputs = feature_map.new_empty((N, self.out_channel))
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            v_chunk = vertices[start:end]
            idx_chunk = neighbor_index[start:end]
            center_features = torch.matmul(feature_map[start:end], weight_center) + bias_center

            neighbors_coords = vertices[idx_chunk]
            neighbors_input = feature_map[idx_chunk]

            neighbor_direction = neighbors_coords - v_chunk.unsqueeze(1)
            neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)

            max_support_activation = None
            for support_idx in range(self.support_num):
                channel_start = support_idx * self.out_channel
                channel_end = channel_start + self.out_channel
                support_dir_chunk = support_direction_norm[:, channel_start:channel_end]
                support_weight_chunk = weight_support[:, channel_start:channel_end]
                support_bias_chunk = bias_support[channel_start:channel_end]

                theta = self.relu(torch.matmul(neighbor_direction_norm, support_dir_chunk))
                support_features = torch.matmul(neighbors_input, support_weight_chunk) + support_bias_chunk
                support_activation = theta * support_features

                if max_support_activation is None:
                    max_support_activation = support_activation
                else:
                    max_support_activation = torch.maximum(max_support_activation, support_activation)

                del support_dir_chunk
                del support_weight_chunk
                del support_bias_chunk
                del theta
                del support_features
                del support_activation

            activation_support = torch.sum(max_support_activation, dim=1)
            outputs[start:end] = center_features + activation_support

            del center_features
            del neighbors_coords
            del neighbors_input
            del neighbor_direction
            del neighbor_direction_norm
            del max_support_activation
            del activation_support

        return outputs


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
        graph_conv_chunk_size: int = 1024,
        use_checkpoint: bool = True,
    ):
        idx = neighbor_indices[:, : self.neighbor_num]

        xyz_centered = xyz - xyz.mean(dim=0, keepdim=True)
        xyz_norm = xyz_centered / xyz_centered.std(dim=0, keepdim=True).clamp(min=1e-6)
        log_scaling = torch.log(scaling.clamp(min=1e-6))
        features = torch.cat([xyz_norm, log_scaling], dim=-1)

        x = F.relu(self.conv1(xyz, features, idx, chunk_size=graph_conv_chunk_size))
        if use_checkpoint and x.requires_grad:
            conv2_out = checkpoint(
                lambda feat: self.conv2(xyz, feat, idx, chunk_size=graph_conv_chunk_size),
                x,
            )
        else:
            conv2_out = self.conv2(xyz, x, idx, chunk_size=graph_conv_chunk_size)
        x = F.relu(conv2_out)

        if style_code.ndim == 1:
            style_code = style_code.unsqueeze(0)
        style_scale, style_bias = self.style_proj(style_code).chunk(2, dim=-1)
        style_scale = style_scale.expand(x.shape[0], -1)
        style_bias = style_bias.expand(x.shape[0], -1)
        x = x * (1.0 + 0.1 * torch.tanh(style_scale)) + style_bias
        x = self.fusion(x)

        return self.head_albedo(x), self.head_roughness(x), self.head_metallic(x)
