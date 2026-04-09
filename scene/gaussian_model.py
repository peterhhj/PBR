#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, build_rotation, strip_symmetric, build_scaling_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.setup_functions()
        
        # 1. 强制关闭 SH 系数
        self.active_sh_degree = 0 
        self.max_sh_degree = 0
        
        # 2. 基础几何属性
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # 3. PBR 材质属性
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        
        # 移除原版内置 optimizer
        self.optimizer = None
        self.spatial_lr_scale = 0

    # ================= 属性读取 Getter =================
    @property
    def get_scaling(self):  
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_albedo(self):
        return torch.sigmoid(self._albedo)
        
    @property
    def get_roughness(self):
        return torch.clamp(torch.sigmoid(self._roughness), min=0.04, max=1.0)
        
    @property
    def get_metallic(self):
        return torch.sigmoid(self._metallic)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # ================= 自定义保存 PBR 点云 =================
    def construct_list_of_attributes(self):
        # 重新定义保存的属性列表，剥离 SH，加入 PBR
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
            
        # 加入 PBR 属性
        l.extend(['albedo_0', 'albedo_1', 'albedo_2'])
        l.append('roughness')
        l.append('metallic')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        # 提取当前网络预测的 PBR 属性
        albedo = self._albedo.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # 将原本放 SH 的地方替换为 albedo, roughness, metallic
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation, albedo, roughness, metallic), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # ================= 从 PLY 加载并冻结几何 =================
    def load_ply(self, path):
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")], key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        
        # 彻底冻结几何拓扑
        self._xyz.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
        num_pts = xyz.shape[0]
        self._albedo = nn.Parameter(torch.zeros((num_pts, 3), dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.zeros((num_pts, 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.zeros((num_pts, 1), dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        print(f"Loaded {num_pts} frozen gaussians for PBR stylization.")