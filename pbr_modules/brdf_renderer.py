import torch
import torch.nn.functional as F
import math

def render_pbr_image(albedo, roughness, metallic, normal, view_dir, light_dir, light_color):
    """
    基于 Cook-Torrance BRDF 的延迟渲染器
    输入参数均为 (C, H, W) 形状的张量
    """
    eps = 1e-6
    
    # 规范化方向向量
    N = normal
    V = F.normalize(view_dir, p=2, dim=0)
    L = F.normalize(light_dir, p=2, dim=0)
    H = F.normalize(V + L, p=2, dim=0)
    
    # 点乘计算 (限制在 0 避免负数)
    NdotL = torch.clamp(torch.sum(N * L, dim=0, keepdim=True), min=eps, max=1.0)
    NdotV = torch.clamp(torch.sum(N * V, dim=0, keepdim=True), min=eps, max=1.0)
    NdotH = torch.clamp(torch.sum(N * H, dim=0, keepdim=True), min=eps, max=1.0)
    VdotH = torch.clamp(torch.sum(V * H, dim=0, keepdim=True), min=eps, max=1.0)
    
    # 1. 基础反射率 F0 (非金属为0.04，金属则等于基础色)
    F0 = torch.full_like(albedo, 0.04)
    F0 = torch.lerp(F0, albedo, metallic)
    
    # 2. Fresnel (Schlick Approximation)
    F_fresnel = F0 + (1.0 - F0) * torch.pow((1.0 - VdotH), 5.0)
    
    # 3. Normal Distribution Function (GGX)
    alpha = roughness ** 2
    alpha_sq = alpha ** 2
    denom = (NdotH ** 2) * (alpha_sq - 1.0) + 1.0
    D = alpha_sq / (math.pi * (denom ** 2) + eps)
    
    # 4. Geometry Function (Smith Schlick-GGX)
    k = ((roughness + 1.0) ** 2) / 8.0
    G_V = NdotV / (NdotV * (1.0 - k) + k + eps)
    G_L = NdotL / (NdotL * (1.0 - k) + k + eps)
    G = G_V * G_L
    
    # 计算高光 Specular
    specular = (D * F_fresnel * G) / (4.0 * NdotV * NdotL + eps)
    
    # 计算漫反射 Diffuse (能量守恒：非金属且未被高光反射的光)
    kD = (1.0 - F_fresnel) * (1.0 - metallic)
    diffuse = albedo / math.pi
    
    # 最终光照计算
    color = (kD * diffuse + specular) * light_color * NdotL
    
    # 添加简单的环境光以避免纯黑区域
    ambient = 0.03 * albedo
    color = color + ambient
    
    # Gamma 校正
    color = torch.pow(torch.clamp(color, min=0.0), 1.0/2.2)
    return color