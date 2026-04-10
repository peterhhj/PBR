from typing import Optional

import math

import torch
import torch.nn.functional as F

from pbr import CubemapLight, get_brdf_lut, pbr_shading


def _legacy_point_light(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: torch.Tensor,
    view_dir: torch.Tensor,
    light_dir: torch.Tensor,
    light_color: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-6
    N = F.normalize(normal, p=2, dim=0)
    V = F.normalize(view_dir, p=2, dim=0)
    L = F.normalize(light_dir, p=2, dim=0)
    H = F.normalize(V + L, p=2, dim=0)

    NdotL = torch.clamp(torch.sum(N * L, dim=0, keepdim=True), min=eps, max=1.0)
    NdotV = torch.clamp(torch.sum(N * V, dim=0, keepdim=True), min=eps, max=1.0)
    NdotH = torch.clamp(torch.sum(N * H, dim=0, keepdim=True), min=eps, max=1.0)
    VdotH = torch.clamp(torch.sum(V * H, dim=0, keepdim=True), min=eps, max=1.0)

    F0 = torch.full_like(albedo, 0.04)
    F0 = torch.lerp(F0, albedo, metallic)
    F_fresnel = F0 + (1.0 - F0) * torch.pow(1.0 - VdotH, 5.0)

    alpha = roughness**2
    alpha_sq = alpha**2
    denom = (NdotH**2) * (alpha_sq - 1.0) + 1.0
    distribution = alpha_sq / (math.pi * (denom**2) + eps)

    k = ((roughness + 1.0) ** 2) / 8.0
    G_V = NdotV / (NdotV * (1.0 - k) + k + eps)
    G_L = NdotL / (NdotL * (1.0 - k) + k + eps)
    geometry = G_V * G_L

    specular = (distribution * F_fresnel * geometry) / (4.0 * NdotV * NdotL + eps)
    diffuse = ((1.0 - F_fresnel) * (1.0 - metallic)) * albedo / math.pi
    return (diffuse + specular) * light_color * NdotL


def render_pbr_image(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: torch.Tensor,
    view_dir: Optional[torch.Tensor] = None,
    light_dir: Optional[torch.Tensor] = None,
    light_color: Optional[torch.Tensor] = None,
    light: Optional[CubemapLight] = None,
    view_dirs: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    brdf_lut: Optional[torch.Tensor] = None,
    gamma: bool = False,
):
    if light is not None:
        if view_dirs is None:
            raise ValueError("view_dirs is required when rendering with a cubemap light.")
        if mask is None:
            mask = torch.ones_like(albedo[:1]).permute(1, 2, 0).bool()
        if brdf_lut is None:
            brdf_lut = get_brdf_lut()
        return pbr_shading(
            light=light,
            normals=normal.permute(1, 2, 0),
            view_dirs=view_dirs,
            albedo=albedo.permute(1, 2, 0),
            roughness=roughness.permute(1, 2, 0),
            metallic=metallic.permute(1, 2, 0),
            mask=mask,
            brdf_lut=brdf_lut,
            gamma=gamma,
        )["render_rgb"].permute(2, 0, 1)

    if view_dir is None or light_dir is None or light_color is None:
        raise ValueError("Either provide a cubemap `light` or the legacy point-light arguments.")
    return _legacy_point_light(albedo, roughness, metallic, normal, view_dir, light_dir, light_color)
