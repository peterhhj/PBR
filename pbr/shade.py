import os
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from .light import CubemapLight


def envBRDF_approx(roughness: torch.Tensor, NoV: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=roughness.device)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=roughness.device)
    c2 = torch.tensor([-1.04, 1.04], device=roughness.device)
    r = roughness * c0 + c1
    a004 = (
        torch.minimum(torch.pow(r[..., (0,)], 2), torch.exp2(-9.28 * NoV)) * r[..., (0,)] + r[..., (1,)]
    )
    return (a004 * c2 + r[..., 2:]).clamp(min=0.0, max=1.0)


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=1e-4, max=1.0)


def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    return rgb.clamp(min=0.0, max=1.0)


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    eps = np.finfo(np.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)


def get_brdf_lut(search_root: Optional[str] = None) -> Optional[torch.Tensor]:
    candidate_paths = [os.path.join(os.path.dirname(__file__), "brdf_256_256.bin")]
    if search_root is not None:
        candidate_paths.append(os.path.join(search_root, "brdf_256_256.bin"))

    for path in candidate_paths:
        if os.path.exists(path):
            return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, 256, 256, 2))
    return None


def pbr_shading(
    light: CubemapLight,
    normals: torch.Tensor,
    view_dirs: torch.Tensor,
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    mask: torch.Tensor,
    tone: bool = False,
    gamma: bool = False,
    metallic: Optional[torch.Tensor] = None,
    brdf_lut: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for PBR shading.") from exc

    H, W, _ = normals.shape
    background = torch.zeros_like(normals) if background is None else background

    normals = normals.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)
    metallic = None if metallic is None else metallic.reshape(1, H, W, 1)

    ref_dirs = 2.0 * (normals * view_dirs).sum(dim=-1, keepdim=True).clamp(min=0.0) * normals - view_dirs

    diffuse_light = dr.texture(
        light.diffuse[None, ...], normals.contiguous(), filter_mode="linear", boundary_mode="cube"
    )
    diffuse_rgb = diffuse_light * albedo

    NoV = saturate_dot(normals, view_dirs)
    if brdf_lut is not None:
        fg_uv = torch.cat((NoV, roughness), dim=-1)
        fg_lookup = dr.texture(
            brdf_lut.to(normals.device), fg_uv.contiguous(), filter_mode="linear", boundary_mode="clamp"
        )
    else:
        fg_lookup = envBRDF_approx(roughness, NoV)

    miplevel = light.get_mip(roughness)
    specular_env = dr.texture(
        light.specular[0][None, ...],
        ref_dirs.contiguous(),
        mip=[mip[None, ...] for mip in light.specular[1:]],
        mip_level_bias=miplevel[..., 0],
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )

    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
    reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
    specular_rgb = specular_env * reflectance

    render_rgb = diffuse_rgb + specular_rgb
    render_rgb = render_rgb.squeeze(0)
    diffuse_rgb = diffuse_rgb.squeeze(0)
    specular_rgb = specular_rgb.squeeze(0)

    if tone:
        render_rgb = aces_film(render_rgb)
    if gamma:
        render_rgb = linear_to_srgb(render_rgb)

    render_rgb = torch.where(mask, render_rgb, background)
    diffuse_rgb = torch.where(mask, diffuse_rgb, background)
    specular_rgb = torch.where(mask, specular_rgb, torch.zeros_like(background))

    return {
        "render_rgb": render_rgb,
        "diffuse_rgb": diffuse_rgb,
        "specular_rgb": specular_rgb,
        "diffuse_light": diffuse_light.squeeze(0),
    }
