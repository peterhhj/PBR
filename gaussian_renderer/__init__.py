import math
import inspect
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def _median_blur_3x3(depth_map: torch.Tensor) -> torch.Tensor:
    if depth_map.ndim != 3:
        raise ValueError(f"Expected [1, H, W] depth map, got {depth_map.shape}")
    padded = F.pad(depth_map.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    unfolded = padded.unfold(2, 3, 1).unfold(3, 3, 1).contiguous()
    unfolded = unfolded.view(1, 1, depth_map.shape[1], depth_map.shape[2], 9)
    blurred = unfolded.median(dim=-1).values
    return blurred.squeeze(0)


def _depth_to_normal(depth_map: torch.Tensor, viewpoint_camera) -> torch.Tensor:
    H, W = depth_map.shape[1:]
    device = depth_map.device
    depth = _median_blur_3x3(depth_map)[0]

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_x = W / (2.0 * tanfovx)
    focal_y = H / (2.0 * tanfovy)
    cen_x = W / 2.0
    cen_y = H / 2.0

    x, y = torch.meshgrid(
        torch.arange(W, device=device, dtype=depth.dtype),
        torch.arange(H, device=device, dtype=depth.dtype),
        indexing="xy",
    )

    px = (x - cen_x + 0.5) / focal_x * depth
    py = (y - cen_y + 0.5) / focal_y * depth
    pts = torch.stack((px, py, depth), dim=-1)

    dx = pts[1:-1, 2:, :] - pts[1:-1, :-2, :]
    dy = pts[2:, 1:-1, :] - pts[:-2, 1:-1, :]
    normal = torch.cross(dx, dy, dim=-1)
    normal = F.normalize(normal, p=2, dim=-1)

    out = torch.zeros((H, W, 3), dtype=depth.dtype, device=device)
    out[1:-1, 1:-1] = normal

    valid = depth > 0
    valid_inner = valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:] & valid[:-2, 1:-1] & valid[2:, 1:-1]
    out[1:-1, 1:-1] = out[1:-1, 1:-1] * valid_inner.unsqueeze(-1)
    return out.permute(2, 0, 1)


def render(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    override_basecolor: Optional[torch.Tensor] = None,
    override_roughness: Optional[torch.Tensor] = None,
    override_metallic: Optional[torch.Tensor] = None,
    inference: bool = False,
    pad_normal: bool = True,
    derive_normal: bool = True,
) -> Dict[str, torch.Tensor]:
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device=pc.get_xyz.device
    )

    settings_kwargs = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": math.tan(viewpoint_camera.FoVx * 0.5),
        "tanfovy": math.tan(viewpoint_camera.FoVy * 0.5),
        "bg": bg_color,
        "scale_modifier": scaling_modifier,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "debug": False,
        "inference": inference,
        "argmax_depth": False,
        "antialiasing": False,
    }
    supported_setting_fields = getattr(GaussianRasterizationSettings, "_fields", None)
    if supported_setting_fields is not None:
        settings_kwargs = {
            key: value for key, value in settings_kwargs.items() if key in supported_setting_fields
        }
    raster_settings = GaussianRasterizationSettings(**settings_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    normal = pc.get_normal
    albedo = override_basecolor if override_basecolor is not None else pc.get_basecolor
    roughness = override_roughness if override_roughness is not None else pc.get_roughness
    metallic = override_metallic if override_metallic is not None else pc.get_metallic
    shs = None if override_color is not None else pc.get_features
    colors_precomp = override_color
    scales = pc.get_scaling
    rotations = pc.get_rotation

    rasterizer_kwargs = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacity,
        "normal": normal,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "albedo": albedo,
        "roughness": roughness,
        "metallic": metallic,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": None,
        "derive_normal": derive_normal,
    }
    try:
        rasterizer_sig = inspect.signature(rasterizer.forward)
        supported_rasterizer_fields = set(rasterizer_sig.parameters.keys())
        rasterizer_kwargs = {
            key: value for key, value in rasterizer_kwargs.items() if key in supported_rasterizer_fields
        }
    except (TypeError, ValueError):
        pass

    raster_outputs = rasterizer(**rasterizer_kwargs)
    if not isinstance(raster_outputs, tuple):
        raise RuntimeError("Unexpected rasterizer output type; expected a tuple of G-buffer tensors.")
    if len(raster_outputs) not in (8, 9):
        raise RuntimeError(
            "Current diff_gaussian_rasterization build returned "
            f"{len(raster_outputs)} outputs, but this PBR pipeline expects 8 or 9 "
            "(with optional normal_from_depth). "
            "Please use the GS-IR-compatible rasterizer build."
        )

    if len(raster_outputs) == 9:
        (
            rendered_image,
            radii,
            opacity_map,
            depth_map,
            normal_map_from_depth,
            normal_map,
            albedo_map,
            roughness_map,
            metallic_map,
        ) = raster_outputs
    else:
        (
            rendered_image,
            radii,
            opacity_map,
            depth_map,
            normal_map,
            albedo_map,
            roughness_map,
            metallic_map,
        ) = raster_outputs
        normal_map_from_depth = (
            _depth_to_normal(depth_map, viewpoint_camera) if derive_normal else torch.zeros_like(normal_map)
        )
    normal_map = torch.where(
        torch.norm(normal_map, dim=0, keepdim=True) > 0,
        F.normalize(normal_map, p=2, dim=0),
        normal_map,
    )
    normal_map_from_depth = torch.where(
        torch.norm(normal_map_from_depth, dim=0, keepdim=True) > 0,
        F.normalize(normal_map_from_depth, p=2, dim=0),
        normal_map_from_depth,
    )

    dot = (normal_map * normal_map_from_depth).sum(dim=0, keepdim=True)
    normal_map_from_depth = torch.where(dot < 0, -normal_map_from_depth, normal_map_from_depth)

    normal_mask = (normal_map != 0).all(dim=0, keepdim=True)
    normal_from_depth_mask = (normal_map_from_depth != 0).all(dim=0, keepdim=True)

    if pad_normal:
        opacity_mask = opacity_map.clamp(min=0.0, max=1.0)
        normal_bg = torch.tensor([0.0, 0.0, 1.0], dtype=normal_map.dtype, device=normal_map.device)
        normal_map = normal_map * opacity_mask + (1.0 - opacity_mask) * normal_bg[:, None, None]
        mask_from_depth = (normal_map_from_depth == 0.0).all(dim=0, keepdim=True).float()
        normal_map_from_depth = normal_map_from_depth * (1.0 - mask_from_depth) + mask_from_depth * normal_bg[:, None, None]

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity_map": opacity_map,
        "depth_map": depth_map,
        "normal_map_from_depth": normal_map_from_depth,
        "normal_from_depth_mask": normal_from_depth_mask,
        "normal_map": normal_map,
        "normal_mask": normal_mask,
        "albedo_map": albedo_map,
        "roughness_map": roughness_map,
        "metallic_map": metallic_map,
    }
