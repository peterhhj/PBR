import json
import os
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from train_pbr import (
    MATERIAL_PRESETS,
    composite_background,
    load_style_image,
    load_train_cameras,
    prepare_source_image,
    save_tensor_image,
)
from pbr import linear_to_srgb, srgb_to_linear


PSEUDO_VIEW_PRESETS: Dict[str, Dict[str, float]] = {
    "wood": {
        "shade_min": 0.34,
        "shade_max": 0.88,
        "shade_gamma": 1.35,
        "highlight_strength": 0.16,
        "highlight_gamma": 1.1,
    },
    "jade": {
        "shade_min": 0.42,
        "shade_max": 0.96,
        "shade_gamma": 1.1,
        "highlight_strength": 0.24,
        "highlight_gamma": 0.9,
    },
    "ceramic": {
        "shade_min": 0.48,
        "shade_max": 0.98,
        "shade_gamma": 1.0,
        "highlight_strength": 0.28,
        "highlight_gamma": 0.88,
    },
    "metal": {
        "shade_min": 0.18,
        "shade_max": 0.72,
        "shade_gamma": 1.0,
        "highlight_strength": 0.42,
        "highlight_gamma": 0.8,
    },
}

STYLE_COLOR_PRESETS: Dict[str, Dict[str, float]] = {
    "wood": {
        "target_luma_min": 0.24,
        "target_luma_max": 0.58,
        "target_luma_gamma": 1.2,
        "palette_blend": 0.35,
    },
    "jade": {
        "target_luma_min": 0.38,
        "target_luma_max": 0.66,
        "target_luma_gamma": 1.0,
        "palette_blend": 0.25,
    },
    "ceramic": {
        "target_luma_min": 0.62,
        "target_luma_max": 0.92,
        "target_luma_gamma": 0.95,
        "palette_blend": 0.28,
    },
    "metal": {
        "target_luma_min": 0.18,
        "target_luma_max": 0.64,
        "target_luma_gamma": 0.9,
        "palette_blend": 0.15,
    },
}

BASECOLOR_TRANSFER_PRESETS: Dict[str, Dict[str, float]] = {
    "wood": {
        "triplanar_power": 6.0,
        "contrast": 1.18,
        "saturation": 1.12,
        "palette_mix": 0.65,
    },
    "jade": {
        "triplanar_power": 4.0,
        "contrast": 1.08,
        "saturation": 1.03,
        "palette_mix": 0.45,
    },
    "ceramic": {
        "triplanar_power": 5.0,
        "contrast": 1.05,
        "saturation": 0.92,
        "palette_mix": 0.58,
    },
    "metal": {
        "triplanar_power": 8.0,
        "contrast": 1.10,
        "saturation": 0.98,
        "palette_mix": 0.35,
    },
}


def normalize_positions(xyz: torch.Tensor) -> torch.Tensor:
    xyz_min = xyz.min(dim=0).values
    xyz_max = xyz.max(dim=0).values
    scale = (xyz_max - xyz_min).clamp(min=1e-6)
    return (xyz - xyz_min) / scale


def wrap01(coords: torch.Tensor) -> torch.Tensor:
    return coords - torch.floor(coords)


def percentile_1d(values: torch.Tensor, q: float) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.new_tensor(0.5)
    sorted_values, _ = torch.sort(flat)
    index = int(round((sorted_values.numel() - 1) * q))
    index = max(0, min(index, sorted_values.numel() - 1))
    return sorted_values[index]


def preprocess_style_texture(style_texture: torch.Tensor, material_preset: str) -> torch.Tensor:
    color_preset = STYLE_COLOR_PRESETS[material_preset]
    texture = style_texture.clamp(1e-4, 1.0)
    luma = 0.299 * texture[:, 0:1] + 0.587 * texture[:, 1:2] + 0.114 * texture[:, 2:3]

    # Remove large-scale illumination so the pseudo baseColor tracks material color rather than lighting.
    local_illum = F.avg_pool2d(luma, kernel_size=41, stride=1, padding=20)
    flattened = texture / local_illum.clamp(min=1e-3) * local_illum.mean()

    flat_luma = 0.299 * flattened[:, 0:1] + 0.587 * flattened[:, 1:2] + 0.114 * flattened[:, 2:3]
    q10 = percentile_1d(flat_luma, 0.10)
    q90 = percentile_1d(flat_luma, 0.90)
    norm_luma = ((flat_luma - q10) / (q90 - q10).clamp(min=1e-4)).clamp(0.0, 1.0)

    target_luma = color_preset["target_luma_min"] + (
        color_preset["target_luma_max"] - color_preset["target_luma_min"]
    ) * norm_luma.pow(color_preset["target_luma_gamma"])
    blend_strength = color_preset["palette_blend"]

    chroma = flattened / flat_luma.clamp(min=1e-4)
    intrinsic = (chroma * target_luma).clamp(0.02, 0.95)

    # Pull the texture back toward its robust mid-tone palette to suppress highlight pollution.
    midtone_mask = (norm_luma > 0.18) & (norm_luma < 0.82)
    if midtone_mask.any():
        palette = intrinsic.permute(0, 2, 3, 1)[midtone_mask.squeeze(1)].mean(dim=0)
        intrinsic = (1.0 - blend_strength) * intrinsic + blend_strength * palette.view(1, 3, 1, 1)

    return intrinsic.clamp(0.02, 0.95)


def compute_texture_palette(style_texture: torch.Tensor) -> Dict[str, torch.Tensor]:
    pixels = style_texture[0].permute(1, 2, 0).reshape(-1, 3)
    luma = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    q10 = percentile_1d(luma, 0.10)
    q90 = percentile_1d(luma, 0.90)
    keep = (luma >= q10) & (luma <= q90)
    if keep.any():
        pixels = pixels[keep]
    mean = pixels.mean(dim=0)
    std = pixels.std(dim=0).clamp(min=1e-4)
    return {"mean": mean, "std": std}


def recolor_basecolor(
    basecolor: torch.Tensor,
    style_palette: Dict[str, torch.Tensor],
    material_preset: str,
) -> torch.Tensor:
    preset = BASECOLOR_TRANSFER_PRESETS[material_preset]
    mean = basecolor.mean(dim=0)
    std = basecolor.std(dim=0).clamp(min=1e-4)
    matched = (basecolor - mean) / std
    matched = matched * style_palette["std"] * preset["contrast"] + style_palette["mean"]

    matched_mean = matched.mean(dim=-1, keepdim=True)
    matched = matched_mean + preset["saturation"] * (matched - matched_mean)

    palette_color = style_palette["mean"].view(1, 3)
    matched = (1.0 - preset["palette_mix"]) * matched + preset["palette_mix"] * palette_color
    return matched.clamp(0.02, 0.95)


def sample_texture(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    if texture.ndim != 4:
        raise ValueError(f"Expected style texture in [1, 3, H, W], got {texture.shape}")
    grid = uv.mul(2.0).sub(1.0).view(1, -1, 1, 2)
    sampled = F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def build_triplanar_materials(
    gaussians: GaussianModel,
    style_texture: torch.Tensor,
    material_preset: str,
    texture_scale: float,
    texture_offset: Tuple[float, float, float],
) -> Dict[str, torch.Tensor]:
    preset = MATERIAL_PRESETS[material_preset]
    transfer_preset = BASECOLOR_TRANSFER_PRESETS[material_preset]
    xyz = gaussians.get_xyz.detach()
    normals = gaussians.get_normal.detach()
    normalized_xyz = normalize_positions(xyz)
    shifted = wrap01(
        normalized_xyz * texture_scale
        + torch.tensor(texture_offset, dtype=normalized_xyz.dtype, device=normalized_xyz.device).view(1, 3)
    )

    sample_x = sample_texture(style_texture, shifted[:, [1, 2]])
    sample_y = sample_texture(style_texture, shifted[:, [0, 2]])
    sample_z = sample_texture(style_texture, shifted[:, [0, 1]])

    weights = normals.abs().pow(transfer_preset["triplanar_power"])
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    basecolor = (
        weights[:, 0:1] * sample_x
        + weights[:, 1:2] * sample_y
        + weights[:, 2:3] * sample_z
    ).clamp(0.02, 0.98)
    basecolor = recolor_basecolor(basecolor, compute_texture_palette(style_texture), material_preset)

    luminance = (0.299 * basecolor[:, 0] + 0.587 * basecolor[:, 1] + 0.114 * basecolor[:, 2]).unsqueeze(-1)
    rough_center = 0.5 * (preset["roughness_min"] + preset["roughness_max"])
    rough_span = max(preset["roughness_max"] - preset["roughness_min"], 1e-4)
    roughness = rough_center + rough_span * 0.2 * (0.55 - luminance)
    if material_preset == "wood":
        roughness = roughness.clamp(max=min(preset["roughness_max"], 0.42))
    elif material_preset == "jade":
        roughness = roughness.clamp(max=min(preset["roughness_max"], 0.22))
    elif material_preset == "ceramic":
        roughness = roughness.clamp(max=min(preset["roughness_max"], 0.20))
    roughness = roughness.clamp(preset["roughness_min"], preset["roughness_max"])

    metallic = torch.full_like(roughness, preset["metallic_target"])
    if material_preset == "metal":
        metallic = (metallic + 0.08 * (luminance - 0.5)).clamp(0.75, 0.99)
    else:
        metallic = metallic.clamp(0.0, 0.08)

    return {
        "baseColor": basecolor,
        "roughness": roughness,
        "metallic": metallic,
    }


def save_material_cache(output_dir: str, materials: Dict[str, torch.Tensor]) -> None:
    npz_path = os.path.join(output_dir, "pseudo_material_3d.npz")
    arrays = {key: value.detach().cpu().numpy().astype("float32") for key, value in materials.items()}
    import numpy as np

    np.savez(npz_path, **arrays)


def save_linear_tensor(path: str, tensor: torch.Tensor) -> None:
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().numpy().astype("float32")
    np.save(path, array)


def save_style_debug(output_dir: str, raw_srgb: torch.Tensor, processed_linear: torch.Tensor) -> None:
    save_tensor_image(raw_srgb[0], os.path.join(output_dir, "style_input_srgb.png"))
    save_tensor_image(linear_to_srgb(processed_linear[0]), os.path.join(output_dir, "style_input_linear_preview.png"))


def masked_mean(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(image).clamp(min=0.0, max=1.0)
    denom = weight.sum().clamp(min=1.0)
    return (image * weight).sum() / denom


def masked_percentile(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    masked_values = values[mask > 0.03]
    if masked_values.numel() == 0:
        return values.new_tensor(0.5)
    sorted_values, _ = torch.sort(masked_values)
    index = int(round((sorted_values.numel() - 1) * q))
    index = max(0, min(index, sorted_values.numel() - 1))
    return sorted_values[index]


def source_guided_shading(
    source_image: torch.Tensor,
    opacity: torch.Tensor,
    shade_min: float,
    shade_max: float,
    shade_gamma: float,
    highlight_gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_luma = (
        0.299 * source_image[0:1]
        + 0.587 * source_image[1:2]
        + 0.114 * source_image[2:3]
    )
    q_low = masked_percentile(source_luma, opacity, 0.1)
    q_high = masked_percentile(source_luma, opacity, 0.9)
    normalized = (source_luma - q_low) / (q_high - q_low).clamp(min=1e-4)
    normalized = normalized.clamp(0.0, 1.0).pow(shade_gamma)
    diffuse_factor = shade_min + (shade_max - shade_min) * normalized

    pooled = F.avg_pool2d(source_luma.unsqueeze(0), kernel_size=13, stride=1, padding=6)[0]
    local_peak = F.relu(source_luma - pooled - 0.02)
    edge_peak = F.relu(normalized - 0.78)
    highlight = 0.6 * local_peak / local_peak.amax().clamp(min=1e-4) + 0.4 * edge_peak / edge_peak.amax().clamp(
        min=1e-4
    )
    highlight = highlight * opacity
    return diffuse_factor, highlight.clamp(0.0, 1.0).pow(highlight_gamma)


def render_pseudo_targets(
    ply_path: str,
    source_path: str,
    style_image_path: str,
    material_preset: str,
    output_dir: str,
    texture_scale: float,
    texture_offset: Tuple[float, float, float],
    style_preprocess: str,
    shade_min: Optional[float],
    shade_max: Optional[float],
    shade_gamma: Optional[float],
    highlight_strength: Optional[float],
    highlight_gamma: Optional[float],
) -> None:
    if material_preset not in MATERIAL_PRESETS:
        raise ValueError(f"Unknown material preset: {material_preset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ("baseColor", "diffuse", "final", "mask"):
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    pseudo_preset = PSEUDO_VIEW_PRESETS[material_preset]
    shade_min = pseudo_preset["shade_min"] if shade_min is None else shade_min
    shade_max = pseudo_preset["shade_max"] if shade_max is None else shade_max
    shade_gamma = pseudo_preset["shade_gamma"] if shade_gamma is None else shade_gamma
    highlight_strength = pseudo_preset["highlight_strength"] if highlight_strength is None else highlight_strength
    highlight_gamma = pseudo_preset["highlight_gamma"] if highlight_gamma is None else highlight_gamma

    print("1. Loading frozen gaussian geometry...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    print("2. Building 3D-consistent pseudo material field from the style image...")
    style_texture_srgb = load_style_image(style_image_path, device, resolution=512)
    style_texture = srgb_to_linear(style_texture_srgb).clamp(0.0, 1.0)
    if style_preprocess == "intrinsic":
        style_texture = preprocess_style_texture(style_texture, material_preset)
    elif style_preprocess != "raw":
        raise ValueError(f"Unsupported style_preprocess mode: {style_preprocess}")
    save_style_debug(output_dir, style_texture_srgb, style_texture)
    materials = build_triplanar_materials(
        gaussians=gaussians,
        style_texture=style_texture,
        material_preset=material_preset,
        texture_scale=texture_scale,
        texture_offset=texture_offset,
    )
    save_material_cache(output_dir, materials)

    preview_ply = os.path.join(output_dir, "pseudo_material_preview.ply")
    gaussians.set_materials(materials["baseColor"], materials["roughness"], materials["metallic"])
    gaussians.save_ply(preview_ply)

    print("3. Loading COLMAP training cameras...")
    train_cameras = load_train_cameras(source_path, device)
    if not train_cameras:
        raise RuntimeError("No training cameras were found.")

    print("4. Rendering pseudo supervision targets...")

    with torch.no_grad():
        for view_cam in tqdm(train_cameras, desc="Pseudo targets"):
            render_pkg = render(
                viewpoint_camera=view_cam,
                pc=gaussians,
                bg_color=torch.zeros(3, dtype=torch.float32, device=device),
                override_basecolor=materials["baseColor"],
                override_roughness=materials["roughness"],
                override_metallic=materials["metallic"],
                derive_normal=True,
                pad_normal=True,
            )

            opacity = render_pkg["opacity_map"].clamp(0.0, 1.0)
            source_image = prepare_source_image(view_cam, device)
            source_shading, source_highlight = source_guided_shading(
                source_image,
                opacity,
                shade_min=shade_min,
                shade_max=shade_max,
                shade_gamma=shade_gamma,
                highlight_gamma=highlight_gamma,
            )
            image_key = os.path.splitext(os.path.basename(getattr(view_cam, "image_name", f"{view_cam.uid:04d}")))[0]
            base_linear = render_pkg["albedo_map"].clamp(0.0, 1.0)
            diffuse_linear = (base_linear * source_shading).clamp(0.0, 1.0)
            highlight_rgb = source_highlight.repeat(3, 1, 1) * (0.92 + 0.08 * base_linear)
            final_linear = (diffuse_linear + highlight_strength * highlight_rgb).clamp(0.0, 1.0)

            save_linear_tensor(os.path.join(output_dir, "baseColor", f"{image_key}.npy"), base_linear)
            save_linear_tensor(os.path.join(output_dir, "diffuse", f"{image_key}.npy"), diffuse_linear)
            save_linear_tensor(os.path.join(output_dir, "final", f"{image_key}.npy"), final_linear)
            save_linear_tensor(os.path.join(output_dir, "mask", f"{image_key}.npy"), opacity)

            save_tensor_image(
                composite_background(linear_to_srgb(base_linear), opacity),
                os.path.join(output_dir, "baseColor", f"{image_key}.png"),
            )
            save_tensor_image(
                composite_background(linear_to_srgb(diffuse_linear), opacity),
                os.path.join(output_dir, "diffuse", f"{image_key}.png"),
            )
            save_tensor_image(
                composite_background(linear_to_srgb(final_linear), opacity),
                os.path.join(output_dir, "final", f"{image_key}.png"),
            )
            save_tensor_image(opacity.repeat(3, 1, 1), os.path.join(output_dir, "mask", f"{image_key}.png"))

    metadata = {
        "ply_path": os.path.abspath(ply_path),
        "source": os.path.abspath(source_path),
        "style_image": os.path.abspath(style_image_path),
        "style_preprocess": style_preprocess,
        "material_preset": material_preset,
        "texture_scale": texture_scale,
        "texture_offset": list(texture_offset),
        "shade_min": shade_min,
        "shade_max": shade_max,
        "shade_gamma": shade_gamma,
        "highlight_strength": highlight_strength,
        "highlight_gamma": highlight_gamma,
        "preview_ply": os.path.abspath(preview_ply),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Finished. Pseudo targets saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--material_preset", type=str, default="wood", choices=sorted(MATERIAL_PRESETS.keys()))
    parser.add_argument("--output_dir", type=str, default="pseudo_targets")
    parser.add_argument("--texture_scale", type=float, default=3.0)
    parser.add_argument("--texture_offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--style_preprocess", type=str, default="intrinsic", choices=["intrinsic", "raw"])
    parser.add_argument("--shade_min", type=float, default=None)
    parser.add_argument("--shade_max", type=float, default=None)
    parser.add_argument("--shade_gamma", type=float, default=None)
    parser.add_argument("--highlight_strength", type=float, default=None)
    parser.add_argument("--highlight_gamma", type=float, default=None)
    args = parser.parse_args()

    render_pseudo_targets(
        ply_path=args.ply_path,
        source_path=args.source,
        style_image_path=args.style_image,
        material_preset=args.material_preset,
        output_dir=args.output_dir,
        texture_scale=args.texture_scale,
        texture_offset=tuple(args.texture_offset),
        style_preprocess=args.style_preprocess,
        shade_min=args.shade_min,
        shade_max=args.shade_max,
        shade_gamma=args.shade_gamma,
        highlight_strength=args.highlight_strength,
        highlight_gamma=args.highlight_gamma,
    )
