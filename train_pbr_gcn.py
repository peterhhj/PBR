import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Sequence, Tuple
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from pbr_modules.predictor import PBRMaterialPredictor
from pbr_modules.style_loss import VGGStyleLoss
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos


MATERIAL_PRESETS: Dict[str, Dict[str, float]] = {
    "jade": {
        "metallic_target": 0.02,
        "metallic_weight": 0.05,
        "roughness_min": 0.08,
        "roughness_max": 0.35,
        "roughness_weight": 0.08,
        "saturation_max": 0.88,
    },
    "wood": {
        "metallic_target": 0.01,
        "metallic_weight": 0.05,
        "roughness_min": 0.22,
        "roughness_max": 0.7,
        "roughness_weight": 0.08,
        "saturation_max": 0.92,
    },
    "ceramic": {
        "metallic_target": 0.0,
        "metallic_weight": 0.08,
        "roughness_min": 0.18,
        "roughness_max": 0.42,
        "roughness_weight": 0.10,
        "saturation_max": 0.62,
    },
    "metal": {
        "metallic_target": 0.92,
        "metallic_weight": 0.08,
        "roughness_min": 0.04,
        "roughness_max": 0.45,
        "roughness_weight": 0.08,
        "saturation_max": 0.98,
    },
}


def safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(x.clamp(min=eps, max=1.0 - eps))


def compute_knn_cpu(xyz_tensor: torch.Tensor, k: int = 21, chunk_size: int = 2000) -> torch.Tensor:
    num_points = xyz_tensor.shape[0]
    neighbor_indices = torch.zeros((num_points, k - 1), dtype=torch.long, device="cpu")
    print(f"Building CPU KNN graph for {num_points} gaussians...")
    for start in tqdm(range(0, num_points, chunk_size), desc="KNN"):
        end = min(start + chunk_size, num_points)
        dist = torch.cdist(xyz_tensor[start:end], xyz_tensor)
        _, indices = torch.topk(dist, k=k, dim=1, largest=False)
        neighbor_indices[start:end] = indices[:, 1:]
    return neighbor_indices


def expand_subgraph_indices(
    seed_indices: torch.Tensor,
    global_neighbor_indices: torch.Tensor,
    total_nodes: int,
    target_size: int,
    hops: int,
) -> torch.Tensor:
    device = global_neighbor_indices.device
    selected_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    frontier = seed_indices.unique()
    selected_mask[frontier] = True

    for _ in range(max(int(hops), 0)):
        if frontier.numel() == 0:
            break
        if int(selected_mask.sum().item()) >= target_size:
            break

        candidate_neighbors = global_neighbor_indices[frontier].reshape(-1).unique()
        if candidate_neighbors.numel() == 0:
            break
        candidate_neighbors = candidate_neighbors[~selected_mask[candidate_neighbors]]
        if candidate_neighbors.numel() == 0:
            break

        remaining = target_size - int(selected_mask.sum().item())
        if candidate_neighbors.numel() > remaining:
            perm = torch.randperm(candidate_neighbors.numel(), device=device)[:remaining]
            candidate_neighbors = candidate_neighbors[perm]

        selected_mask[candidate_neighbors] = True
        frontier = candidate_neighbors

    current_size = int(selected_mask.sum().item())
    if current_size < target_size:
        remaining_candidates = (~selected_mask).nonzero(as_tuple=False).flatten()
        if remaining_candidates.numel() > 0:
            add_count = min(target_size - current_size, remaining_candidates.numel())
            perm = torch.randperm(remaining_candidates.numel(), device=device)[:add_count]
            selected_mask[remaining_candidates[perm]] = True

    return selected_mask.nonzero(as_tuple=False).flatten()


def build_subgraph_neighbor_indices(
    global_neighbor_indices: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    device = global_neighbor_indices.device
    total_nodes = global_neighbor_indices.shape[0]
    local_mapping = torch.full((total_nodes,), -1, dtype=torch.long, device=device)
    local_mapping[selected_indices] = torch.arange(selected_indices.shape[0], device=device)

    local_neighbors = local_mapping[global_neighbor_indices[selected_indices]]
    self_indices = torch.arange(selected_indices.shape[0], device=device).unsqueeze(1).expand_as(local_neighbors)
    local_neighbors = torch.where(local_neighbors >= 0, local_neighbors, self_indices)
    return local_neighbors


def load_style_image(image_path: str, device: torch.device, resolution: int = 512) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    crop_size = min(image.size)
    left = (image.width - crop_size) // 2
    top = (image.height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size)).resize((resolution, resolution))
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def resize_for_vgg(image: torch.Tensor, size: int = 512) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)


def save_tensor_image(image: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.ndim == 4:
        image = image[0]
    image = image.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def get_tv_loss(gt_image: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True))
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True))
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)
    return (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()


def get_masked_tv_loss(mask: torch.Tensor, gt_image: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True))
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True))
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]
    return (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()


def get_canonical_rays(camera) -> torch.Tensor:
    H, W = camera.image_height, camera.image_width
    cen_x = W / 2.0
    cen_y = H / 2.0
    focal_x = W / (2.0 * np.tan(camera.FoVx * 0.5))
    focal_y = H / (2.0 * np.tan(camera.FoVy * 0.5))
    x, y = torch.meshgrid(
        torch.arange(W, device=camera.camera_center.device),
        torch.arange(H, device=camera.camera_center.device),
        indexing="xy",
    )
    camera_dirs = F.pad(
        torch.stack([(x.flatten() - cen_x + 0.5) / focal_x, (y.flatten() - cen_y + 0.5) / focal_y], dim=-1),
        (0, 1),
        value=1.0,
    )
    return camera_dirs


def get_view_dirs(camera) -> torch.Tensor:
    canonical_rays = get_canonical_rays(camera)
    c2w = torch.inverse(camera.world_view_transform.T)
    return -(
        (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3]).sum(dim=-1).reshape(
            camera.image_height, camera.image_width, 3
        )
    )


def composite_background(image: torch.Tensor, opacity: torch.Tensor, bg_value: float = 1.0) -> torch.Tensor:
    bg = torch.full_like(image, bg_value)
    return image * opacity + bg * (1.0 - opacity)


def grayscale_triplet(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0, keepdim=True)
    return gray.repeat(3, 1, 1)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.ndim == 4:
        prediction = prediction[0]
    if target.ndim == 4:
        target = target[0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(target.unsqueeze(0), size=prediction.shape[-2:], mode="bilinear", align_corners=False)[0]
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(0), size=prediction.shape[-2:], mode="bilinear", align_corners=False)[0]
    weight = mask.expand_as(prediction).clamp(min=0.0, max=1.0)
    denom = weight.sum().clamp(min=1.0)
    return (weight * (prediction - target).abs()).sum() / denom


def compute_rgb_stats(image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if image.ndim == 4:
        image = image[0]
    pixels = image.permute(1, 2, 0).reshape(-1, image.shape[0])
    if mask is None:
        weights = torch.ones((pixels.shape[0], 1), device=image.device, dtype=image.dtype)
    else:
        if mask.ndim == 3:
            mask = mask[0]
        weights = mask.reshape(-1, 1).to(image.dtype)
    weight_sum = weights.sum().clamp(min=1.0)
    mean = (pixels * weights).sum(dim=0) / weight_sum
    var = ((pixels - mean.unsqueeze(0)) ** 2 * weights).sum(dim=0) / weight_sum
    std = torch.sqrt(var.clamp(min=1e-6))
    return {"mean": mean, "std": std}


def rgb_distribution_loss(
    image: torch.Tensor,
    target_stats: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    current_stats = compute_rgb_stats(image, mask)
    return F.l1_loss(current_stats["mean"], target_stats["mean"]) + F.l1_loss(
        current_stats["std"], target_stats["std"]
    )


def crop_to_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding: int = 16,
    fill_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    binary_mask = mask[0] > 0.03
    nonzero = torch.nonzero(binary_mask)
    if nonzero.numel() == 0:
        return image, mask

    y_min = max(int(nonzero[:, 0].min().item()) - padding, 0)
    y_max = min(int(nonzero[:, 0].max().item()) + padding + 1, image.shape[-2])
    x_min = max(int(nonzero[:, 1].min().item()) - padding, 0)
    x_max = min(int(nonzero[:, 1].max().item()) + padding + 1, image.shape[-1])

    cropped_image = image[:, y_min:y_max, x_min:x_max]
    cropped_mask = mask[:, y_min:y_max, x_min:x_max]
    if fill_color is not None:
        fill_color = fill_color.to(image.device, image.dtype).view(-1, 1, 1)
        cropped_image = cropped_image * cropped_mask + fill_color * (1.0 - cropped_mask)
    return cropped_image, cropped_mask


def load_image_tensor(path: str, device: Optional[torch.device] = None, mode: str = "RGB") -> Optional[torch.Tensor]:
    if not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        array = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor if device is None else tensor.to(device)
    image = Image.open(path).convert(mode)
    array = np.asarray(image).astype(np.float32) / 255.0
    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor if device is None else tensor.to(device)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor if device is None else tensor.to(device)


def load_pseudo_view_targets(
    pseudo_target_dir: Optional[str],
    image_name: str,
    device: torch.device,
    cache: Dict[str, Dict[str, Optional[torch.Tensor]]],
) -> Optional[Dict[str, Optional[torch.Tensor]]]:
    if pseudo_target_dir is None:
        return None
    if image_name in cache:
        cached_targets = cache[image_name]
        if cached_targets is None:
            return None
        return {
            key: None if value is None else value.to(device=device, non_blocking=True)
            for key, value in cached_targets.items()
        }

    stem = os.path.splitext(os.path.basename(image_name))[0]
    alt_stem = image_name if os.path.splitext(image_name)[1] == "" else os.path.splitext(image_name)[0]

    def resolve_image(subdir: str, extensions: Sequence[str]) -> Optional[str]:
        candidates = [
            *[os.path.join(pseudo_target_dir, subdir, f"{stem}.{ext}") for ext in extensions],
            *[os.path.join(pseudo_target_dir, subdir, f"{alt_stem}.{ext}") for ext in extensions],
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    targets = {
        "baseColor": load_image_tensor(resolve_image("baseColor", ("npy", "png")) or "", device=None),
        "final": load_image_tensor(resolve_image("final", ("npy", "png")) or "", device=None),
        "diffuse": load_image_tensor(resolve_image("diffuse", ("npy", "png")) or "", device=None),
        "mask": load_image_tensor(resolve_image("mask", ("npy", "png")) or "", device=None, mode="L"),
    }
    if all(value is None for value in targets.values()):
        cache[image_name] = None
    else:
        cache[image_name] = targets
    cached_targets = cache[image_name]
    if cached_targets is None:
        return None
    return {
        key: None if value is None else value.to(device=device, non_blocking=True)
        for key, value in cached_targets.items()
    }


def load_pseudo_material_cache(
    pseudo_target_dir: Optional[str],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    if pseudo_target_dir is None:
        return None
    material_path = os.path.join(pseudo_target_dir, "pseudo_material_3d.npz")
    if not os.path.exists(material_path):
        return None

    data = np.load(material_path)
    return {
        "baseColor": torch.from_numpy(data["baseColor"]).to(device=device, dtype=torch.float32),
        "roughness": torch.from_numpy(data["roughness"]).to(device=device, dtype=torch.float32),
        "metallic": torch.from_numpy(data["metallic"]).to(device=device, dtype=torch.float32),
    }


def knn_material_smoothness(
    basecolor: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    neighbor_base = basecolor[neighbor_indices]
    neighbor_rough = roughness[neighbor_indices]
    neighbor_metal = metallic[neighbor_indices]
    center_base = basecolor.unsqueeze(1).expand_as(neighbor_base)
    center_rough = roughness.unsqueeze(1).expand_as(neighbor_rough)
    center_metal = metallic.unsqueeze(1).expand_as(neighbor_metal)
    loss = F.smooth_l1_loss(center_base, neighbor_base)
    loss = loss + F.smooth_l1_loss(center_rough, neighbor_rough)
    loss = loss + F.smooth_l1_loss(center_metal, neighbor_metal)
    return loss


def material_prior_loss(
    basecolor: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    preset_name: str,
) -> Dict[str, torch.Tensor]:
    preset = MATERIAL_PRESETS[preset_name]
    metallic_loss = F.mse_loss(metallic, torch.full_like(metallic, preset["metallic_target"]))
    rough_low = F.relu(preset["roughness_min"] - roughness).mean()
    rough_high = F.relu(roughness - preset["roughness_max"]).mean()
    roughness_loss = rough_low + rough_high

    brightness = basecolor.mean(dim=-1)
    saturation = basecolor.max(dim=-1).values - basecolor.min(dim=-1).values
    brightness_loss = F.relu(0.05 - brightness).mean() + F.relu(brightness - 0.95).mean()
    saturation_loss = F.relu(saturation - preset["saturation_max"]).mean()

    return {
        "metallic": metallic_loss * preset["metallic_weight"],
        "roughness": roughness_loss * preset["roughness_weight"],
        "basecolor": 0.05 * brightness_loss + 0.05 * saturation_loss,
    }


def get_env_lights(env_mode: str, device: torch.device) -> Dict[str, CubemapLight]:
    if env_mode == "default":
        preset_names = ["neutral", "studio", "rim", "warm"]
    else:
        preset_names = ["neutral", env_mode] if env_mode != "neutral" else ["neutral"]

    lights: Dict[str, CubemapLight] = {}
    for name in preset_names:
        light = CubemapLight.from_preset(name, resolution=128, device=device)
        light.build_mips()
        lights[name] = light
    return lights


def prepare_source_image(view_cam, device: torch.device) -> torch.Tensor:
    gt_image = view_cam.original_image.to(device)
    alpha_mask = getattr(view_cam, "gt_alpha_mask", None)
    if alpha_mask is None:
        return gt_image
    return gt_image * alpha_mask + (1.0 - alpha_mask)


def load_train_cameras(source_path: str, device: torch.device):
    is_synthetic = os.path.exists(os.path.join(source_path, "transforms_train.json"))
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", eval=False, train_test_exp=False, depths="")
    elif is_synthetic:
        scene_info = sceneLoadTypeCallbacks["Blender"](source_path, True, "", False)
    else:
        raise RuntimeError(f"Could not infer scene type from {source_path}")

    cam_args = Namespace(resolution=1, data_device=str(device), train_test_exp=False)
    train_cameras = cameraList_from_camInfos(
        scene_info.train_cameras,
        1.0,
        cam_args,
        is_nerf_synthetic=is_synthetic,
        is_test_dataset=False,
    )
    return train_cameras


def export_debug_bundle(
    debug_dir: str,
    iteration: int,
    render_pkg: Dict[str, torch.Tensor],
    neutral_result: Dict[str, torch.Tensor],
) -> None:
    prefix = os.path.join(debug_dir, f"iter_{iteration:05d}")
    save_tensor_image(composite_background(render_pkg["albedo_map"], render_pkg["opacity_map"]), prefix + "_baseColor.png")
    save_tensor_image(render_pkg["roughness_map"].repeat(3, 1, 1), prefix + "_roughness.png")
    save_tensor_image(render_pkg["metallic_map"].repeat(3, 1, 1), prefix + "_metallic.png")
    save_tensor_image((render_pkg["normal_map"] + 1.0) * 0.5, prefix + "_normal.png")
    save_tensor_image(render_pkg["opacity_map"].repeat(3, 1, 1), prefix + "_opacity.png")
    save_tensor_image(
        composite_background(neutral_result["diffuse_rgb"].permute(2, 0, 1), render_pkg["opacity_map"]),
        prefix + "_diffuse.png",
    )
    save_tensor_image(
        composite_background(neutral_result["specular_rgb"].permute(2, 0, 1), render_pkg["opacity_map"], bg_value=0.0),
        prefix + "_specular.png",
    )
    save_tensor_image(
        composite_background(neutral_result["render_rgb"].permute(2, 0, 1), render_pkg["opacity_map"]),
        prefix + "_final.png",
    )


def train_pbr_stylization(
    ply_path: str,
    source_path: str,
    style_image_path: str,
    iterations: int = 3000,
    material_preset: str = "jade",
    warmup_iters: int = 800,
    views_per_iter: int = 1,
    perceptual_views_per_iter: int = 1,
    env_mode: str = "default",
    anchor_weight: float = 0.15,
    style_resolution: int = 256,
    max_envs_per_iter: int = 1,
    style_crop_padding: int = 20,
    color_distribution_weight: float = 0.25,
    pseudo_target_dir: Optional[str] = None,
    pseudo_image_weight: float = 0.35,
    pseudo_base_weight: Optional[float] = None,
    pseudo_diffuse_weight: Optional[float] = None,
    pseudo_final_weight: Optional[float] = None,
    pseudo_material_weight: float = 0.2,
    graph_conv_chunk_size: int = 1024,
    gcn_subgraph_size: int = 4096,
    gcn_seed_size: int = 1024,
    gcn_expand_hops: int = 1,
    predictor_checkpoint: bool = True,
    material_optimization_mode: str = "auto",
    direct_mode_gaussian_threshold: int = 120000,
    direct_material_lr: float = 5e-2,
    output_dir: str = "pbr_outputs",
) -> str:
    if material_preset not in MATERIAL_PRESETS:
        raise ValueError(f"Unknown material preset: {material_preset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("1. Loading scene cameras...")
    train_cameras = load_train_cameras(source_path, device)
    if not train_cameras:
        raise RuntimeError("No training cameras were found.")

    print("2. Loading frozen gaussian geometry...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    xyz_cpu = gaussians.get_xyz.detach().cpu()
    neighbor_indices = compute_knn_cpu(xyz_cpu, k=17, chunk_size=2000).to(device)

    print("3. Preparing style encoder and material predictor...")
    style_target = load_style_image(style_image_path, device, resolution=style_resolution)
    style_encoder = VGGStyleLoss(device=str(device))
    with torch.no_grad():
        style_code = style_encoder.encode_style(style_target).detach()
        style_target_cache = style_encoder.build_style_cache(style_target)
        style_target_stats = compute_rgb_stats(style_target[0])
        style_fill_color = style_target[0].mean(dim=(-1, -2))

    pseudo_view_cache: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
    pseudo_material_cache = load_pseudo_material_cache(pseudo_target_dir, device)
    if pseudo_base_weight is None:
        pseudo_base_weight = pseudo_image_weight
    if pseudo_diffuse_weight is None:
        pseudo_diffuse_weight = pseudo_image_weight
    if pseudo_final_weight is None:
        pseudo_final_weight = pseudo_image_weight
    num_gaussians = gaussians.get_xyz.shape[0]
    if material_optimization_mode == "auto":
        material_optimization_mode = "direct" if num_gaussians > direct_mode_gaussian_threshold else "gcn"
    if material_optimization_mode not in {"gcn", "direct"}:
        raise ValueError(f"Unknown material_optimization_mode: {material_optimization_mode}")

    predictor = None
    direct_base_logits = None
    direct_rough_logits = None
    direct_metal_logits = None
    if material_optimization_mode == "gcn":
        predictor = PBRMaterialPredictor(
            in_channels=6,
            support_num=4,
            neighbor_num=min(16, neighbor_indices.shape[1]),
            style_dim=style_code.numel(),
            hidden_dim=128,
        ).to(device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
        material_base_cache = (
            pseudo_material_cache["baseColor"].detach().clone()
            if pseudo_material_cache is not None
            else gaussians.source_basecolor.detach().clone()
        )
        material_rough_cache = (
            pseudo_material_cache["roughness"].detach().clone()
            if pseudo_material_cache is not None
            else gaussians.source_roughness.detach().clone()
        )
        material_metal_cache = (
            pseudo_material_cache["metallic"].detach().clone()
            if pseudo_material_cache is not None
            else gaussians.source_metallic.detach().clone()
        )
        subgraph_queue = torch.randperm(num_gaussians, device=device)
        subgraph_ptr = 0
    else:
        init_base = gaussians.source_basecolor
        init_rough = gaussians.source_roughness
        init_metal = gaussians.source_metallic
        if pseudo_material_cache is not None:
            init_base = pseudo_material_cache["baseColor"]
            init_rough = pseudo_material_cache["roughness"]
            init_metal = pseudo_material_cache["metallic"]
        direct_base_logits = nn.Parameter(safe_logit(init_base.detach().clone()))
        direct_rough_logits = nn.Parameter(safe_logit(init_rough.detach().clone()))
        direct_metal_logits = nn.Parameter(safe_logit(init_metal.detach().clone()))
        optimizer = torch.optim.Adam(
            [direct_base_logits, direct_rough_logits, direct_metal_logits],
            lr=direct_material_lr,
        )
    brdf_lut = get_brdf_lut()
    lights = get_env_lights(env_mode, device)
    neutral_light = lights["neutral"]
    loss_jsonl_path = os.path.join(output_dir, "loss_log.jsonl")
    loss_csv_path = os.path.join(output_dir, "loss_log.csv")
    if os.path.exists(loss_jsonl_path):
        os.remove(loss_jsonl_path)
    with open(loss_csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "iteration",
                "Tot",
                "Sty",
                "Anchor",
                "Clr",
                "Pseudo",
                "PseudoBase",
                "PseudoDiffuse",
                "PseudoFinal",
                "PseudoMat",
                "TV",
                "Rgh",
                "Met",
            ],
        )
        writer.writeheader()

    progress_bar = tqdm(range(1, iterations + 1), desc="PBR Training")
    for iteration in progress_bar:
        if predictor is not None:
            predictor.train()
        optimizer.zero_grad(set_to_none=True)

        subgraph_indices = None
        pred_sub_basecolor = None
        pred_sub_roughness = None
        pred_sub_metallic = None
        if material_optimization_mode == "gcn":
            use_subgraph = 0 < gcn_subgraph_size < num_gaussians
            if use_subgraph:
                target_size = min(max(int(gcn_subgraph_size), 1), num_gaussians)
                seed_size = min(max(int(gcn_seed_size), 1), target_size)
                if subgraph_ptr + seed_size <= num_gaussians:
                    seed_indices = subgraph_queue[subgraph_ptr : subgraph_ptr + seed_size]
                    subgraph_ptr += seed_size
                else:
                    remaining = subgraph_queue[subgraph_ptr:]
                    refill_count = seed_size - remaining.numel()
                    subgraph_queue = torch.randperm(num_gaussians, device=device)
                    refill = subgraph_queue[:refill_count]
                    seed_indices = torch.cat([remaining, refill], dim=0)
                    subgraph_ptr = refill_count

                subgraph_indices = expand_subgraph_indices(
                    seed_indices=seed_indices,
                    global_neighbor_indices=neighbor_indices,
                    total_nodes=num_gaussians,
                    target_size=target_size,
                    hops=gcn_expand_hops,
                )
                subgraph_neighbor_indices = build_subgraph_neighbor_indices(neighbor_indices, subgraph_indices)
                base_logits, rough_logits, metal_logits = predictor(
                    gaussians.get_xyz[subgraph_indices],
                    gaussians.get_scaling[subgraph_indices],
                    subgraph_neighbor_indices,
                    style_code,
                    graph_conv_chunk_size=graph_conv_chunk_size,
                    use_checkpoint=predictor_checkpoint,
                )
                pred_sub_basecolor = torch.sigmoid(base_logits).clamp(0.02, 0.98)
                pred_sub_roughness = torch.sigmoid(rough_logits).clamp(0.04, 0.98)
                pred_sub_metallic = torch.sigmoid(metal_logits).clamp(0.0, 0.999)

                pred_basecolor = material_base_cache.detach().clone()
                pred_roughness = material_rough_cache.detach().clone()
                pred_metallic = material_metal_cache.detach().clone()
                pred_basecolor[subgraph_indices] = pred_sub_basecolor
                pred_roughness[subgraph_indices] = pred_sub_roughness
                pred_metallic[subgraph_indices] = pred_sub_metallic
            else:
                base_logits, rough_logits, metal_logits = predictor(
                    gaussians.get_xyz,
                    gaussians.get_scaling,
                    neighbor_indices,
                    style_code,
                    graph_conv_chunk_size=graph_conv_chunk_size,
                    use_checkpoint=predictor_checkpoint,
                )
                pred_basecolor = torch.sigmoid(base_logits).clamp(0.02, 0.98)
                pred_roughness = torch.sigmoid(rough_logits).clamp(0.04, 0.98)
                pred_metallic = torch.sigmoid(metal_logits).clamp(0.0, 0.999)
        else:
            base_logits = direct_base_logits
            rough_logits = direct_rough_logits
            metal_logits = direct_metal_logits
            pred_basecolor = torch.sigmoid(base_logits).clamp(0.02, 0.98)
            pred_roughness = torch.sigmoid(rough_logits).clamp(0.04, 0.98)
            pred_metallic = torch.sigmoid(metal_logits).clamp(0.0, 0.999)

        transfer_progress = max(iteration - warmup_iters, 0) / max(iterations - warmup_iters, 1)
        warmup_progress = min(iteration / max(warmup_iters, 1), 1.0)

        normal_loss = pred_basecolor.new_tensor(0.0)
        tv_loss = pred_basecolor.new_tensor(0.0)
        style_loss_base = pred_basecolor.new_tensor(0.0)
        style_loss_render = pred_basecolor.new_tensor(0.0)
        color_loss = pred_basecolor.new_tensor(0.0)
        content_loss = pred_basecolor.new_tensor(0.0)
        firefly_loss = pred_basecolor.new_tensor(0.0)
        pseudo_base_loss = pred_basecolor.new_tensor(0.0)
        pseudo_diffuse_loss = pred_basecolor.new_tensor(0.0)
        pseudo_final_loss = pred_basecolor.new_tensor(0.0)

        sampled_views = random.sample(train_cameras, k=min(views_per_iter, len(train_cameras)))
        perceptual_views = set(
            random.sample(sampled_views, k=min(perceptual_views_per_iter, len(sampled_views)))
        )
        debug_render_pkg = None
        debug_neutral = None
        env_names = [name for name in lights.keys() if name != "neutral"]
        if not env_names:
            env_names = ["neutral"]
        if transfer_progress > 0:
            sampled_envs = random.sample(env_names, k=min(max_envs_per_iter, len(env_names)))
        else:
            sampled_envs = ["neutral"]

        for view_cam in sampled_views:
            bg_color = torch.zeros(3, dtype=torch.float32, device=device)
            render_pkg = render(
                viewpoint_camera=view_cam,
                pc=gaussians,
                bg_color=bg_color,
                override_basecolor=pred_basecolor,
                override_roughness=pred_roughness,
                override_metallic=pred_metallic,
                derive_normal=True,
                pad_normal=True,
            )

            normal_mask = render_pkg["normal_mask"]
            depth_mask = render_pkg["normal_from_depth_mask"]
            valid_normal = normal_mask & depth_mask
            if valid_normal.any():
                normal_loss = normal_loss + F.l1_loss(
                    render_pkg["normal_map"][:, valid_normal[0]],
                    render_pkg["normal_map_from_depth"][:, valid_normal[0]],
                )

            material_maps = torch.cat(
                [render_pkg["albedo_map"], render_pkg["roughness_map"], render_pkg["metallic_map"]], dim=0
            )
            if (~normal_mask).any():
                tv_loss = tv_loss + get_masked_tv_loss(normal_mask.float(), prepare_source_image(view_cam, device), material_maps)
            else:
                tv_loss = tv_loss + get_tv_loss(prepare_source_image(view_cam, device), material_maps)

            view_dirs = get_view_dirs(view_cam)
            neutral_result = pbr_shading(
                light=neutral_light,
                normals=render_pkg["normal_map"].permute(1, 2, 0),
                view_dirs=view_dirs,
                albedo=render_pkg["albedo_map"].permute(1, 2, 0),
                roughness=render_pkg["roughness_map"].permute(1, 2, 0),
                metallic=render_pkg["metallic_map"].permute(1, 2, 0),
                mask=render_pkg["normal_mask"].permute(1, 2, 0),
                brdf_lut=brdf_lut,
            )

            object_mask = render_pkg["opacity_map"].clamp(min=0.0, max=1.0)
            if view_cam in perceptual_views:
                base_crop, _ = crop_to_mask(
                    render_pkg["albedo_map"],
                    object_mask,
                    padding=style_crop_padding,
                    fill_color=style_fill_color,
                )
                diffuse_crop, _ = crop_to_mask(
                    neutral_result["diffuse_rgb"].permute(2, 0, 1),
                    object_mask,
                    padding=style_crop_padding,
                    fill_color=style_fill_color,
                )
                color_loss = color_loss + rgb_distribution_loss(
                    render_pkg["albedo_map"],
                    style_target_stats,
                    mask=object_mask,
                )

            if transfer_progress > 0 and view_cam in perceptual_views:
                style_loss_base = style_loss_base + 0.5 * style_encoder.forward_from_cache(
                    resize_for_vgg(base_crop, size=style_resolution), style_target_cache
                )
                style_loss_base = style_loss_base + 0.5 * style_encoder.forward_from_cache(
                    resize_for_vgg(diffuse_crop, size=style_resolution), style_target_cache
                )

                for env_name in sampled_envs:
                    env_light = lights[env_name]
                    env_result = pbr_shading(
                        light=env_light,
                        normals=render_pkg["normal_map"].permute(1, 2, 0),
                        view_dirs=view_dirs,
                        albedo=render_pkg["albedo_map"].permute(1, 2, 0),
                        roughness=render_pkg["roughness_map"].permute(1, 2, 0),
                        metallic=render_pkg["metallic_map"].permute(1, 2, 0),
                        mask=render_pkg["normal_mask"].permute(1, 2, 0),
                        brdf_lut=brdf_lut,
                    )
                    render_crop, _ = crop_to_mask(
                        env_result["render_rgb"].permute(2, 0, 1),
                        object_mask,
                        padding=style_crop_padding,
                        fill_color=style_fill_color,
                    )
                    style_loss_render = style_loss_render + style_encoder.forward_from_cache(
                        resize_for_vgg(render_crop, size=style_resolution), style_target_cache
                    )
                    firefly_loss = firefly_loss + F.relu(env_result["specular_rgb"] - 2.5).mean()

            pseudo_targets = load_pseudo_view_targets(
                pseudo_target_dir,
                getattr(view_cam, "image_name", str(getattr(view_cam, "uid", iteration))),
                device,
                pseudo_view_cache,
            )
            if pseudo_targets is not None:
                pseudo_mask = pseudo_targets["mask"]
                if pseudo_mask is None:
                    pseudo_mask = object_mask
                pseudo_mask = pseudo_mask.clamp(min=0.0, max=1.0)
                if pseudo_targets["baseColor"] is not None:
                    pseudo_base_loss = pseudo_base_loss + masked_l1(
                        render_pkg["albedo_map"],
                        pseudo_targets["baseColor"],
                        pseudo_mask,
                    )
                if pseudo_targets["diffuse"] is not None:
                    pseudo_diffuse_loss = pseudo_diffuse_loss + masked_l1(
                        neutral_result["diffuse_rgb"].permute(2, 0, 1), pseudo_targets["diffuse"], pseudo_mask
                    )
                if pseudo_targets["final"] is not None:
                    neutral_render = neutral_result["render_rgb"].permute(2, 0, 1)
                    pseudo_final_loss = pseudo_final_loss + masked_l1(
                        neutral_render,
                        pseudo_targets["final"],
                        pseudo_mask,
                    )

            if view_cam in perceptual_views:
                source_image = prepare_source_image(view_cam, device)
                neutral_render = composite_background(
                    neutral_result["render_rgb"].permute(2, 0, 1),
                    render_pkg["opacity_map"],
                )
                source_content_cache = style_encoder.build_content_cache(
                    resize_for_vgg(grayscale_triplet(source_image), size=style_resolution)
                )
                content_loss = content_loss + style_encoder.content_loss_from_cache(
                    resize_for_vgg(grayscale_triplet(neutral_render), size=style_resolution),
                    source_content_cache,
                )

            debug_render_pkg = render_pkg
            debug_neutral = neutral_result

        num_views = max(len(sampled_views), 1)
        normal_loss = normal_loss / num_views
        tv_loss = tv_loss / num_views
        num_perceptual_views = max(len(perceptual_views), 1)
        style_loss_base = style_loss_base / max(num_perceptual_views, 1)
        style_loss_render = style_loss_render / max(num_perceptual_views * len(sampled_envs), 1)
        color_loss = color_loss / num_perceptual_views
        content_loss = content_loss / num_perceptual_views
        firefly_loss = firefly_loss / max(num_views * len(sampled_envs), 1)
        pseudo_base_loss = pseudo_base_loss / num_views
        pseudo_diffuse_loss = pseudo_diffuse_loss / num_views
        pseudo_final_loss = pseudo_final_loss / num_views

        knn_loss = knn_material_smoothness(pred_basecolor, pred_roughness, pred_metallic, neighbor_indices)
        priors = material_prior_loss(pred_basecolor, pred_roughness, pred_metallic, material_preset)
        anchor_material = F.l1_loss(pred_basecolor, gaussians.source_basecolor)
        anchor_material = anchor_material + 0.5 * F.l1_loss(pred_roughness, gaussians.source_roughness)
        anchor_material = anchor_material + 0.5 * F.l1_loss(pred_metallic, gaussians.source_metallic)
        pseudo_material_loss = pred_basecolor.new_tensor(0.0)
        if pseudo_material_cache is not None:
            pseudo_material_loss = F.l1_loss(pred_basecolor, pseudo_material_cache["baseColor"])
            pseudo_material_loss = pseudo_material_loss + 0.5 * F.l1_loss(
                pred_roughness,
                pseudo_material_cache["roughness"],
            )
            pseudo_material_loss = pseudo_material_loss + 0.5 * F.l1_loss(
                pred_metallic,
                pseudo_material_cache["metallic"],
            )

        total_loss = pred_basecolor.new_tensor(0.0)
        total_loss = total_loss + 0.25 * normal_loss
        total_loss = total_loss + 0.08 * tv_loss
        total_loss = total_loss + 0.03 * knn_loss
        total_loss = total_loss + priors["basecolor"] + priors["roughness"] + priors["metallic"]
        total_loss = total_loss + anchor_weight * (1.0 - 0.5 * transfer_progress) * content_loss
        total_loss = total_loss + 0.2 * anchor_weight * (1.0 - transfer_progress) * anchor_material
        total_loss = total_loss + color_distribution_weight * transfer_progress * color_loss
        total_loss = total_loss + 0.6 * transfer_progress * style_loss_base
        total_loss = total_loss + 1.0 * transfer_progress * style_loss_render
        total_loss = total_loss + pseudo_base_weight * transfer_progress * pseudo_base_loss
        total_loss = total_loss + pseudo_diffuse_weight * transfer_progress * pseudo_diffuse_loss
        total_loss = total_loss + pseudo_final_weight * transfer_progress * pseudo_final_loss
        total_loss = total_loss + pseudo_material_weight * transfer_progress * pseudo_material_loss
        total_loss = total_loss + 0.02 * transfer_progress * firefly_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Encountered non-finite loss at iteration {iteration}")

        total_loss.backward()
        optimizer.step()

        if material_optimization_mode == "gcn":
            with torch.no_grad():
                if subgraph_indices is not None:
                    material_base_cache[subgraph_indices] = pred_sub_basecolor.detach()
                    material_rough_cache[subgraph_indices] = pred_sub_roughness.detach()
                    material_metal_cache[subgraph_indices] = pred_sub_metallic.detach()
                else:
                    material_base_cache = pred_basecolor.detach().clone()
                    material_rough_cache = pred_roughness.detach().clone()
                    material_metal_cache = pred_metallic.detach().clone()

        pseudo_total = pseudo_base_loss + pseudo_diffuse_loss + pseudo_final_loss + pseudo_material_loss

        loss_row = {
            "iteration": iteration,
            "Tot": float(total_loss.item()),
            "Sty": float((style_loss_base + style_loss_render).item()),
            "Anchor": float(content_loss.item()),
            "Clr": float(color_loss.item()),
            "Pseudo": float(pseudo_total.item()),
            "PseudoBase": float(pseudo_base_loss.item()),
            "PseudoDiffuse": float(pseudo_diffuse_loss.item()),
            "PseudoFinal": float(pseudo_final_loss.item()),
            "PseudoMat": float(pseudo_material_loss.item()),
            "TV": float(tv_loss.item()),
            "Rgh": float(priors["roughness"].item()),
            "Met": float(priors["metallic"].item()),
        }

        progress_bar.set_postfix(
            {
                "Tot": f"{loss_row['Tot']:.4f}",
                "Sty": f"{loss_row['Sty']:.4f}",
                "Anchor": f"{loss_row['Anchor']:.4f}",
                "Clr": f"{loss_row['Clr']:.4f}",
                "PsB": f"{loss_row['PseudoBase']:.4f}",
                "PsD": f"{loss_row['PseudoDiffuse']:.4f}",
                "PsF": f"{loss_row['PseudoFinal']:.4f}",
                "PsM": f"{loss_row['PseudoMat']:.4f}",
                "TV": f"{loss_row['TV']:.4f}",
                "Rgh": f"{loss_row['Rgh']:.4f}",
                "Met": f"{loss_row['Met']:.4f}",
            }
        )

        with open(loss_jsonl_path, "a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(loss_row, ensure_ascii=False) + "\n")
        with open(loss_csv_path, "a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(loss_row.keys()))
            writer.writerow(loss_row)

        if debug_render_pkg is not None and debug_neutral is not None and (iteration == 1 or iteration % 200 == 0):
            export_debug_bundle(debug_dir, iteration, debug_render_pkg, debug_neutral)

    print("Saving final PBR gaussian model...")
    if material_optimization_mode == "gcn":
        gaussians.set_materials(material_base_cache.detach(), material_rough_cache.detach(), material_metal_cache.detach())
    else:
        gaussians.set_materials(pred_basecolor.detach(), pred_roughness.detach(), pred_metallic.detach())
    final_path = os.path.join(output_dir, "final_pbr_model.ply")
    gaussians.save_ply(final_path)

    metadata = {
        "material_preset": material_preset,
        "warmup_iters": warmup_iters,
        "views_per_iter": views_per_iter,
        "perceptual_views_per_iter": perceptual_views_per_iter,
        "env_mode": env_mode,
        "anchor_weight": anchor_weight,
        "style_resolution": style_resolution,
        "max_envs_per_iter": max_envs_per_iter,
        "style_crop_padding": style_crop_padding,
        "color_distribution_weight": color_distribution_weight,
        "pseudo_target_dir": None if pseudo_target_dir is None else os.path.abspath(pseudo_target_dir),
        "pseudo_image_weight": pseudo_image_weight,
        "pseudo_base_weight": pseudo_base_weight,
        "pseudo_diffuse_weight": pseudo_diffuse_weight,
        "pseudo_final_weight": pseudo_final_weight,
        "pseudo_material_weight": pseudo_material_weight,
        "graph_conv_chunk_size": graph_conv_chunk_size,
        "gcn_subgraph_size": gcn_subgraph_size,
        "gcn_seed_size": gcn_seed_size,
        "gcn_expand_hops": gcn_expand_hops,
        "predictor_checkpoint": predictor_checkpoint,
        "material_optimization_mode": material_optimization_mode,
        "direct_mode_gaussian_threshold": direct_mode_gaussian_threshold,
        "direct_material_lr": direct_material_lr,
        "style_image": os.path.abspath(style_image_path),
        "source": os.path.abspath(source_path),
        "ply_path": os.path.abspath(ply_path),
        "loss_jsonl": os.path.abspath(loss_jsonl_path),
        "loss_csv": os.path.abspath(loss_csv_path),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Finished. Saved stylized PBR model to {final_path}")
    return final_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--material_preset", type=str, default="jade", choices=sorted(MATERIAL_PRESETS.keys()))
    parser.add_argument("--warmup_iters", type=int, default=800)
    parser.add_argument("--views_per_iter", type=int, default=1)
    parser.add_argument("--perceptual_views_per_iter", type=int, default=1)
    parser.add_argument("--env_mode", type=str, default="default")
    parser.add_argument("--anchor_weight", type=float, default=0.15)
    parser.add_argument("--style_resolution", type=int, default=256)
    parser.add_argument("--max_envs_per_iter", type=int, default=1)
    parser.add_argument("--style_crop_padding", type=int, default=20)
    parser.add_argument("--color_distribution_weight", type=float, default=0.25)
    parser.add_argument("--pseudo_target_dir", type=str, default=None)
    parser.add_argument("--pseudo_image_weight", type=float, default=0.35)
    parser.add_argument("--pseudo_base_weight", type=float, default=None)
    parser.add_argument("--pseudo_diffuse_weight", type=float, default=None)
    parser.add_argument("--pseudo_final_weight", type=float, default=None)
    parser.add_argument("--pseudo_material_weight", type=float, default=0.2)
    parser.add_argument("--graph_conv_chunk_size", type=int, default=1024)
    parser.add_argument("--gcn_subgraph_size", type=int, default=4096)
    parser.add_argument("--gcn_seed_size", type=int, default=1024)
    parser.add_argument("--gcn_expand_hops", type=int, default=1)
    parser.add_argument("--predictor_checkpoint", dest="predictor_checkpoint", action="store_true")
    parser.add_argument("--no_predictor_checkpoint", dest="predictor_checkpoint", action="store_false")
    parser.set_defaults(predictor_checkpoint=True)
    parser.add_argument("--material_optimization_mode", type=str, default="auto", choices=["auto", "gcn", "direct"])
    parser.add_argument("--direct_mode_gaussian_threshold", type=int, default=120000)
    parser.add_argument("--direct_material_lr", type=float, default=5e-2)
    parser.add_argument("--output_dir", type=str, default="pbr_outputs")
    args = parser.parse_args()

    train_pbr_stylization(
        ply_path=args.ply_path,
        source_path=args.source,
        style_image_path=args.style_image,
        iterations=args.iterations,
        material_preset=args.material_preset,
        warmup_iters=args.warmup_iters,
        views_per_iter=args.views_per_iter,
        perceptual_views_per_iter=args.perceptual_views_per_iter,
        env_mode=args.env_mode,
        anchor_weight=args.anchor_weight,
        style_resolution=args.style_resolution,
        max_envs_per_iter=args.max_envs_per_iter,
        style_crop_padding=args.style_crop_padding,
        color_distribution_weight=args.color_distribution_weight,
        pseudo_target_dir=args.pseudo_target_dir,
        pseudo_image_weight=args.pseudo_image_weight,
        pseudo_base_weight=args.pseudo_base_weight,
        pseudo_diffuse_weight=args.pseudo_diffuse_weight,
        pseudo_final_weight=args.pseudo_final_weight,
        pseudo_material_weight=args.pseudo_material_weight,
        graph_conv_chunk_size=args.graph_conv_chunk_size,
        gcn_subgraph_size=args.gcn_subgraph_size,
        gcn_seed_size=args.gcn_seed_size,
        gcn_expand_hops=args.gcn_expand_hops,
        predictor_checkpoint=args.predictor_checkpoint,
        material_optimization_mode=args.material_optimization_mode,
        direct_mode_gaussian_threshold=args.direct_mode_gaussian_threshold,
        direct_material_lr=args.direct_material_lr,
        output_dir=args.output_dir,
    )
