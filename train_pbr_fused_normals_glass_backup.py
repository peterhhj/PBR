import csv
import json
import os
import random
from argparse import ArgumentParser
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import importlib

from gaussian_renderer import render
from pbr import get_brdf_lut, pbr_shading
from pbr_modules.predictor import PBRMaterialPredictor
from pbr_modules.style_loss import VGGStyleLoss
from scene.gaussian_model import GaussianModel


def _import_train_module():
    for module_name in ("train_pbr", "train_pbr_gcn"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError("Could not import either 'train_pbr' or 'train_pbr_gcn'.")


_train_module = _import_train_module()

MATERIAL_PRESETS = _train_module.MATERIAL_PRESETS
build_subgraph_neighbor_indices = _train_module.build_subgraph_neighbor_indices
rgb_distribution_loss = _train_module.rgb_distribution_loss
composite_background = _train_module.composite_background
compute_knn_cpu = _train_module.compute_knn_cpu
compute_rgb_stats = _train_module.compute_rgb_stats
crop_to_mask = _train_module.crop_to_mask
expand_subgraph_indices = _train_module.expand_subgraph_indices
get_env_lights = _train_module.get_env_lights
get_masked_tv_loss = _train_module.get_masked_tv_loss
get_tv_loss = _train_module.get_tv_loss
get_view_dirs = _train_module.get_view_dirs
grayscale_triplet = _train_module.grayscale_triplet
knn_material_smoothness = _train_module.knn_material_smoothness
load_image_tensor = _train_module.load_image_tensor
load_pseudo_material_cache = _train_module.load_pseudo_material_cache
load_pseudo_view_targets = _train_module.load_pseudo_view_targets
load_style_image = _train_module.load_style_image
load_train_cameras = _train_module.load_train_cameras
masked_l1 = _train_module.masked_l1
material_prior_loss = _train_module.material_prior_loss
prepare_source_image = _train_module.prepare_source_image
resize_for_vgg = _train_module.resize_for_vgg
safe_logit = _train_module.safe_logit
save_tensor_image = _train_module.save_tensor_image


def normalize_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load_fused_normal_view(
    fused_normal_dir: Optional[str],
    image_name: str,
    device: torch.device,
    cache: Dict[str, Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    if fused_normal_dir is None:
        return None
    cache_key = normalize_name(image_name)
    if cache_key in cache:
        tensor = cache[cache_key]
        return None if tensor is None else tensor.to(device=device, non_blocking=True)

    stem = cache_key
    candidates = [
        os.path.join(fused_normal_dir, f"{stem}_fused_normal.npy"),
        os.path.join(fused_normal_dir, f"{stem}_fused_normal.png"),
        os.path.join(fused_normal_dir, f"{stem}.npy"),
        os.path.join(fused_normal_dir, f"{stem}.png"),
    ]
    recursive_candidates = []
    for root, _, files in os.walk(fused_normal_dir):
        for filename in files:
            if filename in {
                f"{stem}_fused_normal.npy",
                f"{stem}_fused_normal.png",
                f"{stem}.npy",
                f"{stem}.png",
            }:
                recursive_candidates.append(os.path.join(root, filename))
    candidates.extend(sorted(set(recursive_candidates)))

    tensor = None
    for candidate in candidates:
        if os.path.exists(candidate):
            tensor = load_image_tensor(candidate, device=None, mode="RGB")
            break

    if tensor is None:
        cache[cache_key] = None
        return None

    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor * 2.0 - 1.0
    tensor = F.normalize(tensor, dim=0, eps=1e-6)
    cache[cache_key] = tensor.cpu()
    return tensor.to(device=device, non_blocking=True)


def prepare_shading_normal(
    render_pkg: Dict[str, torch.Tensor],
    fused_normal_dir: Optional[str],
    image_name: str,
    device: torch.device,
    cache: Dict[str, Optional[torch.Tensor]],
    external_normal_weight: float,
) -> torch.Tensor:
    weight = float(min(max(external_normal_weight, 0.0), 1.0))
    base_normal = render_pkg["normal_map"]
    if fused_normal_dir is None:
        return base_normal

    external_normal = load_fused_normal_view(fused_normal_dir, image_name, device, cache)
    if external_normal is None:
        raise RuntimeError(
            f"Could not find fused normal for view '{image_name}' in {fused_normal_dir}. "
            "Expected files like '<view>_fused_normal.png', '<view>_fused_normal.npy', '<view>.png', or '<view>.npy'."
        )

    if external_normal.shape[-2:] != base_normal.shape[-2:]:
        external_normal = F.interpolate(
            external_normal.unsqueeze(0),
            size=base_normal.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        external_normal = F.normalize(external_normal, dim=0, eps=1e-6)

    fused = (1.0 - weight) * base_normal + weight * external_normal
    fused = F.normalize(fused, dim=0, eps=1e-6)
    opacity = render_pkg["opacity_map"]
    bg_normal = torch.tensor([0.0, 0.0, 1.0], dtype=fused.dtype, device=fused.device).view(3, 1, 1)
    return torch.where(opacity > 1e-5, fused, bg_normal)


def metal_brightness_floor_loss(
    basecolor: torch.Tensor,
    metallic: torch.Tensor,
    target_luminance: float = 0.55,
    metallic_threshold: float = 0.5,
) -> torch.Tensor:
    metallic_weights = (metallic.detach().clamp(0.0, 1.0) > metallic_threshold).to(basecolor.dtype)
    if metallic_weights.sum() < 1.0:
        return basecolor.new_tensor(0.0)
    luminance = (
        0.2126 * basecolor[..., 0:1] +
        0.7152 * basecolor[..., 1:2] +
        0.0722 * basecolor[..., 2:3]
    )
    penalty = F.relu(float(target_luminance) - luminance)
    return (penalty * metallic_weights).sum() / metallic_weights.sum().clamp(min=1.0)


def metal_neutrality_loss(
    basecolor: torch.Tensor,
    metallic: torch.Tensor,
    metallic_threshold: float = 0.5,
) -> torch.Tensor:
    metallic_weights = (metallic.detach().clamp(0.0, 1.0) > metallic_threshold).to(basecolor.dtype)
    if metallic_weights.sum() < 1.0:
        return basecolor.new_tensor(0.0)
    channel_mean = basecolor.mean(dim=-1, keepdim=True)
    chroma = (basecolor - channel_mean).abs().mean(dim=-1, keepdim=True)
    return (chroma * metallic_weights).sum() / metallic_weights.sum().clamp(min=1.0)


def metal_roughness_floor_loss(
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    target_roughness: float = 0.3,
    metallic_threshold: float = 0.5,
) -> torch.Tensor:
    metallic_weights = (metallic.detach().clamp(0.0, 1.0) > metallic_threshold).to(roughness.dtype)
    if metallic_weights.sum() < 1.0:
        return roughness.new_tensor(0.0)
    penalty = F.relu(float(target_roughness) - roughness)
    return (penalty * metallic_weights).sum() / metallic_weights.sum().clamp(min=1.0)


def metal_roughness_smoothness_loss(
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    neighbor_indices: torch.Tensor,
    metallic_threshold: float = 0.5,
) -> torch.Tensor:
    metallic_mask = (metallic.detach().clamp(0.0, 1.0) > metallic_threshold).to(roughness.dtype)
    if metallic_mask.sum() < 1.0:
        return roughness.new_tensor(0.0)
    neighbor_roughness = roughness[neighbor_indices]
    neighbor_mask = metallic_mask[neighbor_indices]
    center_mask = metallic_mask.unsqueeze(1)
    pair_mask = center_mask * neighbor_mask
    if pair_mask.sum() < 1.0:
        return roughness.new_tensor(0.0)
    diffs = (roughness.unsqueeze(1) - neighbor_roughness).abs()
    return (diffs * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)


def metal_render_dark_region_loss(
    render_rgb: torch.Tensor,
    metallic_map: torch.Tensor,
    opacity_map: torch.Tensor,
    target_luminance: float = 0.2,
    metallic_threshold: float = 0.5,
) -> torch.Tensor:
    if render_rgb.ndim != 3:
        raise ValueError("render_rgb must be CHW")
    if metallic_map.ndim == 3:
        metallic_map = metallic_map[0]
    if opacity_map.ndim == 3:
        opacity_map = opacity_map[0]
    luminance = (
        0.2126 * render_rgb[0] +
        0.7152 * render_rgb[1] +
        0.0722 * render_rgb[2]
    )
    metallic_mask = (metallic_map.detach().clamp(0.0, 1.0) > metallic_threshold).to(render_rgb.dtype)
    visible_mask = (opacity_map > 1e-5).to(render_rgb.dtype)
    mask = metallic_mask * visible_mask
    if mask.sum() < 1.0:
        return render_rgb.new_tensor(0.0)
    penalty = F.relu(float(target_luminance) - luminance)
    return (penalty * mask).sum() / mask.sum().clamp(min=1.0)


def export_debug_bundle_fused(
    debug_dir: str,
    iteration: int,
    render_pkg: Dict[str, torch.Tensor],
    shading_normal: torch.Tensor,
    neutral_result: Dict[str, torch.Tensor],
) -> None:
    prefix = os.path.join(debug_dir, f"iter_{iteration:05d}")
    save_tensor_image(composite_background(render_pkg["albedo_map"], render_pkg["opacity_map"]), prefix + "_baseColor.png")
    save_tensor_image(render_pkg["roughness_map"].repeat(3, 1, 1), prefix + "_roughness.png")
    save_tensor_image(render_pkg["metallic_map"].repeat(3, 1, 1), prefix + "_metallic.png")
    save_tensor_image((render_pkg["normal_map"] + 1.0) * 0.5, prefix + "_normal.png")
    save_tensor_image((render_pkg["normal_map_from_depth"] + 1.0) * 0.5, prefix + "_normal_from_depth.png")
    save_tensor_image((shading_normal + 1.0) * 0.5, prefix + "_fused_normal.png")
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


def train_pbr_stylization_fused_normals(
    ply_path: str,
    source_path: str,
    style_image_path: str,
    fused_normal_dir: str,
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
    external_normal_weight: float = 1.0,
    metal_brightness_floor_weight: float = 0.08,
    metal_brightness_target: float = 0.55,
    metal_neutrality_weight: float = 0.04,
    metal_region_threshold: float = 0.5,
    metal_roughness_floor_weight: float = 0.10,
    metal_roughness_target: float = 0.30,
    metal_roughness_smoothness_weight: float = 0.08,
    metal_render_dark_weight: float = 0.10,
    metal_render_dark_target: float = 0.22,
    output_dir: str = "pbr_outputs_fused_normals",
) -> str:
    if material_preset not in MATERIAL_PRESETS:
        raise ValueError(f"Unknown material preset: {material_preset}")
    if not os.path.isdir(fused_normal_dir):
        raise RuntimeError(f"Fused normal directory does not exist: {fused_normal_dir}")

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

    print("3. Preparing style encoder, material predictor, and fused normal cache...")
    style_target = load_style_image(style_image_path, device, resolution=style_resolution)
    style_encoder = VGGStyleLoss(device=str(device))
    with torch.no_grad():
        style_code = style_encoder.encode_style(style_target).detach()
        style_target_cache = style_encoder.build_style_cache(style_target)
        style_target_stats = compute_rgb_stats(style_target[0])
        style_fill_color = style_target[0].mean(dim=(-1, -2))

    fused_normal_cache: Dict[str, Optional[torch.Tensor]] = {}
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
                "iteration", "Tot", "Sty", "Anchor", "Clr", "Pseudo",
                "PseudoBase", "PseudoDiffuse", "PseudoFinal", "PseudoMat",
                "TV", "Rgh", "Met", "MBF", "MNeu", "MRF", "MRS", "MRD",
            ],
        )
        writer.writeheader()

    progress_bar = tqdm(range(1, iterations + 1), desc="PBR Training (Fused Normals)")
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
                    seed_indices = subgraph_queue[subgraph_ptr: subgraph_ptr + seed_size]
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
            pred_basecolor = torch.sigmoid(direct_base_logits).clamp(0.02, 0.98)
            pred_roughness = torch.sigmoid(direct_rough_logits).clamp(0.04, 0.98)
            pred_metallic = torch.sigmoid(direct_metal_logits).clamp(0.0, 0.999)

        transfer_progress = max(iteration - warmup_iters, 0) / max(iterations - warmup_iters, 1)
        normal_loss = pred_basecolor.new_tensor(0.0)
        tv_loss = pred_basecolor.new_tensor(0.0)
        style_loss_base = pred_basecolor.new_tensor(0.0)
        style_loss_render = pred_basecolor.new_tensor(0.0)
        color_loss = pred_basecolor.new_tensor(0.0)
        content_loss = pred_basecolor.new_tensor(0.0)
        firefly_loss = pred_basecolor.new_tensor(0.0)
        metal_render_dark = pred_basecolor.new_tensor(0.0)
        pseudo_base_loss = pred_basecolor.new_tensor(0.0)
        pseudo_diffuse_loss = pred_basecolor.new_tensor(0.0)
        pseudo_final_loss = pred_basecolor.new_tensor(0.0)

        sampled_views = random.sample(train_cameras, k=min(views_per_iter, len(train_cameras)))
        perceptual_views = set(random.sample(sampled_views, k=min(perceptual_views_per_iter, len(sampled_views))))
        debug_render_pkg = None
        debug_neutral = None
        debug_shading_normal = None
        env_names = [name for name in lights.keys() if name != "neutral"] or ["neutral"]
        sampled_envs = random.sample(env_names, k=min(max_envs_per_iter, len(env_names))) if transfer_progress > 0 else ["neutral"]

        for view_cam in sampled_views:
            render_pkg = render(
                viewpoint_camera=view_cam,
                pc=gaussians,
                bg_color=torch.zeros(3, dtype=torch.float32, device=device),
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

            shading_normal = prepare_shading_normal(
                render_pkg,
                fused_normal_dir=fused_normal_dir,
                image_name=getattr(view_cam, "image_name", str(getattr(view_cam, "uid", iteration))),
                device=device,
                cache=fused_normal_cache,
                external_normal_weight=external_normal_weight,
            )
            view_dirs = get_view_dirs(view_cam)
            neutral_result = pbr_shading(
                light=neutral_light,
                normals=shading_normal.permute(1, 2, 0),
                view_dirs=view_dirs,
                albedo=render_pkg["albedo_map"].permute(1, 2, 0),
                roughness=render_pkg["roughness_map"].permute(1, 2, 0),
                metallic=render_pkg["metallic_map"].permute(1, 2, 0),
                mask=render_pkg["normal_mask"].permute(1, 2, 0),
                brdf_lut=brdf_lut,
            )

            object_mask = render_pkg["opacity_map"].clamp(min=0.0, max=1.0)
            if view_cam in perceptual_views:
                base_crop, _ = crop_to_mask(render_pkg["albedo_map"], object_mask, padding=style_crop_padding, fill_color=style_fill_color)
                diffuse_crop, _ = crop_to_mask(
                    neutral_result["diffuse_rgb"].permute(2, 0, 1),
                    object_mask,
                    padding=style_crop_padding,
                    fill_color=style_fill_color,
                )
                color_loss = color_loss + rgb_distribution_loss(render_pkg["albedo_map"], style_target_stats, mask=object_mask)

            if transfer_progress > 0 and view_cam in perceptual_views:
                style_loss_base = style_loss_base + 0.5 * style_encoder.forward_from_cache(
                    resize_for_vgg(base_crop, size=style_resolution), style_target_cache
                )
                style_loss_base = style_loss_base + 0.5 * style_encoder.forward_from_cache(
                    resize_for_vgg(diffuse_crop, size=style_resolution), style_target_cache
                )
                for env_name in sampled_envs:
                    env_result = pbr_shading(
                        light=lights[env_name],
                        normals=shading_normal.permute(1, 2, 0),
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
                pseudo_mask = pseudo_targets["mask"] if pseudo_targets["mask"] is not None else object_mask
                pseudo_mask = pseudo_mask.clamp(min=0.0, max=1.0)
                if pseudo_targets["baseColor"] is not None:
                    pseudo_base_loss = pseudo_base_loss + masked_l1(render_pkg["albedo_map"], pseudo_targets["baseColor"], pseudo_mask)
                if pseudo_targets["diffuse"] is not None:
                    pseudo_diffuse_loss = pseudo_diffuse_loss + masked_l1(
                        neutral_result["diffuse_rgb"].permute(2, 0, 1), pseudo_targets["diffuse"], pseudo_mask
                    )
                if pseudo_targets["final"] is not None:
                    pseudo_final_loss = pseudo_final_loss + masked_l1(
                        neutral_result["render_rgb"].permute(2, 0, 1), pseudo_targets["final"], pseudo_mask
                    )

            if view_cam in perceptual_views:
                source_image = prepare_source_image(view_cam, device)
                neutral_render = composite_background(neutral_result["render_rgb"].permute(2, 0, 1), render_pkg["opacity_map"])
                source_content_cache = style_encoder.build_content_cache(
                    resize_for_vgg(grayscale_triplet(source_image), size=style_resolution)
                )
                content_loss = content_loss + style_encoder.content_loss_from_cache(
                    resize_for_vgg(grayscale_triplet(neutral_render), size=style_resolution),
                    source_content_cache,
                )

            if material_preset == "metal":
                metal_render_dark = metal_render_dark + metal_render_dark_region_loss(
                    neutral_result["render_rgb"].permute(2, 0, 1),
                    render_pkg["metallic_map"],
                    render_pkg["opacity_map"],
                    target_luminance=metal_render_dark_target,
                    metallic_threshold=metal_region_threshold,
                )

            debug_render_pkg = render_pkg
            debug_neutral = neutral_result
            debug_shading_normal = shading_normal

        num_views = max(len(sampled_views), 1)
        normal_loss = normal_loss / num_views
        tv_loss = tv_loss / num_views
        num_perceptual_views = max(len(perceptual_views), 1)
        style_loss_base = style_loss_base / num_perceptual_views
        style_loss_render = style_loss_render / max(num_perceptual_views * len(sampled_envs), 1)
        color_loss = color_loss / num_perceptual_views
        content_loss = content_loss / num_perceptual_views
        firefly_loss = firefly_loss / max(num_views * len(sampled_envs), 1)
        metal_render_dark = metal_render_dark / num_views
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
            pseudo_material_loss = pseudo_material_loss + 0.5 * F.l1_loss(pred_roughness, pseudo_material_cache["roughness"])
            pseudo_material_loss = pseudo_material_loss + 0.5 * F.l1_loss(pred_metallic, pseudo_material_cache["metallic"])

        metal_brightness_loss = pred_basecolor.new_tensor(0.0)
        metal_neutral_loss = pred_basecolor.new_tensor(0.0)
        metal_roughness_floor = pred_basecolor.new_tensor(0.0)
        metal_roughness_smooth = pred_basecolor.new_tensor(0.0)
        metal_render_dark = pred_basecolor.new_tensor(0.0)
        if material_preset == "metal":
            metal_brightness_loss = metal_brightness_floor_loss(
                pred_basecolor,
                pred_metallic,
                target_luminance=metal_brightness_target,
                metallic_threshold=metal_region_threshold,
            )
            metal_neutral_loss = metal_neutrality_loss(
                pred_basecolor,
                pred_metallic,
                metallic_threshold=metal_region_threshold,
            )
            metal_roughness_floor = metal_roughness_floor_loss(
                pred_roughness,
                pred_metallic,
                target_roughness=metal_roughness_target,
                metallic_threshold=metal_region_threshold,
            )
            metal_roughness_smooth = metal_roughness_smoothness_loss(
                pred_roughness,
                pred_metallic,
                neighbor_indices,
                metallic_threshold=metal_region_threshold,
            )
            if debug_neutral is not None and debug_render_pkg is not None:
                metal_render_dark = metal_render_dark_region_loss(
                    debug_neutral["render_rgb"].permute(2, 0, 1),
                    debug_render_pkg["metallic_map"],
                    debug_render_pkg["opacity_map"],
                    target_luminance=metal_render_dark_target,
                    metallic_threshold=metal_region_threshold,
                )

        total_loss = pred_basecolor.new_tensor(0.0)
        total_loss = total_loss + 0.25 * normal_loss + 0.08 * tv_loss + 0.03 * knn_loss
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
        total_loss = total_loss + metal_brightness_floor_weight * transfer_progress * metal_brightness_loss
        total_loss = total_loss + metal_neutrality_weight * transfer_progress * metal_neutral_loss
        total_loss = total_loss + metal_roughness_floor_weight * transfer_progress * metal_roughness_floor
        total_loss = total_loss + metal_roughness_smoothness_weight * transfer_progress * metal_roughness_smooth
        total_loss = total_loss + metal_render_dark_weight * transfer_progress * metal_render_dark

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
            "MBF": float(metal_brightness_loss.item()),
            "MNeu": float(metal_neutral_loss.item()),
            "MRF": float(metal_roughness_floor.item()),
            "MRS": float(metal_roughness_smooth.item()),
            "MRD": float(metal_render_dark.item()),
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
                "MBF": f"{loss_row['MBF']:.4f}",
                "MNeu": f"{loss_row['MNeu']:.4f}",
                "MRF": f"{loss_row['MRF']:.4f}",
                "MRS": f"{loss_row['MRS']:.4f}",
                "MRD": f"{loss_row['MRD']:.4f}",
            }
        )
        with open(loss_jsonl_path, "a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(loss_row, ensure_ascii=False) + "\n")
        with open(loss_csv_path, "a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(loss_row.keys()))
            writer.writerow(loss_row)

        if debug_render_pkg is not None and debug_neutral is not None and debug_shading_normal is not None and (iteration == 1 or iteration % 200 == 0):
            export_debug_bundle_fused(debug_dir, iteration, debug_render_pkg, debug_shading_normal, debug_neutral)

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
        "fused_normal_dir": os.path.abspath(fused_normal_dir),
        "external_normal_weight": external_normal_weight,
        "metal_brightness_floor_weight": metal_brightness_floor_weight,
        "metal_brightness_target": metal_brightness_target,
        "metal_neutrality_weight": metal_neutrality_weight,
        "metal_region_threshold": metal_region_threshold,
        "metal_roughness_floor_weight": metal_roughness_floor_weight,
        "metal_roughness_target": metal_roughness_target,
        "metal_roughness_smoothness_weight": metal_roughness_smoothness_weight,
        "metal_render_dark_weight": metal_render_dark_weight,
        "metal_render_dark_target": metal_render_dark_target,
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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--fused_normal_dir", type=str, required=True)
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
    parser.add_argument("--external_normal_weight", type=float, default=1.0)
    parser.add_argument("--metal_brightness_floor_weight", type=float, default=0.08)
    parser.add_argument("--metal_brightness_target", type=float, default=0.55)
    parser.add_argument("--metal_neutrality_weight", type=float, default=0.04)
    parser.add_argument("--metal_region_threshold", type=float, default=0.5)
    parser.add_argument("--metal_roughness_floor_weight", type=float, default=0.10)
    parser.add_argument("--metal_roughness_target", type=float, default=0.30)
    parser.add_argument("--metal_roughness_smoothness_weight", type=float, default=0.08)
    parser.add_argument("--metal_render_dark_weight", type=float, default=0.10)
    parser.add_argument("--metal_render_dark_target", type=float, default=0.22)
    parser.add_argument("--output_dir", type=str, default="pbr_outputs_fused_normals")
    args = parser.parse_args()

    train_pbr_stylization_fused_normals(
        ply_path=args.ply_path,
        source_path=args.source,
        style_image_path=args.style_image,
        fused_normal_dir=args.fused_normal_dir,
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
        external_normal_weight=args.external_normal_weight,
        metal_brightness_floor_weight=args.metal_brightness_floor_weight,
        metal_brightness_target=args.metal_brightness_target,
        metal_neutrality_weight=args.metal_neutrality_weight,
        metal_region_threshold=args.metal_region_threshold,
        metal_roughness_floor_weight=args.metal_roughness_floor_weight,
        metal_roughness_target=args.metal_roughness_target,
        metal_roughness_smoothness_weight=args.metal_roughness_smoothness_weight,
        metal_render_dark_weight=args.metal_render_dark_weight,
        metal_render_dark_target=args.metal_render_dark_target,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
