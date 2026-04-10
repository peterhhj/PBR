import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
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
    "metal": {
        "metallic_target": 0.92,
        "metallic_weight": 0.08,
        "roughness_min": 0.04,
        "roughness_max": 0.45,
        "roughness_weight": 0.08,
        "saturation_max": 0.98,
    },
}


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


def knn_material_smoothness(
    basecolor: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    neighbor_base = basecolor[neighbor_indices]
    neighbor_rough = roughness[neighbor_indices]
    neighbor_metal = metallic[neighbor_indices]
    loss = F.smooth_l1_loss(basecolor.unsqueeze(1), neighbor_base)
    loss = loss + F.smooth_l1_loss(roughness.unsqueeze(1), neighbor_rough)
    loss = loss + F.smooth_l1_loss(metallic.unsqueeze(1), neighbor_metal)
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
    env_mode: str = "default",
    anchor_weight: float = 0.15,
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
    style_target = load_style_image(style_image_path, device)
    style_encoder = VGGStyleLoss(device=str(device))
    with torch.no_grad():
        style_code = style_encoder.encode_style(style_target).detach()

    predictor = PBRMaterialPredictor(
        in_channels=6,
        support_num=4,
        neighbor_num=min(16, neighbor_indices.shape[1]),
        style_dim=style_code.numel(),
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    brdf_lut = get_brdf_lut()
    lights = get_env_lights(env_mode, device)
    neutral_light = lights["neutral"]

    progress_bar = tqdm(range(1, iterations + 1), desc="PBR Training")
    for iteration in progress_bar:
        predictor.train()
        optimizer.zero_grad(set_to_none=True)

        base_logits, rough_logits, metal_logits = predictor(
            gaussians.get_xyz,
            gaussians.get_scaling,
            neighbor_indices,
            style_code,
        )
        pred_basecolor = torch.sigmoid(base_logits).clamp(0.02, 0.98)
        pred_roughness = torch.sigmoid(rough_logits).clamp(0.04, 0.98)
        pred_metallic = torch.sigmoid(metal_logits).clamp(0.0, 0.999)

        transfer_progress = max(iteration - warmup_iters, 0) / max(iterations - warmup_iters, 1)
        warmup_progress = min(iteration / max(warmup_iters, 1), 1.0)

        normal_loss = pred_basecolor.new_tensor(0.0)
        tv_loss = pred_basecolor.new_tensor(0.0)
        style_loss_base = pred_basecolor.new_tensor(0.0)
        style_loss_render = pred_basecolor.new_tensor(0.0)
        content_loss = pred_basecolor.new_tensor(0.0)
        firefly_loss = pred_basecolor.new_tensor(0.0)

        sampled_views = random.sample(train_cameras, k=min(views_per_iter, len(train_cameras)))
        debug_render_pkg = None
        debug_neutral = None
        env_names = [name for name in lights.keys() if name != "neutral"]
        if not env_names:
            env_names = ["neutral"]
        sampled_envs = env_names if transfer_progress > 0 else ["neutral"]

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

            if transfer_progress > 0:
                base_vis = composite_background(render_pkg["albedo_map"], render_pkg["opacity_map"])
                diffuse_vis = composite_background(
                    neutral_result["diffuse_rgb"].permute(2, 0, 1), render_pkg["opacity_map"]
                )
                style_loss_base = style_loss_base + 0.5 * style_encoder(
                    resize_for_vgg(base_vis), style_target
                )
                style_loss_base = style_loss_base + 0.5 * style_encoder(
                    resize_for_vgg(diffuse_vis), style_target
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
                    render_vis = composite_background(env_result["render_rgb"].permute(2, 0, 1), render_pkg["opacity_map"])
                    style_loss_render = style_loss_render + style_encoder(resize_for_vgg(render_vis), style_target)
                    firefly_loss = firefly_loss + F.relu(env_result["specular_rgb"] - 2.5).mean()

            source_image = prepare_source_image(view_cam, device)
            neutral_render = composite_background(neutral_result["render_rgb"].permute(2, 0, 1), render_pkg["opacity_map"])
            content_loss = content_loss + style_encoder.content_loss(
                resize_for_vgg(grayscale_triplet(neutral_render)),
                resize_for_vgg(grayscale_triplet(source_image)),
            )

            debug_render_pkg = render_pkg
            debug_neutral = neutral_result

        num_views = max(len(sampled_views), 1)
        normal_loss = normal_loss / num_views
        tv_loss = tv_loss / num_views
        style_loss_base = style_loss_base / max(num_views, 1)
        style_loss_render = style_loss_render / max(num_views * len(sampled_envs), 1)
        content_loss = content_loss / num_views
        firefly_loss = firefly_loss / max(num_views * len(sampled_envs), 1)

        knn_loss = knn_material_smoothness(pred_basecolor, pred_roughness, pred_metallic, neighbor_indices)
        priors = material_prior_loss(pred_basecolor, pred_roughness, pred_metallic, material_preset)
        anchor_material = F.l1_loss(pred_basecolor, gaussians.source_basecolor)
        anchor_material = anchor_material + 0.5 * F.l1_loss(pred_roughness, gaussians.source_roughness)
        anchor_material = anchor_material + 0.5 * F.l1_loss(pred_metallic, gaussians.source_metallic)

        total_loss = pred_basecolor.new_tensor(0.0)
        total_loss = total_loss + 0.25 * normal_loss
        total_loss = total_loss + 0.08 * tv_loss
        total_loss = total_loss + 0.03 * knn_loss
        total_loss = total_loss + priors["basecolor"] + priors["roughness"] + priors["metallic"]
        total_loss = total_loss + anchor_weight * (1.0 - 0.5 * transfer_progress) * content_loss
        total_loss = total_loss + 0.2 * anchor_weight * (1.0 - transfer_progress) * anchor_material
        total_loss = total_loss + 0.6 * transfer_progress * style_loss_base
        total_loss = total_loss + 1.0 * transfer_progress * style_loss_render
        total_loss = total_loss + 0.02 * transfer_progress * firefly_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Encountered non-finite loss at iteration {iteration}")

        total_loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            {
                "Tot": f"{total_loss.item():.4f}",
                "Sty": f"{(style_loss_base + style_loss_render).item():.4f}",
                "Anchor": f"{content_loss.item():.4f}",
                "TV": f"{tv_loss.item():.4f}",
                "Rgh": f"{priors['roughness'].item():.4f}",
                "Met": f"{priors['metallic'].item():.4f}",
            }
        )

        if debug_render_pkg is not None and debug_neutral is not None and (iteration == 1 or iteration % 200 == 0):
            export_debug_bundle(debug_dir, iteration, debug_render_pkg, debug_neutral)

    print("Saving final PBR gaussian model...")
    gaussians.set_materials(pred_basecolor.detach(), pred_roughness.detach(), pred_metallic.detach())
    final_path = os.path.join(output_dir, "final_pbr_model.ply")
    gaussians.save_ply(final_path)

    metadata = {
        "material_preset": material_preset,
        "warmup_iters": warmup_iters,
        "views_per_iter": views_per_iter,
        "env_mode": env_mode,
        "anchor_weight": anchor_weight,
        "style_image": os.path.abspath(style_image_path),
        "source": os.path.abspath(source_path),
        "ply_path": os.path.abspath(ply_path),
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
    parser.add_argument("--env_mode", type=str, default="default")
    parser.add_argument("--anchor_weight", type=float, default=0.15)
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
        env_mode=args.env_mode,
        anchor_weight=args.anchor_weight,
        output_dir=args.output_dir,
    )
