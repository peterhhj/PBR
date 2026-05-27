import importlib
import os
from argparse import ArgumentParser
from argparse import Namespace

import torch
import torch.nn.functional as F

from gaussian_renderer import render
from pbr import get_brdf_lut, pbr_shading
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos


def _import_train_module():
    for module_name in ("train_pbr", "train_pbr_gcn"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError("Could not import either 'train_pbr' or 'train_pbr_gcn'.")


_train_module = _import_train_module()
composite_background = _train_module.composite_background
get_env_lights = _train_module.get_env_lights
get_view_dirs = _train_module.get_view_dirs
load_image_tensor = _train_module.load_image_tensor
save_tensor_image = _train_module.save_tensor_image


def normalize_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load_cameras(source_path: str, device: torch.device):
    is_synthetic = os.path.exists(os.path.join(source_path, "transforms_train.json"))
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", eval=False, train_test_exp=False, depths="")
    elif is_synthetic:
        scene_info = sceneLoadTypeCallbacks["Blender"](source_path, True, "", False)
    else:
        raise RuntimeError(f"Could not infer scene type from {source_path}")

    cam_args = Namespace(resolution=1, data_device=str(device), train_test_exp=False)
    return cameraList_from_camInfos(
        scene_info.train_cameras,
        1.0,
        cam_args,
        is_nerf_synthetic=is_synthetic,
        is_test_dataset=False,
    )


def load_fused_normal_view(
    fused_normal_dir: str,
    image_name: str,
    device: torch.device,
    cache,
):
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
    render_pkg,
    fused_normal_dir: str,
    image_name: str,
    device: torch.device,
    cache,
    external_normal_weight: float,
):
    weight = float(min(max(external_normal_weight, 0.0), 1.0))
    base_normal = render_pkg["normal_map"]
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


def render_all_final_views_fused_normals(
    ply_path: str,
    source_path: str,
    fused_normal_dir: str,
    output_dir: str,
    env_name: str = "studio",
    external_normal_weight: float = 1.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading cameras...")
    cameras = load_cameras(source_path, device)
    if not cameras:
        raise RuntimeError("No training cameras were found.")

    print("2. Loading PBR gaussian model...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    print("3. Preparing environment light...")
    lights = get_env_lights(env_name, device)
    light = lights["neutral"] if env_name == "default" else lights[env_name]
    brdf_lut = get_brdf_lut()
    fused_normal_cache = {}

    print("4. Rendering final images for all views with fused normals...")
    for camera in cameras:
        render_pkg = render(
            viewpoint_camera=camera,
            pc=gaussians,
            bg_color=torch.zeros(3, dtype=torch.float32, device=device),
            derive_normal=True,
            pad_normal=True,
        )

        shading_normal = prepare_shading_normal(
            render_pkg,
            fused_normal_dir=fused_normal_dir,
            image_name=getattr(camera, "image_name", f"view_{camera.uid:04d}"),
            device=device,
            cache=fused_normal_cache,
            external_normal_weight=external_normal_weight,
        )

        result = pbr_shading(
            light=light,
            normals=shading_normal.permute(1, 2, 0),
            view_dirs=get_view_dirs(camera),
            albedo=render_pkg["albedo_map"].permute(1, 2, 0),
            roughness=render_pkg["roughness_map"].permute(1, 2, 0),
            metallic=render_pkg["metallic_map"].permute(1, 2, 0),
            mask=render_pkg["normal_mask"].permute(1, 2, 0),
            brdf_lut=brdf_lut,
        )

        final_image = composite_background(
            result["render_rgb"].permute(2, 0, 1),
            render_pkg["opacity_map"],
        )
        camera_name = normalize_name(getattr(camera, "image_name", f"view_{camera.uid:04d}"))
        save_tensor_image(final_image, os.path.join(output_dir, f"{camera_name}_final.png"))

    print(f"Finished. Saved {len(cameras)} fused-normal final renders to {output_dir}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--fused_normal_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="all_final_views_fused_normals")
    parser.add_argument("--env_name", type=str, default="studio")
    parser.add_argument("--external_normal_weight", type=float, default=1.0)
    args = parser.parse_args()

    render_all_final_views_fused_normals(
        ply_path=args.ply,
        source_path=args.source,
        fused_normal_dir=args.fused_normal_dir,
        output_dir=args.output_dir,
        env_name=args.env_name,
        external_normal_weight=args.external_normal_weight,
    )


if __name__ == "__main__":
    main()
