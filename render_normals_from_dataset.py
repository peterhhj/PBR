import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Iterable
from typing import List
from typing import Optional

import torch
import torch.nn.functional as F

from gaussian_renderer import render
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from train_pbr import composite_background, save_tensor_image
from utils.camera_utils import cameraList_from_camInfos


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


def normalize_camera_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def build_camera_index(cameras) -> dict:
    return {
        normalize_camera_name(getattr(camera, "image_name", f"view_{camera.uid:04d}")): camera
        for camera in cameras
    }


def select_cameras(
    cameras,
    view_names: Optional[Iterable[str]] = None,
    view_indices: Optional[Iterable[int]] = None,
    all_views: bool = False,
) -> List:
    if all_views or (not view_names and not view_indices):
        return list(cameras)

    selected = []
    seen = set()
    camera_by_name = build_camera_index(cameras)

    if view_names:
        for view_name in view_names:
            normalized = normalize_camera_name(view_name)
            if normalized not in camera_by_name:
                raise RuntimeError(f"Could not find camera named '{view_name}'.")
            if normalized not in seen:
                selected.append(camera_by_name[normalized])
                seen.add(normalized)

    if view_indices:
        for view_index in view_indices:
            if view_index < 0 or view_index >= len(cameras):
                raise RuntimeError(f"view_index {view_index} is out of range for {len(cameras)} cameras.")
            camera = cameras[view_index]
            normalized = normalize_camera_name(getattr(camera, "image_name", f"view_{camera.uid:04d}"))
            if normalized not in seen:
                selected.append(camera)
                seen.add(normalized)

    return selected


def fuse_normal_maps(
    normal_map: torch.Tensor,
    normal_from_depth: torch.Tensor,
    opacity: torch.Tensor,
    depth_weight: float = 0.7,
) -> torch.Tensor:
    weight = float(min(max(depth_weight, 0.0), 1.0))
    fused = (1.0 - weight) * normal_map + weight * normal_from_depth
    fused = F.normalize(fused, dim=0, eps=1e-6)
    background = torch.tensor([0.0, 0.0, 1.0], dtype=fused.dtype, device=fused.device).view(3, 1, 1)
    return torch.where(opacity > 1e-5, fused, background)


def export_normal_for_camera(
    gaussians: GaussianModel,
    camera,
    output_dir: str,
    normal_bg_value: float = 0.5,
    fused_depth_weight: float = 0.7,
) -> str:
    render_pkg = render(
        viewpoint_camera=camera,
        pc=gaussians,
        bg_color=torch.zeros(3, dtype=torch.float32, device=gaussians.get_xyz.device),
        derive_normal=True,
        pad_normal=True,
    )

    camera_name = normalize_camera_name(getattr(camera, "image_name", f"view_{camera.uid:04d}"))
    prefix = os.path.join(output_dir, camera_name)

    normal_vis = (render_pkg["normal_map"] + 1.0) * 0.5
    normal_from_depth_vis = (render_pkg["normal_map_from_depth"] + 1.0) * 0.5
    opacity = render_pkg["opacity_map"]
    fused_normal = fuse_normal_maps(
        render_pkg["normal_map"],
        render_pkg["normal_map_from_depth"],
        opacity,
        depth_weight=fused_depth_weight,
    )
    fused_normal_vis = (fused_normal + 1.0) * 0.5

    save_tensor_image(normal_vis, prefix + "_normal.png")
    save_tensor_image(normal_from_depth_vis, prefix + "_normal_from_depth.png")
    save_tensor_image(fused_normal_vis, prefix + "_fused_normal.png")
    save_tensor_image(opacity.repeat(3, 1, 1), prefix + "_opacity.png")
    save_tensor_image(
        composite_background(normal_vis, opacity, bg_value=normal_bg_value),
        prefix + "_normal_on_bg.png",
    )
    save_tensor_image(
        composite_background(fused_normal_vis, opacity, bg_value=normal_bg_value),
        prefix + "_fused_normal_on_bg.png",
    )

    with open(prefix + "_camera.txt", "w", encoding="utf-8") as file:
        file.write(f"camera_name={camera_name}\n")
        file.write(f"image_name={getattr(camera, 'image_name', '')}\n")
        file.write(f"uid={getattr(camera, 'uid', -1)}\n")
        file.write(f"width={getattr(camera, 'image_width', getattr(camera, 'width', -1))}\n")
        file.write(f"height={getattr(camera, 'image_height', getattr(camera, 'height', -1))}\n")
        file.write(f"fused_depth_weight={fused_depth_weight}\n")
        if hasattr(camera, "world_view_transform"):
            file.write("world_view_transform=\n")
            matrix = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
            for row in matrix:
                file.write("  " + " ".join(f"{value:.8f}" for value in row) + "\n")

    return camera_name


def render_normals_from_dataset(
    ply_path: str,
    source_path: str,
    output_dir: str,
    view_names: Optional[Iterable[str]] = None,
    view_indices: Optional[Iterable[int]] = None,
    all_views: bool = False,
    normal_bg_value: float = 0.5,
    fused_depth_weight: float = 0.7,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    cameras = load_cameras(source_path, device)
    selected_cameras = select_cameras(
        cameras,
        view_names=view_names,
        view_indices=view_indices,
        all_views=all_views,
    )

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    rendered = []
    for camera in selected_cameras:
        rendered.append(
            export_normal_for_camera(
                gaussians,
                camera,
                output_dir,
                normal_bg_value=normal_bg_value,
                fused_depth_weight=fused_depth_weight,
            )
        )

    print(f"Rendered {len(rendered)} dataset-view normal bundles to {output_dir}")
    if rendered:
        print("Views: " + ", ".join(rendered[:10]) + (" ..." if len(rendered) > 10 else ""))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset_normals")
    parser.add_argument("--view_names", type=str, nargs="*", default=None)
    parser.add_argument("--view_indices", type=int, nargs="*", default=None)
    parser.add_argument("--all_views", action="store_true")
    parser.add_argument("--normal_bg_value", type=float, default=0.5)
    parser.add_argument("--fused_depth_weight", type=float, default=0.7)
    args = parser.parse_args()

    render_normals_from_dataset(
        ply_path=args.ply,
        source_path=args.source,
        output_dir=args.output_dir,
        view_names=args.view_names,
        view_indices=args.view_indices,
        all_views=args.all_views,
        normal_bg_value=args.normal_bg_value,
        fused_depth_weight=args.fused_depth_weight,
    )


if __name__ == "__main__":
    main()
