import os
from argparse import ArgumentParser, Namespace

import torch

from gaussian_renderer import render
from pbr import get_brdf_lut, pbr_shading
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from train_pbr import composite_background, get_env_lights, get_view_dirs, save_tensor_image
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


def render_all_final_views(
    ply_path: str,
    source_path: str,
    output_dir: str,
    env_name: str = "studio",
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

    print("4. Rendering final images for all views...")
    for camera in cameras:
        render_pkg = render(
            viewpoint_camera=camera,
            pc=gaussians,
            bg_color=torch.zeros(3, dtype=torch.float32, device=device),
            derive_normal=True,
            pad_normal=True,
        )

        result = pbr_shading(
            light=light,
            normals=render_pkg["normal_map"].permute(1, 2, 0),
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
        camera_name = os.path.splitext(os.path.basename(getattr(camera, "image_name", f"view_{camera.uid:04d}")))[0]
        save_tensor_image(final_image, os.path.join(output_dir, f"{camera_name}_final.png"))

    print(f"Finished. Saved {len(cameras)} final renders to {output_dir}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="all_final_views")
    parser.add_argument("--env_name", type=str, default="studio")
    args = parser.parse_args()

    render_all_final_views(
        ply_path=args.ply,
        source_path=args.source,
        output_dir=args.output_dir,
        env_name=args.env_name,
    )


if __name__ == "__main__":
    main()
