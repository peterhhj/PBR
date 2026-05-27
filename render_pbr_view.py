import os
from argparse import ArgumentParser
from argparse import Namespace

import torch

from gaussian_renderer import render
from pbr import get_brdf_lut, pbr_shading
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from train_pbr import composite_background, get_env_lights, get_view_dirs, save_tensor_image
from utils.camera_utils import cameraList_from_camInfos

# DEFAULT_DEMO_VIEWS = ["0000", "0010", "0099", "0112", "0148", "0192"]
DEFAULT_DEMO_VIEWS = ["0000", "0007", "0020", "0040", "0060","0085", "0100","0111","0125","0140", "0153","0169","0187","0192"]
# DEFAULT_DEMO_VIEWS = ["0140"]

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


def pick_camera(cameras, view_name: str = None, view_index: int = None):
    if view_name is not None:
        normalized = os.path.splitext(os.path.basename(view_name))[0]
        for camera in cameras:
            camera_name = os.path.splitext(os.path.basename(getattr(camera, "image_name", "")))[0]
            if camera_name == normalized:
                return camera
        raise RuntimeError(f"Could not find camera named '{view_name}'.")

    if view_index is None:
        raise RuntimeError("Either --view_name or --view_index must be provided.")
    if view_index < 0 or view_index >= len(cameras):
        raise RuntimeError(f"view_index {view_index} is out of range for {len(cameras)} cameras.")
    return cameras[view_index]


def render_single_camera(
    gaussians: GaussianModel,
    camera,
    output_dir: str,
    light,
    brdf_lut,
) -> str:
    render_pkg = render(
        viewpoint_camera=camera,
        pc=gaussians,
        bg_color=torch.zeros(3, dtype=torch.float32, device=gaussians.get_xyz.device),
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

    camera_name = os.path.splitext(os.path.basename(getattr(camera, "image_name", f"view_{camera.uid:04d}")))[0]
    prefix = os.path.join(output_dir, camera_name)
    opacity = render_pkg["opacity_map"]

    save_tensor_image(composite_background(render_pkg["albedo_map"], opacity), prefix + "_baseColor.png")
    save_tensor_image(composite_background(result["diffuse_rgb"].permute(2, 0, 1), opacity), prefix + "_diffuse.png")
    save_tensor_image(
        composite_background(result["specular_rgb"].permute(2, 0, 1), opacity, bg_value=0.0),
        prefix + "_specular.png",
    )
    save_tensor_image(composite_background(result["render_rgb"].permute(2, 0, 1), opacity), prefix + "_final.png")
    save_tensor_image((render_pkg["normal_map"] + 1.0) * 0.5, prefix + "_normal.png")
    save_tensor_image(render_pkg["roughness_map"].repeat(3, 1, 1), prefix + "_roughness.png")
    save_tensor_image(render_pkg["metallic_map"].repeat(3, 1, 1), prefix + "_metallic.png")
    save_tensor_image(render_pkg["opacity_map"].repeat(3, 1, 1), prefix + "_opacity.png")
    return camera_name


def render_view(
    ply_path: str,
    source_path: str,
    output_dir: str,
    env_name: str,
    view_name: str = None,
    view_index: int = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    cameras = load_cameras(source_path, device)
    camera = pick_camera(cameras, view_name=view_name, view_index=view_index)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    lights = get_env_lights(env_name, device)
    light = lights["neutral"] if env_name == "default" else lights[env_name]
    brdf_lut = get_brdf_lut()
    camera_name = render_single_camera(gaussians, camera, output_dir, light, brdf_lut)
    print(f"Rendered view '{camera_name}' to {output_dir}")


def render_demo_views(
    ply_path: str,
    source_path: str,
    output_dir: str,
    env_name: str,
    view_names,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    cameras = load_cameras(source_path, device)
    camera_by_name = {
        os.path.splitext(os.path.basename(getattr(camera, "image_name", "")))[0]: camera
        for camera in cameras
    }

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    lights = get_env_lights(env_name, device)
    light = lights["neutral"] if env_name == "default" else lights[env_name]
    brdf_lut = get_brdf_lut()

    rendered = []
    for view_name in view_names:
        normalized = os.path.splitext(os.path.basename(view_name))[0]
        if normalized not in camera_by_name:
            raise RuntimeError(f"Could not find camera named '{view_name}'.")
        rendered_name = render_single_camera(gaussians, camera_by_name[normalized], output_dir, light, brdf_lut)
        rendered.append(rendered_name)

    print(f"Rendered demo views to {output_dir}: {', '.join(rendered)}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--ply", type=str, default="pbr_outputs/final_pbr_model.ply")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="view_renders")
    parser.add_argument("--env_name", type=str, default="studio")
    parser.add_argument("--view_name", type=str, default=None)
    parser.add_argument("--view_index", type=int, default=None)
    parser.add_argument("--demo_views", action="store_true")
    parser.add_argument("--view_names", type=str, nargs="*", default=None)
    args = parser.parse_args()

    if args.demo_views:
        render_demo_views(
            ply_path=args.ply,
            source_path=args.source,
            output_dir=args.output_dir,
            env_name=args.env_name,
            view_names=args.view_names or DEFAULT_DEMO_VIEWS,
        )
    else:
        render_view(
            ply_path=args.ply,
            source_path=args.source,
            output_dir=args.output_dir,
            env_name=args.env_name,
            view_name=args.view_name,
            view_index=args.view_index,
        )


if __name__ == "__main__":
    main()
