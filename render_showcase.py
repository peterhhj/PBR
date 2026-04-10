import math
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from gaussian_renderer import render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix


def save_tensor_image(image: torch.Tensor, path: str) -> None:
    from PIL import Image
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = image.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def make_orbit_camera(
    center: torch.Tensor,
    radius: float,
    angle: float,
    elevation: float,
    resolution: int,
    device: torch.device,
) -> MiniCam:
    cam_pos = center + torch.tensor(
        [
            radius * math.cos(elevation) * math.sin(angle),
            radius * math.sin(elevation),
            radius * math.cos(elevation) * math.cos(angle),
        ],
        dtype=torch.float32,
        device=device,
    )
    forward = torch.nn.functional.normalize(center - cam_pos, dim=0)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    right = torch.nn.functional.normalize(torch.cross(world_up, forward, dim=0), dim=0)
    up = torch.nn.functional.normalize(torch.cross(forward, right, dim=0), dim=0)

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos

    w2c = torch.inverse(c2w)
    world_view_transform = w2c.transpose(0, 1).contiguous()
    fov = math.radians(50.0)
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(device)
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    return MiniCam(
        width=resolution,
        height=resolution,
        fovy=fov,
        fovx=fov,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def get_view_dirs(camera: MiniCam) -> torch.Tensor:
    H, W = camera.image_height, camera.image_width
    cen_x = W / 2.0
    cen_y = H / 2.0
    focal = W / (2.0 * math.tan(camera.FoVx * 0.5))
    x, y = torch.meshgrid(
        torch.arange(W, device=camera.camera_center.device),
        torch.arange(H, device=camera.camera_center.device),
        indexing="xy",
    )
    canonical = torch.nn.functional.pad(
        torch.stack([(x.flatten() - cen_x + 0.5) / focal, (y.flatten() - cen_y + 0.5) / focal], dim=-1),
        (0, 1),
        value=1.0,
    )
    c2w = torch.inverse(camera.world_view_transform.T)
    return -(
        (torch.nn.functional.normalize(canonical[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])
        .sum(dim=-1)
        .reshape(H, W, 3)
    )


def render_pbr_showcase(
    ply_path: str,
    output_dir: str = "showcase_frames",
    num_frames: int = 120,
    resolution: int = 512,
    env_name: str = "studio",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    center = gaussians.get_xyz.mean(dim=0)
    radius = torch.norm(gaussians.get_xyz - center, dim=-1).max().item() * 2.5

    light = CubemapLight.from_preset(env_name, resolution=128, device=device)
    light.build_mips()
    brdf_lut = get_brdf_lut()

    for frame_idx in tqdm(range(num_frames), desc="Rendering showcase"):
        angle = (frame_idx / max(num_frames, 1)) * 2.0 * math.pi
        camera = make_orbit_camera(center, radius, angle, math.radians(20.0), resolution, device)
        render_pkg = render(camera, gaussians, bg_color=torch.zeros(3, dtype=torch.float32, device=device))
        pbr_result = pbr_shading(
            light=light,
            normals=render_pkg["normal_map"].permute(1, 2, 0),
            view_dirs=get_view_dirs(camera),
            albedo=render_pkg["albedo_map"].permute(1, 2, 0),
            roughness=render_pkg["roughness_map"].permute(1, 2, 0),
            metallic=render_pkg["metallic_map"].permute(1, 2, 0),
            mask=render_pkg["normal_mask"].permute(1, 2, 0),
            brdf_lut=brdf_lut,
        )
        image = pbr_result["render_rgb"].permute(2, 0, 1)
        image = image * render_pkg["opacity_map"] + (1.0 - render_pkg["opacity_map"])
        save_tensor_image(image, os.path.join(output_dir, f"frame_{frame_idx:03d}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ply", type=str, default="pbr_outputs/final_pbr_model.ply")
    parser.add_argument("--output_dir", type=str, default="showcase_frames")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--env_name", type=str, default="studio")
    args = parser.parse_args()

    render_pbr_showcase(
        ply_path=args.ply,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        resolution=args.resolution,
        env_name=args.env_name,
    )
