import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for candidate in (THIS_DIR, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from core import compute_vertex_normals
from core import load_triangle_mesh
from core import safe_normalize
from vti_to_mesh import convert_vti_to_mesh
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.graphics_utils import getProjectionMatrix
from utils.graphics_utils import getWorld2View2


def _load_nvdiffrast():
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError(
            "nvdiffrast is required for mesh GT normal rendering. "
            "Install the GS-IR-compatible nvdiffrast package first."
        ) from exc
    return dr


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_uint8_image(image: np.ndarray) -> Image.Image:
    image = np.clip(image, 0.0, 1.0)
    return Image.fromarray(np.round(image * 255.0).astype(np.uint8))


def _save_rgb(path: str, image: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    _to_uint8_image(image).save(path)


def _save_mask(path: str, mask: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    Image.fromarray(np.round(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def load_camera_infos(source_path: str):
    is_synthetic = os.path.exists(os.path.join(source_path, "transforms_train.json"))
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", eval=False, train_test_exp=False, depths="")
    elif is_synthetic:
        scene_info = sceneLoadTypeCallbacks["Blender"](source_path, True, "", False)
    else:
        raise RuntimeError(f"Could not infer scene type from {source_path}")
    return scene_info.train_cameras


def select_camera_infos(camera_infos, view_names: Optional[Iterable[str]] = None) -> List:
    if not view_names:
        return list(camera_infos)
    lookup = {
        Path(camera.image_name).stem: camera
        for camera in camera_infos
    }
    selected = []
    for view_name in view_names:
        normalized = Path(view_name).stem
        if normalized not in lookup:
            raise RuntimeError(f"Could not find camera named '{view_name}'.")
        selected.append(lookup[normalized])
    return selected


def make_camera_matrices(camera_info, device: torch.device):
    world_view = torch.tensor(
        getWorld2View2(camera_info.R, camera_info.T),
        dtype=torch.float32,
        device=device,
    ).transpose(0, 1).contiguous()
    projection = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=camera_info.FovX,
        fovY=camera_info.FovY,
    ).transpose(0, 1).to(device)
    return world_view, projection


def render_mesh_normal_for_camera(
    dr,
    glctx,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_normals: torch.Tensor,
    camera_info,
    device: torch.device,
    background_normal=(0.5, 0.5, 1.0),
) -> dict:
    height = int(camera_info.height)
    width = int(camera_info.width)

    world_view, projection = make_camera_matrices(camera_info, device)
    vertices_h = torch.cat(
        [vertices, torch.ones(vertices.shape[0], 1, dtype=vertices.dtype, device=device)],
        dim=1,
    )
    clip_positions = vertices_h @ world_view @ projection

    rast, _ = dr.rasterize(glctx, clip_positions.unsqueeze(0), faces.int(), resolution=[height, width])
    mask = rast[0, :, :, 3] > 0

    interpolated_normals, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast, faces.int())
    interpolated_normals = safe_normalize(interpolated_normals[0])

    background = torch.tensor(background_normal, dtype=torch.float32, device=device).view(1, 1, 3)
    normal_vis = (interpolated_normals + 1.0) * 0.5
    normal_vis = torch.where(mask.unsqueeze(-1), normal_vis, background)

    normal_on_white = torch.where(
        mask.unsqueeze(-1),
        normal_vis,
        torch.ones_like(normal_vis),
    )

    return {
        "normal": normal_vis.detach().cpu().numpy(),
        "normal_on_white": normal_on_white.detach().cpu().numpy(),
        "mask": mask.detach().cpu().numpy().astype(np.float32),
    }


def render_gt_normals_from_mesh(
    mesh_path: str,
    source_path: str,
    output_dir: str,
    view_names: Optional[Iterable[str]] = None,
) -> None:
    dr = _load_nvdiffrast()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script currently requires CUDA because nvdiffrast rasterization is GPU-based.")

    _ensure_dir(output_dir)
    camera_infos = select_camera_infos(load_camera_infos(source_path), view_names=view_names)

    mesh = load_triangle_mesh(mesh_path, device)
    if mesh.normals is None:
        mesh.normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    vertex_normals = safe_normalize(mesh.normals)

    glctx = dr.RasterizeCudaContext(device=device)

    rendered_names = []
    for camera_info in camera_infos:
        name = Path(camera_info.image_name).stem
        outputs = render_mesh_normal_for_camera(
            dr=dr,
            glctx=glctx,
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_normals=vertex_normals,
            camera_info=camera_info,
            device=device,
        )
        prefix = os.path.join(output_dir, name)
        _save_rgb(prefix + "_gt_normal.png", outputs["normal"])
        _save_rgb(prefix + "_gt_normal_on_white.png", outputs["normal_on_white"])
        _save_mask(prefix + "_gt_mask.png", outputs["mask"])
        with open(prefix + "_camera.txt", "w", encoding="utf-8") as handle:
            handle.write(f"camera_name={name}\n")
            handle.write(f"image_name={camera_info.image_name}\n")
            handle.write(f"width={camera_info.width}\n")
            handle.write(f"height={camera_info.height}\n")
            handle.write(f"fovx={camera_info.FovX}\n")
            handle.write(f"fovy={camera_info.FovY}\n")
            handle.write("R=\n")
            for row in np.asarray(camera_info.R):
                handle.write("  " + " ".join(f"{value:.8f}" for value in row) + "\n")
            handle.write("T=\n")
            handle.write("  " + " ".join(f"{value:.8f}" for value in np.asarray(camera_info.T)) + "\n")
        rendered_names.append(name)

    print(f"Rendered {len(rendered_names)} GT normal views to {output_dir}")
    if rendered_names:
        print("Views: " + ", ".join(rendered_names[:10]) + (" ..." if len(rendered_names) > 10 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a mesh from VTI and batch-render GT normal maps for all COLMAP/Blender views."
    )
    parser.add_argument("--vti_path", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True, help="Where to save the extracted mesh, e.g. mesh.ply")
    parser.add_argument("--source", type=str, required=True, help="COLMAP/Blender dataset root used for camera poses")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for GT normal images")
    parser.add_argument("--view_names", type=str, nargs="*", default=None, help="Optional subset of dataset view names")
    parser.add_argument("--reuse_mesh", action="store_true", help="Skip VTI extraction if --mesh_path already exists")
    parser.add_argument("--scalar_name", type=str, default=None)
    parser.add_argument("--iso_value", type=float, default=None)
    parser.add_argument("--iso_percentile", type=float, default=50.0)
    parser.add_argument("--no_largest_component", dest="largest_component", action="store_false")
    parser.set_defaults(largest_component=True)
    parser.add_argument("--fill_holes", type=float, default=0.0)
    parser.add_argument("--smooth_iters", type=int, default=30)
    parser.add_argument("--relaxation_factor", type=float, default=0.01)
    parser.add_argument("--feature_smoothing", action="store_true")
    parser.add_argument("--boundary_smoothing", action="store_true")
    parser.add_argument("--decimate_ratio", type=float, default=0.0)
    parser.add_argument("--no_recompute_normals", dest="recompute_normals", action="store_false")
    parser.set_defaults(recompute_normals=True)
    args = parser.parse_args()

    if not (args.reuse_mesh and os.path.exists(args.mesh_path)):
        points, faces, used_iso, used_scalar = convert_vti_to_mesh(
            args.vti_path,
            args.mesh_path,
            scalar_name=args.scalar_name,
            iso_value=args.iso_value,
            iso_percentile=args.iso_percentile,
            largest_component=args.largest_component,
            fill_holes=args.fill_holes,
            smooth_iters=args.smooth_iters,
            relaxation_factor=args.relaxation_factor,
            feature_smoothing=args.feature_smoothing,
            boundary_smoothing=args.boundary_smoothing,
            decimate_ratio=args.decimate_ratio,
            recompute_normals=args.recompute_normals,
        )
        print(f"Extracted mesh from {args.vti_path}")
        print(f"Scalar field: {used_scalar}")
        print(f"Iso value: {used_iso:.6f}")
        print(f"Mesh vertices: {points}")
        print(f"Mesh faces: {faces}")
        print(f"Saved mesh to: {args.mesh_path}")
    else:
        print(f"Reusing existing mesh: {args.mesh_path}")

    render_gt_normals_from_mesh(
        mesh_path=args.mesh_path,
        source_path=args.source,
        output_dir=args.output_dir,
        view_names=args.view_names,
    )


if __name__ == "__main__":
    main()
