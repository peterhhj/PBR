import argparse
import os

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from mgadaptor_tools.core import MGAdapter, load_triangle_mesh, save_guidance_npz, save_guidance_preview_ply, subsample_mesh_faces
else:
    from .core import MGAdapter, load_triangle_mesh, save_guidance_npz, save_guidance_preview_ply, subsample_mesh_faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MGAdaptor geometric guidance splats from a mesh.")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--face_stride", type=int, default=1)
    parser.add_argument("--disable_normal_interpolation", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    mesh = load_triangle_mesh(args.mesh_path, device=device)
    mesh = subsample_mesh_faces(mesh, args.face_stride)

    adapter = MGAdapter()
    guide_splats, offsets = adapter.make(mesh, normal_interpolation=not args.disable_normal_interpolation)

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, "mgadapter_guidance.npz")
    preview_path = os.path.join(args.output_dir, "mgadapter_guidance_preview.ply")
    save_guidance_npz(npz_path, guide_splats, offsets)
    save_guidance_preview_ply(preview_path, guide_splats)

    print(f"Loaded mesh with {mesh.num_vertices} vertices and {mesh.num_faces} faces.")
    print(f"Built {guide_splats.means.shape[0]} MGAdaptor guidance splats.")
    print(f"Saved guidance package to {npz_path}")
    print(f"Saved preview ply to {preview_path}")


if __name__ == "__main__":
    main()
