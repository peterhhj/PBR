import argparse
import os

import numpy as np
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from mgadaptor_tools.core import (
        MGAdapter,
        get_existing_normals,
        get_gaussian_positions,
        load_gaussian_vertex_data,
        load_guidance_npz,
        load_triangle_mesh,
        save_guidance_npz,
        save_guidance_preview_ply,
        subsample_mesh_faces,
        transfer_normals_from_guidance,
        update_vertex_fields,
        write_gaussian_ply,
    )
else:
    from .core import (
        MGAdapter,
        get_existing_normals,
        get_gaussian_positions,
        load_gaussian_vertex_data,
        load_guidance_npz,
        load_triangle_mesh,
        save_guidance_npz,
        save_guidance_preview_ply,
        subsample_mesh_faces,
        transfer_normals_from_guidance,
        update_vertex_fields,
        write_gaussian_ply,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer MGAdaptor-guided normals onto a Gaussian PLY.")
    parser.add_argument("--gaussian_ply", type=str, required=True)
    parser.add_argument("--output_ply", type=str, required=True)
    parser.add_argument("--guidance_npz", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--guidance_output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--face_stride", type=int, default=1)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--query_chunk_size", type=int, default=8192)
    parser.add_argument("--knn_backend", type=str, default="scipy", choices=["scipy", "torch"])
    parser.add_argument("--scipy_workers", type=int, default=1)
    parser.add_argument("--blend_existing", type=float, default=0.0)
    parser.add_argument("--disable_normal_interpolation", action="store_true")
    args = parser.parse_args()

    if args.guidance_npz is None and args.mesh_path is None:
        raise ValueError("Either --guidance_npz or --mesh_path must be provided.")

    device = torch.device(args.device)
    if args.guidance_npz is not None:
        guide_splats, _ = load_guidance_npz(args.guidance_npz, device=device)
    else:
        mesh = load_triangle_mesh(args.mesh_path, device=device)
        mesh = subsample_mesh_faces(mesh, args.face_stride)
        adapter = MGAdapter()
        guide_splats, offsets = adapter.make(mesh, normal_interpolation=not args.disable_normal_interpolation)
        if args.guidance_output_dir is not None:
            os.makedirs(args.guidance_output_dir, exist_ok=True)
            save_guidance_npz(os.path.join(args.guidance_output_dir, "mgadapter_guidance.npz"), guide_splats, offsets)
            save_guidance_preview_ply(
                os.path.join(args.guidance_output_dir, "mgadapter_guidance_preview.ply"),
                guide_splats,
            )

    ply, vertex_data = load_gaussian_vertex_data(args.gaussian_ply)
    positions = get_gaussian_positions(vertex_data)
    positions_t = torch.from_numpy(positions).to(device=device, dtype=torch.float32)
    guided_normals = transfer_normals_from_guidance(
        positions_t,
        guide_splats.means,
        guide_splats.normals,
        k=args.k,
        query_chunk_size=args.query_chunk_size,
        knn_backend=args.knn_backend,
        scipy_workers=args.scipy_workers,
    )
    guided_normals_np = guided_normals.detach().cpu().numpy().astype(np.float32)

    blend = float(np.clip(args.blend_existing, 0.0, 1.0))
    existing = get_existing_normals(vertex_data)
    if existing is not None and blend > 0.0:
        fused = guided_normals_np * (1.0 - blend) + existing.astype(np.float32) * blend
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        guided_normals_np = fused / np.clip(norms, 1e-6, None)

    updated = update_vertex_fields(
        vertex_data,
        {
            "normal_0": guided_normals_np[:, 0],
            "normal_1": guided_normals_np[:, 1],
            "normal_2": guided_normals_np[:, 2],
        },
    )
    write_gaussian_ply(args.output_ply, ply, updated)

    print(f"Transferred MGAdaptor guidance normals to {positions.shape[0]} gaussians.")
    print(f"Saved patched Gaussian ply to {args.output_ply}")


if __name__ == "__main__":
    main()
