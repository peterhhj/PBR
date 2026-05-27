# MGAdaptor Tools

This folder contains a low-intrusion MGAdaptor pipeline adapted from GeoSplatting for the PBR project.

## Goal

Use a triangle mesh to build geometry-guided splats and transfer their normals onto an existing Gaussian `ply`.
The existing PBR code will then automatically read `normal_0/1/2` from the patched Gaussian `ply`.

## Scripts

### 1. Extract a mesh from a VTI volume

```bash
python vti_to_mesh.py \
  --vti_path /home/generate_data/vti/teapot.vti \
  --output_mesh /home/PBR/mesh/teapot.ply \
  --iso_value 0 \
  --smooth_iters 30
```
python vti2mesh.py \
  --vti_path /home/generate_data/vti/teapot.vti \
  --output_mesh /home/PBR/mesh/teapot.ply \
  --iso_value 0 \
  --smooth_iters 30

Common notes:

- If you already know the desired iso value, prefer `--iso_value`.
- If not, start from `--iso_percentile 50` and then try `40 / 60 / 70`.
- The output mesh can be `.ply` or `.obj`.

### 2. Build guidance splats

```bash
python mgadaptor_tools/build_guidance.py \
  --mesh_path /home/PBR/mesh/teapot.ply \
  --output_dir /home/PBR/mesh_guidance/teapot \
  --face_stride 1
```

Outputs:

- `mgadapter_guidance.npz`
- `mgadapter_guidance_preview.ply`

### 3. Patch a Gaussian ply with MGAdaptor normals

If you already built the guidance package:
export KMP_AFFINITY=disabled
```bash
python mgadaptor_tools/apply_normals.py \
  --gaussian_ply /home/gaussian-splatting/output/white_bunny/point_cloud/iteration_30000/point_cloud.ply \
  --guidance_npz /home/PBR/mesh_guidance/bunny/mgadapter_guidance.npz \
  --output_ply /home/PBR/pbr_modules/guided_ply/bunny/point_cloud_mg_normals.ply \
  --k 8
```

Or build guidance from the mesh on the fly:

```bash
python mgadaptor_tools/apply_normals.py \
  --gaussian_ply /path/to/point_cloud.ply \
  --mesh_path /path/to/mesh.obj \
  --output_ply /path/to/point_cloud_mg_normals.ply \
  --guidance_output_dir /path/to/mg_guidance \
  --k 8
```

## Recommended use in the current project

1. Start from your `vti` volume and extract a triangle mesh with `vti_to_mesh.py`.
2. Build MGAdaptor guidance splats from the mesh.
3. Patch the frozen Gaussian `ply` with the transferred normals.
4. Train/render with the patched `ply` instead of the original one.

Because `scene/gaussian_model.py` already checks for `normal_0/1/2`, no further modification to the current training code is required.
