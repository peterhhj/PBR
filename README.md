# 1.baseColor，材质的基础颜色，即反射率(albedo)，可以是常数或者由纹理提供。
2.subsurface，次表面散射系数，用于控制材质的漫反射项向次表面散射靠拢的程度，默认值为0。
# 3.metallic，金属度，用于控制材质的外观向金属靠拢的程度，默认值为0。
3.specular，镜面度，用于控制非金属微表面镜面反射项的大小，由菲涅尔项进行插值，默认值为0.5。
4.specularTint，用于控制镜面反射光的颜色向基本颜色靠拢的程度，越小镜面光表现为白色，默认值为0。
# 5.roughness，用于控制材质表面的粗糙程度，默认值为0.5。
6.anisotropic，各向异性，用于控制材质镜面反射的非对称程度，默认值为0。
7.sheen，模拟纺织物边缘的明亮效果，即绒毛效果，默认值为0，
8.sheenTint，用于控制sheen分量颜色向基本颜色靠拢的程度，默认值为0.5。
9.clearcoat，模拟清漆的效果，类似于镀了一层膜的效果，默认值为0，
10.clearcoatGloss，用于控制清漆的光滑程度，默认值为1


unset OMP_PLACES
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export KMP_AFFINITY=disabled
export GOMP_CPU_AFFINITY=""

obj engine_high
psnr 39.69

obj engine_low
psnr 


# step1
python generate_pseudo_targets.py \
  --ply_path /home/gaussian-splatting/output/armadillo/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_armadillo_1110 \
  --style_image /home/PBR/wood2.jpg \
  --material_preset wood \
  --output_dir pseudo_targets_wood_armadillo \
  --texture_scale 3.0

python generate_pseudo_targets.py \
  --ply_path /home/gaussian-splatting/output/white_bunny/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/PBR/ceramic.png \
  --material_preset ceramic \
  --output_dir pseudo_targets_ceramic_bunny \
  --texture_scale 3.0

# step2
python train_pbr.py \
  --ply_path /home/gaussian-splatting/output/armadillo/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_armadillo_1110 \
  --style_image /home/PBR/wood2.jpg \
  --material_preset wood \
  --material_optimization_mode gcn \
  --direct_material_lr 0.05 \
  --warmup_iters 300 \
  --views_per_iter 2 \
  --perceptual_views_per_iter 1 \
  --anchor_weight 0.03 \
  --style_resolution 192 \
  --max_envs_per_iter 1 \
  --pseudo_target_dir /home/PBR/pseudo_targets_wood_armadillo \
  --pseudo_base_weight 0.22 \
  --pseudo_diffuse_weight 0.05 \
  --pseudo_final_weight 0.10 \
  --pseudo_material_weight 0.08 \
  --output_dir result_armadillo_gcn



python train_pbr.py \
  --ply_path /home/gaussian-splatting/output/engine_high/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/engine_high_1110 \
  --style_image /home/PBR/metal.jpg \
  --material_preset metal \
  --warmup_iters 300 \
  --views_per_iter 4 \
  --perceptual_views_per_iter 1 \
  --anchor_weight 0.05 \
  --style_resolution 224 \
  --max_envs_per_iter 1 \
  --pseudo_target_dir /home/PBR/pseudo_targets_metal \
  --pseudo_base_weight 0.08 \
  --pseudo_diffuse_weight 0.02 \
  --pseudo_final_weight 0.22 \
  --pseudo_material_weight 0.10 \
  --output_dir result_engine_high_new


python train_pbr.py \
  --ply_path /home/gaussian-splatting/output/white_bunny/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/PBR/wood3.png \
  --material_preset wood \
  --warmup_iters 300 \
  --views_per_iter 4 \
  --perceptual_views_per_iter 1 \
  --anchor_weight 0.05 \
  --style_resolution 896 \
  --max_envs_per_iter 1 \
  --output_dir result_54


python train_pbr_gcn.py \
  --ply_path /home/gaussian-splatting/output/white_bunny/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/PBR/metal.jpg \
  --material_preset metal \
  --material_optimization_mode gcn \
  --warmup_iters 300 \
  --views_per_iter 1 \
  --perceptual_views_per_iter 1 \
  --anchor_weight 0.03 \
  --style_resolution 160 \
  --max_envs_per_iter 1 \
  --pseudo_target_dir /home/PBR/pseudo_targets_metal \
  --pseudo_base_weight 0.08 \
  --pseudo_diffuse_weight 0.02 \
  --pseudo_final_weight 0.22 \
  --pseudo_material_weight 0.10 \
  --graph_conv_chunk_size 512 \
  --gcn_subgraph_size 4096 \
  --gcn_seed_size 1024 \
  --gcn_expand_hops 1 \
  --output_dir result_bunny_metal_gcn_subgraph


# step3
python render_pbr_view.py \
  --ply /home/PBR/result_engine_mid_better_wo/final_pbr_model.ply \
  --source /home/generate_data/Data/engine_mid_1110 \
  --output_dir demo_views_engine_mid_gcn_wo \
  --env_name studio \
  --demo_views


python render_pbr_view.py \
  --ply /home/PBR/result_bunny_metal_fusednorm_v2/final_pbr_model.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --output_dir demo_result_bunny_metal_gcn_fuse_v2 \
  --env_name studio \
  --demo_views

# step4 only final
python render_pbr_all_final.py \
  --ply /home/PBR/result_engine_high_better/final_pbr_model.ply \
  --source /home/generate_data/Data/engine_high_1110 \
  --output_dir all_final_ceramic \
  --env_name studio


studio：最适合展示，层次和高光都比较清楚
neutral：更平，适合做诊断
warm：偏暖，木材看起来通常更讨喜
rim：边缘高光更强，金属常常更有表现力

# PBR Material Transfer on Frozen 3DGS

This project focuses on stage 2 of a 3DGS-based inverse rendering workflow:

- Stage 1: reconstruct a scan object with standard 3DGS and export a fixed `ply`
- Stage 2: freeze geometry and transfer a target material from a single reference image by optimizing
  - `baseColor`
  - `roughness`
  - `metallic`

The current implementation follows a GS-IR-style design:

- full G-buffer rendering from gaussians
- environment-map PBR shading
- 3D-GCN material prediction on frozen gaussian points
- joint optimization with style loss, structural losses, and material priors

## Project Structure

`scene/`

- [scene/gaussian_model.py](/C:/Users/98798/Desktop/new/PBR/scene/gaussian_model.py): loads frozen gaussian geometry from `ply`, keeps geometry fixed, stores and saves PBR attributes
- [scene/cameras.py](/C:/Users/98798/Desktop/new/PBR/scene/cameras.py): camera definitions used by training and rendering
- `dataset_readers.py`: COLMAP / Blender dataset readers

`gaussian_renderer/`

- [gaussian_renderer/__init__.py](/C:/Users/98798/Desktop/new/PBR/gaussian_renderer/__init__.py): GS-IR-style renderer, returns `opacity/depth/normal/baseColor/roughness/metallic`

`pbr/`

- [pbr/light.py](/C:/Users/98798/Desktop/new/PBR/pbr/light.py): procedural environment light presets and mip hierarchy
- [pbr/shade.py](/C:/Users/98798/Desktop/new/PBR/pbr/shade.py): environment-map PBR shading
- [pbr/brdf_256_256.bin](/C:/Users/98798/Desktop/new/PBR/pbr/brdf_256_256.bin): BRDF lookup table

`pbr_modules/`

- [pbr_modules/predictor.py](/C:/Users/98798/Desktop/new/PBR/pbr_modules/predictor.py): 3D-GCN material predictor conditioned on a global style code
- [pbr_modules/style_loss.py](/C:/Users/98798/Desktop/new/PBR/pbr_modules/style_loss.py): VGG style encoder and perceptual losses
- [pbr_modules/brdf_renderer.py](/C:/Users/98798/Desktop/new/PBR/pbr_modules/brdf_renderer.py): compatibility wrapper around the PBR shader

Top-level scripts

- [train_pbr.py](/C:/Users/98798/Desktop/new/PBR/train_pbr.py): main training script
- [render_showcase.py](/C:/Users/98798/Desktop/new/PBR/render_showcase.py): orbit rendering for quick visual inspection

## Training Logic

The training loop in [train_pbr.py](/C:/Users/98798/Desktop/new/PBR/train_pbr.py) has two phases.

1. Warm-up phase

- controlled by `--warmup_iters`
- mainly keeps structure stable
- emphasizes normal consistency, TV smoothness, KNN smoothness, source-content anchoring, and material priors
- style transfer is intentionally weak or disabled here

2. Transfer phase

- starts after `warmup_iters`
- gradually increases style loss on `baseColor/diffuse/final render`
- uses multiple environment presets so the network has a better chance to infer glossiness and reflectance

The geometry is frozen during stage 2:

- `xyz`
- `scaling`
- `rotation`
- `opacity`
- SH color features

Only the material fields are optimized through the predictor:

- `baseColor`
- `roughness`
- `metallic`

## Main Loss Terms

The current optimization includes:

- style loss on `baseColor` and diffuse appearance
- style loss on final PBR render
- object-only crop for style loss, using `opacity_map` to remove white-background dilution
- direct RGB distribution supervision on `baseColor`
- content anchor against original multi-view appearance
- normal consistency between rasterized normal and depth-derived normal
- masked TV loss on material maps
- KNN smoothness on gaussian material values
- material priors based on `--material_preset`
- optional pseudo-image supervision from offline generated `baseColor/diffuse/final/mask`
- optional 3D pseudo-material supervision from `pseudo_material_3d.npz`
- firefly suppression on overly bright specular response

## Material Presets

Supported presets:

- `jade`
- `wood`
- `metal`

These presets affect the target tendency of:

- metallicness
- roughness range
- allowed baseColor saturation

For example:

- `wood`: low metallic, medium-to-high roughness
- `jade`: low metallic, smoother than wood
- `metal`: high metallic, lower roughness allowed

## Run Training

Example:

```bash
python train_pbr.py \
  --ply_path /home/new1/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/textures/images/12.jpg \
  --material_preset wood \
  --warmup_iters 800 \
  --views_per_iter 1
```

Useful arguments:

- `--iterations`: total training iterations, default `3000`
- `--warmup_iters`: number of iterations used for structure-preserving warm-up, default `800`
- `--views_per_iter`: number of training views sampled each iteration, default `1`
- `--material_preset`: `jade | wood | metal`
- `--env_mode`: environment preset selection, default `default`
- `--anchor_weight`: strength of source-content anchoring, default `0.15`
- `--style_crop_padding`: padding around the soft object crop used by style loss, default `20`
- `--color_distribution_weight`: weight for direct `baseColor` RGB statistic matching, default `0.25`
- `--pseudo_target_dir`: optional folder containing offline pseudo targets
- `--pseudo_image_weight`: weight for pseudo-image supervision, default `0.35`
- `--pseudo_material_weight`: weight for 3D pseudo-material supervision, default `0.2`
- `--output_dir`: output folder, default `pbr_outputs`

## Why 3000 Iterations When Warm-up Is 800

- `--iterations` controls the total number of optimization steps
- `--warmup_iters` only controls how many of those steps belong to the warm-up phase

So with:

- `--iterations 3000`
- `--warmup_iters 800`

the schedule is:

- iter `1` to `800`: warm-up
- iter `801` to `3000`: transfer

## Output Directory

The training script writes into `pbr_outputs/` by default.

Important outputs:

- `final_pbr_model.ply`: final gaussian model with optimized material parameters
- `run_config.json`: saved run configuration
- `debug/iter_xxxxx_baseColor.png`: estimated intrinsic base color
- `debug/iter_xxxxx_roughness.png`: roughness map, grayscale
- `debug/iter_xxxxx_metallic.png`: metallic map, grayscale
- `debug/iter_xxxxx_normal.png`: normal map visualization
- `debug/iter_xxxxx_opacity.png`: gaussian opacity / soft silhouette
- `debug/iter_xxxxx_diffuse.png`: diffuse-only appearance under neutral environment
- `debug/iter_xxxxx_specular.png`: specular-only appearance under neutral environment
- `debug/iter_xxxxx_final.png`: final PBR composition under neutral environment

## Meaning of Debug Images

`baseColor`

- the intrinsic reflectance color of the material
- ideally this should move toward the dominant wood / jade / metal color family
- it should not contain strong view-dependent highlights

`diffuse`

- diffuse-only shading result
- useful for checking whether color and coarse texture are moving toward the reference material

`final`

- final PBR render = diffuse + specular
- this is the image that should show gloss, highlights, and overall material feel

`metallic`

- grayscale map of metallicness
- dark means dielectric / non-metal
- bright means metallic

`normal`

- surface normal visualization
- useful for checking whether the object still preserves volume and shape cues

`opacity`

- soft silhouette / transparency accumulation from 3DGS
- used for smooth compositing instead of a hard binary mask

## Typical Issues

1. `baseColor` stays close to the original object color

- this usually means style loss is still too weak compared with content anchoring
- check terminal values: if `Sty` is tiny but `Anchor` is much larger, color transfer will be slow

2. `final` looks dark

- common causes:
  - the predicted `baseColor` is still dark or unchanged
  - `roughness` is too high, causing weak specular response
  - the current environment preset is not strong enough for the material you want

3. Material looks structurally correct but color transfer is weak

- try lowering `--anchor_weight`
- or reduce `--warmup_iters`
- or train longer

4. Training is slow

- KNN graph construction for ~100k gaussians is CPU-heavy
- the main loop also renders full G-buffer plus multiple PBR passes

5. Material color still does not move enough

- check whether `Sty` and `Clr` are both much smaller than `Anchor`
- try the masked-style setup together with pseudo targets instead of only reducing `anchor_weight`

## Pseudo Targets

The repository now supports an offline pseudo-supervision stage.

The idea is:

1. build a 3D-consistent pseudo material field directly on the frozen gaussians
2. render that field from the COLMAP training cameras
3. feed the rendered `baseColor/diffuse/final/mask` images back into training
4. optionally supervise the per-gaussian `baseColor/roughness/metallic` directly with the saved `pseudo_material_3d.npz`

This is important because the pseudo signal is created in 3D first, so all derived 2D views are naturally multi-view consistent.

### Generate Pseudo Targets

Use [generate_pseudo_targets.py](/C:/Users/98798/Desktop/new/PBR/generate_pseudo_targets.py):

```bash
python generate_pseudo_targets.py \
  --ply_path /home/new1/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/textures/images/12.jpg \
  --material_preset wood \
  --output_dir pseudo_targets_wood \
  --texture_scale 3.0 \
  --env_name neutral
```

What this script does:

- loads the frozen gaussian cloud
- samples the style image onto 3D gaussians with a triplanar mapping
- converts the sampled texture into a pseudo `baseColor`
- derives pseudo `roughness/metallic` from the selected material preset
- renders all COLMAP training views and saves:
  - `baseColor/*.png`
  - `diffuse/*.png`
  - `final/*.png`
  - `mask/*.png`
- saves `pseudo_material_3d.npz` for direct 3D supervision
- saves `pseudo_material_preview.ply` for inspection

### Train With Pseudo Targets

```bash
python train_pbr.py \
  --ply_path /home/new1/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/textures/images/12.jpg \
  --material_preset wood \
  --warmup_iters 300 \
  --views_per_iter 4 \
  --anchor_weight 0.05 \
  --style_resolution 256 \
  --max_envs_per_iter 1 \
  --pseudo_target_dir pseudo_targets_wood
```

With `--pseudo_target_dir` set, training will:

- compute style loss only on the cropped rabbit region
- add direct color-distribution supervision on `baseColor`
- add masked pseudo-image losses against the offline rendered targets
- add direct 3D material losses against `pseudo_material_3d.npz` when available

## Suggested Tuning

If wood color is not transferring well, try:

```bash
python train_pbr.py \
  --ply_path /home/new1/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/textures/images/12.jpg \
  --material_preset wood \
  --warmup_iters 400 \
  --anchor_weight 0.08 \
  --views_per_iter 2 \
  --style_crop_padding 24 \
  --color_distribution_weight 0.3 \
  --iterations 4000
```

Why this often helps:

- shorter warm-up lets style transfer start earlier
- smaller anchor weight reduces the pull toward the original blue appearance
- more views per iteration gives stronger multi-view supervision
- longer total iterations gives style loss more time to take effect

## Showcase Rendering

After training:

```bash
python render_showcase.py --ply pbr_outputs/final_pbr_model.ply
```

This script generates orbit-view renderings for quick inspection of:

- highlight continuity
- silhouette smoothness
- overall material appearance

## Dependency Reminder

This project expects:

- a GS-IR-compatible `diff_gaussian_rasterization`
- `nvdiffrast`
- `torchvision`
- `plyfile`
- `imageio`
- `opencv-python`
- `tqdm`

If you use the standard 3DGS rasterizer, the renderer will not be able to produce the full PBR G-buffer required by this stage.



# PBR Material Transfer on Frozen 3DGS

This project focuses on stage 2 of a 3DGS-based inverse rendering workflow:

- Stage 1: reconstruct a scan object with standard 3DGS and export a fixed `ply`
- Stage 2: freeze geometry and transfer a target material from a single reference image by optimizing
  - `baseColor`
  - `roughness`
  - `metallic`

The current implementation follows a GS-IR-style design:

- full G-buffer rendering from gaussians
- environment-map PBR shading
- 3D-GCN material prediction on frozen gaussian points
- joint optimization with style loss, structural losses, and material priors

## Project Structure

`scene/`

- [scene/gaussian_model.py](/C:/Users/98798/Desktop/new/PBR/scene/gaussian_model.py): loads frozen gaussian geometry from `ply`, keeps geometry fixed, stores and saves PBR attributes
- [scene/cameras.py](/C:/Users/98798/Desktop/new/PBR/scene/cameras.py): camera definitions used by training and rendering
- `dataset_readers.py`: COLMAP / Blender dataset readers

`gaussian_renderer/`

- [gaussian_renderer/__init__.py](/C:/Users/98798/Desktop/new/PBR/gaussian_renderer/__init__.py): GS-IR-style renderer, returns `opacity/depth/normal/baseColor/roughness/metallic`

`pbr/`

- [pbr/light.py](/C:/Users/98798/Desktop/new/PBR/pbr/light.py): procedural environment light presets and mip hierarchy
- [pbr/shade.py](/C:/Users/98798/Desktop/new/PBR/pbr/shade.py): environment-map PBR shading
- [pbr/brdf_256_256.bin](/C:/Users/98798/Desktop/new/PBR/pbr/brdf_256_256.bin): BRDF lookup table

`pbr_modules/`

- [pbr_modules/predictor.py](/C:/Users/98798/Desktop/new/PBR/pbr_modules/predictor.py): 3D-GCN material predictor conditioned on a global style code
- [pbr_modules/style_loss.py](/C:/Users/98798/Desktop/new/PBR/pbr_modules/style_loss.py): VGG style encoder and perceptual losses
- [pbr_modules/brdf_renderer.py](/C:/Users/98798/Desktop/new/PBR/pbr_modules/brdf_renderer.py): compatibility wrapper around the PBR shader

Top-level scripts

- [train_pbr.py](/C:/Users/98798/Desktop/new/PBR/train_pbr.py): main training script
- [render_showcase.py](/C:/Users/98798/Desktop/new/PBR/render_showcase.py): orbit rendering for quick visual inspection

## Training Logic

The training loop in [train_pbr.py](/C:/Users/98798/Desktop/new/PBR/train_pbr.py) has two phases.

1. Warm-up phase

- controlled by `--warmup_iters`
- mainly keeps structure stable
- emphasizes normal consistency, TV smoothness, KNN smoothness, source-content anchoring, and material priors
- style transfer is intentionally weak or disabled here

2. Transfer phase

- starts after `warmup_iters`
- gradually increases style loss on `baseColor/diffuse/final render`
- uses multiple environment presets so the network has a better chance to infer glossiness and reflectance

The geometry is frozen during stage 2:

- `xyz`
- `scaling`
- `rotation`
- `opacity`
- SH color features

Only the material fields are optimized through the predictor:

- `baseColor`
- `roughness`
- `metallic`

## Main Loss Terms

The current optimization includes:

- style loss on `baseColor` and diffuse appearance
- style loss on final PBR render
- content anchor against original multi-view appearance
- normal consistency between rasterized normal and depth-derived normal
- masked TV loss on material maps
- KNN smoothness on gaussian material values
- material priors based on `--material_preset`
- firefly suppression on overly bright specular response

## Material Presets

Supported presets:

- `jade`
- `wood`
- `metal`

These presets affect the target tendency of:

- metallicness
- roughness range
- allowed baseColor saturation

For example:

- `wood`: low metallic, medium-to-high roughness
- `jade`: low metallic, smoother than wood
- `metal`: high metallic, lower roughness allowed

## Run Training

Example:

```bash
python train_pbr.py \
  --ply_path /home/new1/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/textures/images/12.jpg \
  --material_preset wood \
  --warmup_iters 800 \
  --views_per_iter 1
```

Useful arguments:

- `--iterations`: total training iterations, default `3000`
- `--warmup_iters`: number of iterations used for structure-preserving warm-up, default `800`
- `--views_per_iter`: number of training views sampled each iteration, default `1`
- `--material_preset`: `jade | wood | metal`
- `--env_mode`: environment preset selection, default `default`
- `--anchor_weight`: strength of source-content anchoring, default `0.15`
- `--output_dir`: output folder, default `pbr_outputs`

## Why 3000 Iterations When Warm-up Is 800

- `--iterations` controls the total number of optimization steps
- `--warmup_iters` only controls how many of those steps belong to the warm-up phase

So with:

- `--iterations 3000`
- `--warmup_iters 800`

the schedule is:

- iter `1` to `800`: warm-up
- iter `801` to `3000`: transfer

## Output Directory

The training script writes into `pbr_outputs/` by default.

Important outputs:

- `final_pbr_model.ply`: final gaussian model with optimized material parameters
- `run_config.json`: saved run configuration
- `debug/iter_xxxxx_baseColor.png`: estimated intrinsic base color
- `debug/iter_xxxxx_roughness.png`: roughness map, grayscale
- `debug/iter_xxxxx_metallic.png`: metallic map, grayscale
- `debug/iter_xxxxx_normal.png`: normal map visualization
- `debug/iter_xxxxx_opacity.png`: gaussian opacity / soft silhouette
- `debug/iter_xxxxx_diffuse.png`: diffuse-only appearance under neutral environment
- `debug/iter_xxxxx_specular.png`: specular-only appearance under neutral environment
- `debug/iter_xxxxx_final.png`: final PBR composition under neutral environment

## Meaning of Debug Images

`baseColor`

- the intrinsic reflectance color of the material
- ideally this should move toward the dominant wood / jade / metal color family
- it should not contain strong view-dependent highlights

`diffuse`

- diffuse-only shading result
- useful for checking whether color and coarse texture are moving toward the reference material

`final`

- final PBR render = diffuse + specular
- this is the image that should show gloss, highlights, and overall material feel

`metallic`

- grayscale map of metallicness
- dark means dielectric / non-metal
- bright means metallic

`normal`

- surface normal visualization
- useful for checking whether the object still preserves volume and shape cues

`opacity`

- soft silhouette / transparency accumulation from 3DGS
- used for smooth compositing instead of a hard binary mask

## Typical Issues

1. `baseColor` stays close to the original object color

- this usually means style loss is still too weak compared with content anchoring
- check terminal values: if `Sty` is tiny but `Anchor` is much larger, color transfer will be slow

2. `final` looks dark

- common causes:
  - the predicted `baseColor` is still dark or unchanged
  - `roughness` is too high, causing weak specular response
  - the current environment preset is not strong enough for the material you want

3. Material looks structurally correct but color transfer is weak

- try lowering `--anchor_weight`
- or reduce `--warmup_iters`
- or train longer

4. Training is slow

- KNN graph construction for ~100k gaussians is CPU-heavy
- the main loop also renders full G-buffer plus multiple PBR passes

## Suggested Tuning

If wood color is not transferring well, try:

```bash
python train_pbr.py \
  --ply_path /home/new1/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/textures/images/12.jpg \
  --material_preset wood \
  --warmup_iters 400 \
  --anchor_weight 0.08 \
  --views_per_iter 2 \
  --iterations 4000
```

Why this often helps:

- shorter warm-up lets style transfer start earlier
- smaller anchor weight reduces the pull toward the original blue appearance
- more views per iteration gives stronger multi-view supervision
- longer total iterations gives style loss more time to take effect

## Showcase Rendering

After training:

```bash
python render_showcase.py --ply pbr_outputs/final_pbr_model.ply
```

This script generates orbit-view renderings for quick inspection of:

- highlight continuity
- silhouette smoothness
- overall material appearance

## Dependency Reminder

This project expects:

- a GS-IR-compatible `diff_gaussian_rasterization`
- `nvdiffrast`
- `torchvision`
- `plyfile`
- `imageio`
- `opencv-python`
- `tqdm`

If you use the standard 3DGS rasterizer, the renderer will not be able to produce the full PBR G-buffer required by this stage.
