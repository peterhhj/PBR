# 训练步骤
## version1记录
### step1 获得伪监督信号
python generate_pseudo_targets.py \
  --ply_path /home/gaussian-splatting/output/armadillo/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_armadillo_1110 \
  --style_image /home/PBR/wood2.jpg \
  --material_preset wood \
  --output_dir pseudo_targets_wood_armadillo \
  --texture_scale 3.0

### step2 训练预测PBR参数
#### train_pbr.py
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
#### train_pbr_gcn.py
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

### step3 渲染指定视角
python render_pbr_view.py \
  --ply /home/PBR/result_bunny_metal_fusednorm_v2/final_pbr_model.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --output_dir demo_result_bunny_metal_gcn_fuse_v2 \
  --env_name studio \
  --demo_views

### step4 only final
python render_pbr_all_final.py \
  --ply /home/PBR/result_engine_high_better/final_pbr_model.ply \
  --source /home/generate_data/Data/engine_high_1110 \
  --output_dir all_final_ceramic \
  --env_name studio


## version2记录
### stage1：先导出全部法线
python /home/PBR/render_normals_from_dataset.py \
  --ply /home/PBR/result_armadillo_gcn/final_pbr_model.ply \
  --source /home/generate_data/Data/colmap_armadillo_1110 \
  --output_dir /home/PBR/dataset_normals_armadillo \
  --all_views

### stage2：融合法线
python /home/PBR/render_normals_from_dataset.py \
  --ply /home/PBR/result_teapot_gcn/final_pbr_model.ply \
  --source /home/generate_data/Data/colmap_teapot_1110 \
  --output_dir /home/PBR/dataset_normals0.2_teapot \
  --view_names  \
  --fused_depth_weight 0.2

### stage3：筛选法线
python /home/PBR/prepare_external_normal_dirs.py \
  --input_dir /home/PBR/dataset_normals \
  --output_root /home/PBR/external_normals_small \
  --variants fused_normal

### stage4：训练预测PBR参数
python train_pbr_fused_normals.py \
  --ply_path /home/gaussian-splatting/output/teapot/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_teapot_1110 \
  --style_image /home/PBR/ceramic.png \
  --fused_normal_dir /home/PBR/dataset_normals0.2_teapot \
  --material_preset ceramic \
  --material_optimization_mode gcn \
  --warmup_iters 300 \
  --views_per_iter 1 \
  --perceptual_views_per_iter 1 \
  --anchor_weight 0.03 \
  --style_resolution 160 \
  --max_envs_per_iter 1 \
  --pseudo_target_dir /home/PBR/pseudo_targets_ceramic_teapot \
  --pseudo_base_weight 0.08 \
  --pseudo_diffuse_weight 0.02 \
  --pseudo_final_weight 0.22 \
  --pseudo_material_weight 0.10 \
  --graph_conv_chunk_size 512 \
  --gcn_subgraph_size 4096 \
  --gcn_seed_size 1024 \
  --gcn_expand_hops 1 \
  --external_normal_weight 1.0 \
  --output_dir result_teapot_fuse

## version3记录




## 存档内容
玻璃材质存档
(gsir) root@instance:/home/PBR# python train_pbr_fused_normals_plastic.py \
  --ply_path /home/gaussian-splatting/output/white_bunny/point_cloud/iteration_30000/point_cloud.ply \
  --source /home/generate_data/Data/colmap_bunny_1110 \
  --style_image /home/PBR/metal.jpg \
  --fused_normal_dir /home/PBR/dataset_normals \
  --material_preset metal \
  --material_optimization_mode gcn \
  --warmup_iters 300 \
  --views_per_iter 1 \
  --perceptual_views_per_iter 1 \
  --anchor_weight 0.02 \
  --style_resolution 160 \
  --max_envs_per_iter 1 \
  --env_mode studio \
  --pseudo_target_dir /home/PBR/pseudo_targets_metal \
  --pseudo_base_weight 0.06 \
  --pseudo_diffuse_weight 0.00 \
  --pseudo_final_weight 0.18 \
  --pseudo_material_weight 0.08 \
  --color_distribution_weight 0.08 \
  --graph_conv_chunk_size 512 \
  --gcn_subgraph_size 4096 \
  --gcn_seed_size 1024 \
  --gcn_expand_hops 1 \
  --external_normal_weight 0.7 \
  --metal_brightness_floor_weight 0.10 \
  --metal_brightness_target 0.60 \
  --output_dir result_bunny_metal_fusednorm_trainfix