import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render_pbr(viewpoint_camera, pc, pipe, bg_color, 
               override_albedo=None, override_roughness=None, override_metallic=None):
    """
    专门为 PBR 材质预测设计的渲染管线。
    """
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0, 
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings)

    means3D = pc.get_xyz
    means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    albedo = override_albedo if override_albedo is not None else pc.get_albedo
    roughness = override_roughness if override_roughness is not None else pc.get_roughness
    metallic = override_metallic if override_metallic is not None else pc.get_metallic

    # --- Pass 1: 光栅化 Albedo ---
    # 【修改】：加上 _, 接收 depth_image
    albedo_map, radii, _ = rasterizer(
        means3D=means3D, means2D=means2D, shs=None, colors_precomp=albedo,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None
    )

    # --- Pass 2: 光栅化 Roughness 和 Metallic ---
    rm_dummy = torch.zeros_like(roughness)
    rm_packed = torch.cat([roughness, metallic, rm_dummy], dim=-1)
    
    # 【修改】：加上 _, _ 接收 radii 和 depth_image
    rm_map, _, _ = rasterizer(
        means3D=means3D, means2D=means2D, shs=None, colors_precomp=rm_packed,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None
    )
    
    roughness_map = rm_map[0:1, :, :] 
    metallic_map = rm_map[1:2, :, :]  

    # --- Pass 3: 计算并光栅化表面法线 Normal ---
    from utils.general_utils import build_rotation
    rot_mats = build_rotation(rotations) 
    normals = rot_mats[:, :, 2] 
    normals_mapped = (normals + 1.0) / 2.0 
    
    # 【修改】：加上 _, _ 接收 radii 和 depth_image
    normal_raw_map, _, _ = rasterizer(
        means3D=means3D, means2D=means2D, shs=None, colors_precomp=normals_mapped,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None
    )
    
    normal_map = normal_raw_map * 2.0 - 1.0
    normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

    return {
        "albedo": albedo_map,
        "roughness": roughness_map,
        "metallic": metallic_map,
        "normal": normal_map,
        "viewspace_points": means2D,
        "radii": radii
    }