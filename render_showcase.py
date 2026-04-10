import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import math
from tqdm import tqdm
import torchvision.transforms as T

# 导入核心组件
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_pbr
from pbr_modules.brdf_renderer import render_pbr_image
from train_pbr import get_dummy_camera  # 复用我们之前的虚拟相机

def render_pbr_showcase(ply_path, output_dir="showcase_frames", num_frames=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"1. 加载 PBR 材质模型: {ply_path} ...")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(ply_path)
    
    # 获取正面视角相机
    view_cam = get_dummy_camera(device)
    bg_color = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=device) # 暗灰色背景
    
    print("2. 渲染 G-Buffer (只需渲染一次，因为相机没动)...")
    with torch.no_grad():
        render_pkg = render_pbr(
            viewpoint_camera=view_cam, 
            pc=gaussians, 
            pipe=None, 
            bg_color=bg_color
        )
        
    print(f"3. 生成 {num_frames} 帧动态光照动画...")
    # 模拟一个从正面看过来的视线方向
    view_dir = torch.tensor([0.0, 0.0, 1.0], device=device).view(3, 1, 1)
    # 光照强度
    light_color = torch.tensor([6.0, 6.0, 6.0], device=device).view(3, 1, 1) 
    
    progress_bar = tqdm(range(num_frames), desc="Rendering Frames")
    for i in progress_bar:
        # 让灯光绕着 Y 轴（垂直方向）旋转
        angle = (i / num_frames) * 2 * math.pi
        light_x = math.sin(angle)
        light_y = 0.5  # 光源稍微偏上一点
        light_z = math.cos(angle)
        
        light_dir = torch.tensor([light_x, light_y, light_z], device=device).view(3, 1, 1)
        
        # 物理着色
        shaded_image = render_pbr_image(
            albedo=render_pkg["albedo"],
            roughness=render_pkg["roughness"],
            metallic=render_pkg["metallic"],
            normal=render_pkg["normal"],
            view_dir=view_dir,
            light_dir=light_dir,
            light_color=light_color
        )
        
        # 保存图片
        save_img = T.ToPILImage()(torch.clamp(shaded_image, 0.0, 1.0).detach().cpu())
        save_img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
        
    print(f"\n渲染完成！请前往 '{output_dir}' 文件夹查看图片序列。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default="pbr_outputs/final_pbr_model.ply")
    args = parser.parse_args()
    
    render_pbr_showcase(args.ply)