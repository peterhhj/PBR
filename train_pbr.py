import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as T
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
import math
import numpy as np
from scipy.spatial import cKDTree

# 导入 3DGS 核心组件
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render_pbr

# 导入我们自定义的 PBR 模块
from pbr_modules.predictor import PBRMaterialPredictor
from pbr_modules.brdf_renderer import render_pbr_image
from pbr_modules.style_loss import VGGStyleLoss

def load_style_image(image_path, device="cuda"):
    """加载风格参考图并预处理"""
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((512, 512)), # 统一大小便于计算 Gram 矩阵
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)

def get_dummy_camera(device="cuda"):
    """
    为了简化测试，生成一个正面的虚拟相机视角进行渲染。
    完美适配带有 Depth 参数的 Camera 类。
    """
    import math
    import numpy as np
    from PIL import Image

    R = np.eye(3, dtype=np.float32)
    T_vec = np.array([0, 0, 3.0], dtype=np.float32) # 相机往后退 3 个单位
    
    # 构造一个真实的 PIL 图像对象以防止 PILtoTorch 报错
    dummy_image = Image.new("RGB", (800, 800), (0, 0, 0))
    
    # 填补缺少的三个必填参数：resolution, depth_params, invdepthmap
    cam = Camera(
        resolution=(800, 800),   # 新增
        colmap_id=0, 
        R=R, 
        T=T_vec, 
        FoVx=math.pi/3, 
        FoVy=math.pi/3, 
        depth_params=None,       # 新增：设为空
        image=dummy_image,       # 修改：传入真实的 PIL 对象
        invdepthmap=None,        # 新增：设为空
        image_name="dummy", 
        uid=0, 
        data_device=device
    )
    return cam

def train_pbr_stylization(ply_path, style_image_path, iterations=3000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. 初始化高斯模型并加载冻结的 PLY...")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(ply_path)
    
    # ========== 新增：CPU 预计算 KNN 邻接图 ==========
    print("1.5. 使用 KDTree 预计算全局 KNN 图 (防止OOM)...")
    xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    tree = cKDTree(xyz_np)
    # k=21 因为第 0 个最近邻是点本身，我们需要排除它
    _, indices = tree.query(xyz_np, k=21, workers=-1)
    # 取后 20 个邻居，并转移到 GPU
    neighbor_indices = torch.tensor(indices[:, 1:], dtype=torch.long, device=device)
    # ================================================

    print("2. 初始化 3D-GCN 材质预测网络...")
    predictor = PBRMaterialPredictor(in_channels=6).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    print("3. 初始化 VGG 风格损失与参考图...")
    style_target = load_style_image(style_image_path, device)
    vgg_loss = VGGStyleLoss(device=device)
    
    # 固定的背景色
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    
    print("--- 开始训练 ---")
    progress_bar = tqdm(range(1, iterations + 1), desc="Training Progress")
    
    for iteration in progress_bar:
        predictor.train()
        optimizer.zero_grad()
        
        # 步骤 A：3D-GCN 预测 PBR 参数 (注意这里传入了 neighbor_indices)
        albedo_logits, rough_logits, metal_logits = predictor(
            gaussians.get_xyz, 
            gaussians.get_scaling, 
            neighbor_indices
        )
        
        pred_albedo = torch.sigmoid(albedo_logits)
        pred_roughness = torch.clamp(torch.sigmoid(rough_logits), min=0.04, max=1.0)
        pred_metallic = torch.sigmoid(metal_logits)
        
        # 步骤 B：获取相机视角并渲染 G-Buffer
        view_cam = get_dummy_camera(device)
        
        render_pkg = render_pbr(
            viewpoint_camera=view_cam, 
            pc=gaussians, 
            pipe=None, 
            bg_color=bg_color,
            override_albedo=pred_albedo,
            override_roughness=pred_roughness,
            override_metallic=pred_metallic
        )
        
        # 步骤 C：物理延迟着色
        light_dir = torch.tensor([1.0, 1.0, 1.0], device=device)
        light_color = torch.tensor([5.0, 5.0, 5.0], device=device)
        view_dir = torch.tensor([0.0, 0.0, 1.0], device=device) 
        
        shaded_image = render_pbr_image(
            albedo=render_pkg["albedo"],
            roughness=render_pkg["roughness"],
            metallic=render_pkg["metallic"],
            normal=render_pkg["normal"],
            view_dir=view_dir.view(3, 1, 1),
            light_dir=light_dir.view(3, 1, 1),
            light_color=light_color.view(3, 1, 1)
        )
        
        # 步骤 D：计算风格损失并反向传播
        shaded_resized = T.functional.resize(shaded_image, (512, 512))
        
        loss = vgg_loss(shaded_resized, style_target)
        loss.backward()
        
        optimizer.step()
        
        # 打印日志
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Style Loss": f"{loss.item():.{5}f}"})
            
        # 每隔 500 步保存一张当前渲染的图片用于观察效果
        if iteration % 500 == 0:
            os.makedirs("pbr_outputs", exist_ok=True)
            save_img = T.ToPILImage()(torch.clamp(shaded_image, 0.0, 1.0).detach().cpu())
            save_img.save(f"pbr_outputs/iter_{iteration}.png")
            
            # 保存当前的 GCN 模型权重
            torch.save(predictor.state_dict(), f"pbr_outputs/predictor_{iteration}.pth")

    # 训练结束后，保存包含 PBR 属性的 3DGS 模型
    print("保存带 PBR 材质的最终 3DGS 模型...")
    # 把预测出的最终材质属性赋给 gaussians 内部的张量
    gaussians._albedo.data = pred_albedo.detach()
    gaussians._roughness.data = pred_roughness.detach()
    gaussians._metallic.data = pred_metallic.detach()
    gaussians.save_ply("pbr_outputs/final_pbr_model.ply")
    print("保存成功！")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train PBR Material Predictor for 3DGS")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the frozen pre-trained .ply file")
    parser.add_argument("--style_image", type=str, required=True, help="Path to the target material image")
    parser.add_argument("--iterations", type=int, default=3000, help="Training iterations")
    
    args = parser.parse_args()
    
    train_pbr_stylization(args.ply_path, args.style_image, args.iterations)