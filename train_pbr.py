import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as T
from PIL import Image
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import math
import numpy as np
import random
import json

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_pbr
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos

from pbr_modules.predictor import PBRMaterialPredictor
from pbr_modules.brdf_renderer import render_pbr_image
from pbr_modules.style_loss import VGGStyleLoss

def compute_knn_cpu(xyz_tensor, k=21, chunk_size=2000):
    N = xyz_tensor.shape[0]
    neighbor_indices = torch.zeros((N, k-1), dtype=torch.long, device="cpu")
    print(f"正在 CPU 上分块计算 KNN (共 {N} 个点)...")
    for i in tqdm(range(0, N, chunk_size), desc="KNN 构建"):
        end = min(i + chunk_size, N)
        dist = torch.cdist(xyz_tensor[i:end], xyz_tensor)
        _, indices = torch.topk(dist, k=k, dim=1, largest=False)
        neighbor_indices[i:end] = indices[:, 1:]
    return neighbor_indices

def load_style_image(image_path, device="cuda"):
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.CenterCrop(min(img.size)), 
        T.Resize((512, 512)), 
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)

def train_pbr_stylization(ply_path, source_path, style_image_path, iterations=3000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. 读取相机及数据集参数...")
    is_synthetic = os.path.exists(os.path.join(source_path, "transforms_train.json"))
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", eval=False, train_test_exp=False, depths="")
    elif is_synthetic:
        scene_info = sceneLoadTypeCallbacks["Blender"](source_path, "white", eval=False, train_test_exp=False, depths="")
    
    cam_args = Namespace(resolution=1, data_device=device, train_test_exp=False)
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, cam_args, is_nerf_synthetic=is_synthetic, is_test_dataset=False)
    
    camera_params_path = os.path.join(source_path, "camera_params.json")
    exact_obj_center = None
    if os.path.exists(camera_params_path):
        with open(camera_params_path, 'r') as f:
            exact_obj_center = torch.tensor(json.load(f)["center"], dtype=torch.float32, device=device)

    print("2. 初始化模型...")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(ply_path)
    
    if exact_obj_center is None:
        exact_obj_center = gaussians.get_xyz.mean(dim=0)

    xyz_cpu = gaussians.get_xyz.detach().cpu()
    neighbor_indices = compute_knn_cpu(xyz_cpu, k=11, chunk_size=2000).to(device)

    predictor = PBRMaterialPredictor(in_channels=6, support_num=4, neighbor_num=10).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    style_target = load_style_image(style_image_path, device)
    vgg_loss = VGGStyleLoss(device=device)
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    
    print("--- 开始训练 ---")
    progress_bar = tqdm(range(1, iterations + 1), desc="Training Progress")
    
    for iteration in progress_bar:
        predictor.train()
        optimizer.zero_grad()
        
        albedo_logits, rough_logits, metal_logits = predictor(gaussians.get_xyz, gaussians.get_scaling, neighbor_indices)
        pred_albedo = torch.sigmoid(albedo_logits) * 0.95 + 0.05 
        pred_roughness = torch.sigmoid(rough_logits) * 0.96 + 0.04 
        pred_metallic = torch.sigmoid(metal_logits)
        
        view_cam = random.choice(train_cameras)
        render_pkg = render_pbr(
            viewpoint_camera=view_cam, pc=gaussians, pipe=None, bg_color=bg_color,
            override_albedo=pred_albedo, override_roughness=pred_roughness, override_metallic=pred_metallic
        )
        
        # ================= 核心修复 1：重塑 3D 立体光影 =================
        cam_pos = view_cam.camera_center
        view_dir = torch.nn.functional.normalize(cam_pos - exact_obj_center, dim=0)
        
        # 主光源 (Key Light)：拉开与视线的角度，从右上方斜打过来，制造强烈的立体阴影！
        light_dir = torch.nn.functional.normalize(view_dir + torch.tensor([1.2, 1.5, 0.0], device=device), dim=0)
        light_color = torch.tensor([6.0, 6.0, 6.0], device=device)
        
        shaded_image = render_pbr_image(
            albedo=render_pkg["albedo"], roughness=render_pkg["roughness"], metallic=render_pkg["metallic"],
            normal=render_pkg["normal"], view_dir=view_dir.view(3, 1, 1), 
            light_dir=light_dir.view(3, 1, 1), light_color=light_color.view(3, 1, 1)
        )
        
        # 环境光 (Ambient Light)：防止背光面变成死黑，补充全局柔和体积感
        ambient_light = render_pkg["albedo"] * 0.15
        shaded_image = torch.clamp(shaded_image + ambient_light, 0.0, 1.0)
        # ================================================================

        # ================= 核心修复 2：柔和平滑掩码 (Soft Mask) =================
        # 用 albedo 的亮度渐变做软掩码，保留高斯渲染的羽化边缘，告别狗牙抠图感
        albedo_max, _ = torch.max(render_pkg["albedo"], dim=0)
        soft_mask = torch.clamp(albedo_max * 3.0, 0.0, 1.0).unsqueeze(0)
        
        # 边界框裁剪仍用二值判断，以确保安全
        binary_mask = (albedo_max > 0.01)
        nonzero_indices = torch.nonzero(binary_mask)
        
        if nonzero_indices.numel() > 100:
            y_min, y_max = nonzero_indices[:, 0].min(), nonzero_indices[:, 0].max()
            x_min, x_max = nonzero_indices[:, 1].min(), nonzero_indices[:, 1].max()
            cropped_shaded = shaded_image[:, y_min:y_max+1, x_min:x_max+1]
        else:
            cropped_shaded = shaded_image
        # ========================================================================
            
        shaded_resized = T.functional.resize(cropped_shaded, (512, 512))
        style_loss = vgg_loss(shaded_resized, style_target)
        
        loss_metallic = torch.mean(pred_metallic) 
        loss_roughness = torch.mean((pred_roughness - 0.15) ** 2) 

        total_loss = style_loss + (0.0005 * loss_metallic) + (0.001 * loss_roughness)
        
        total_loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "Tot": f"{total_loss.item():.4f}", 
                "Sty": f"{style_loss.item():.4f}",
                "Rgh": f"{loss_roughness.item():.4f}"
            })
            
        if iteration == 1 or iteration % 100 == 0:
            print(f"\n--- [DEBUG] Iter: {iteration} ---")
            print(f"Albedo   均值: {pred_albedo.mean().item():.4f}")
            print(f"Rough    均值: {pred_roughness.mean().item():.4f}")
            
        # ================= 核心修复 3：完美融合白色背景 =================
        if iteration == 1 or iteration % 500 == 0:
            os.makedirs("pbr_outputs", exist_ok=True)
            # 使用 soft_mask 平滑过渡：中心不透明，边缘半透明融合白色
            white_bg_image = shaded_image * soft_mask + 1.0 * (1.0 - soft_mask)
            img_tensor = torch.clamp(white_bg_image, 0.0, 1.0).detach().cpu()
            if not torch.isnan(img_tensor).any():
                T.ToPILImage()(img_tensor).save(f"pbr_outputs/iter_{iteration}.png")

    print("保存带 PBR 材质的最终 3DGS 模型...")
    gaussians._albedo.data = pred_albedo.detach()
    gaussians._roughness.data = pred_roughness.detach()
    gaussians._metallic.data = pred_metallic.detach()
    gaussians.save_ply("pbr_outputs/final_pbr_model.ply")
    print("保存成功！")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    train_pbr_stylization(args.ply_path, args.source, args.style_image, args.iterations)