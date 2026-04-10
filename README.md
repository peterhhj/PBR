python train_pbr.py --ply_path point_cloud.ply --style_image jade.jpg --iterations 2000


├── scene/
│   ├── gaussian_model.py       <-- [需修改] 剥离SH，接入PBR属性，冻结几何[]
├── gaussian_renderer/
│   ├── __init__.py             <-- [需修改] 魔改渲染入口，输出 G-Buffer(法线, 基础色, 粗糙度等)
├── pbr_modules/                <-- [新建目录] 你的核心工作区
│   ├── __init__.py
│   ├── predictor.py            <-- [新建] 基于 3dgcn 编写的 PBR 材质预测网络
│   ├── brdf_renderer.py        <-- [新建] 纯 PyTorch 实现的物理渲染方程(Cook-Torrance)
│   ├── style_loss.py           <-- [新建] VGG 特征提取与 Gram 矩阵风格损失
├── gcn3d.py                    <-- [直接复制] 来自 chih-hao-lin/3dgcn 的核心文件
├── train_pbr.py                <-- [新建] 专用于风格化训练的主循环脚本


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



python train_pbr.py --ply_path <your_ply> --source <your_dataset> --style_image jade.jpg --material_preset jade --warmup_iters 800 --views_per_iter 1
