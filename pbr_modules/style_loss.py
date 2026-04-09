import torch
import torch.nn as nn
import torchvision.models as models

class VGGStyleLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # 加载预训练的 VGG19，只需要特征提取部分
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        # 冻结 VGG 权重
        for param in vgg.parameters():
            param.requires_grad = False
            
        # 对于材质/光泽感的迁移，我们只需要极其浅层的特征 (例如 relu1_1, relu2_1)
        # 过深的层会引入不需要的图像宏观结构
        self.style_layers = {'0': 'relu1_1', '5': 'relu2_1', '10': 'relu3_1'}
        self.vgg_layers = vgg
        
        # 图像标准化参数 (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def get_features(self, image):
        # 图像需要加上 batch 维度 (1, C, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # 标准化
        x = (image - self.mean) / self.std
        features = {}
        for name, layer in self.vgg_layers._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[self.style_layers[name]] = x
        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (d * h * w)

    def forward(self, pred_image, target_image):
        pred_features = self.get_features(pred_image)
        target_features = self.get_features(target_image)
        
        style_loss = 0
        for layer in pred_features:
            pred_gram = self.gram_matrix(pred_features[layer])
            target_gram = self.gram_matrix(target_features[layer])
            style_loss += torch.mean((pred_gram - target_gram) ** 2)
            
        return style_loss