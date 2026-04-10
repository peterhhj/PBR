from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGStyleLoss(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:
            raise RuntimeError("torchvision is required for the VGG style encoder.") from exc

        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features.to(device).eval()
        except AttributeError:
            vgg = models.vgg19(pretrained=True).features.to(device).eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.style_layers = {"0": "relu1_1", "5": "relu2_1", "10": "relu3_1"}
        self.content_layer = "relu3_1"
        self.vgg_layers = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return (image - self.mean) / self.std

    def get_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(image)
        features = {}
        for name, layer in self.vgg_layers._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[self.style_layers[name]] = x
        return features

    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        reshaped = tensor.view(batch, channels, height * width)
        gram = torch.bmm(reshaped, reshaped.transpose(1, 2))
        return gram / (channels * height * width)

    def encode_style(self, image: torch.Tensor) -> torch.Tensor:
        image = image if image.ndim == 4 else image.unsqueeze(0)
        features = self.get_features(image)
        pooled = [feat.mean(dim=(-1, -2)) for feat in features.values()]
        color_stats = torch.cat([image.mean(dim=(-1, -2)), image.std(dim=(-1, -2))], dim=-1)
        return torch.cat(pooled + [color_stats], dim=-1).squeeze(0)

    def forward(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        layer_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        layer_weights = layer_weights or {"relu1_1": 1.0, "relu2_1": 1.0, "relu3_1": 1.0}
        pred_features = self.get_features(pred_image)
        target_features = self.get_features(target_image)

        loss = pred_image.new_tensor(0.0)
        for layer_name, pred_feature in pred_features.items():
            pred_gram = self.gram_matrix(pred_feature)
            target_gram = self.gram_matrix(target_features[layer_name])
            loss = loss + layer_weights.get(layer_name, 1.0) * F.mse_loss(pred_gram, target_gram)
        return loss

    def content_loss(self, pred_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        pred_features = self.get_features(pred_image)
        target_features = self.get_features(target_image)
        return F.l1_loss(pred_features[self.content_layer], target_features[self.content_layer])
