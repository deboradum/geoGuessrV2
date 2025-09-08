import torch
import torchvision

import torch.nn as nn

from typing import Any

class GeoGuessrModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_features: int, num_classes: int, freeze_weights: bool):
        super().__init__()
        self.backbone = backbone
        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.flatten = nn.Flatten()
        self.lon_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.lat_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        lon = self.lon_head(x)
        lat = self.lat_head(x)
        out = torch.stack([lon, lat], dim=-1)  # BxCx2

        return out


def get_convnext(size: str, num_classes: int, freeze_weights: bool) -> nn.Module:
    if size == "tiny":  # 29M params
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        model = torchvision.models.convnext_tiny
    elif size == "small":  # 50M parmas
        weights = torchvision.models.ConvNeXt_Small_Weights.DEFAULT
        model = torchvision.models.convnext_small
    elif size == "base":  # 89M params
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        model = torchvision.models.convnext_base
    elif size == "large":  # 198M params
        weights = torchvision.models.ConvNeXt_Large_Weights.DEFAULT
        model = torchvision.models.convnext_large
    else:
        print(f"{size} not supported")
        exit(0)

    backbone = model(weights=weights)
    backbone_features = nn.Sequential(*list(backbone.children())[:-1])
    num_features = backbone.classifier[2].in_features

    return GeoGuessrModel(backbone_features, num_features, num_classes, freeze_weights)


def get_net( num_classes: int, freeze_weights: bool, net_name:str="convnext-tiny", device="cpu") -> torch.nn.Module | Any:
    if "convnext" in net_name:
        return get_convnext(net_name.split("-")[-1], num_classes, freeze_weights).to(device)
    else:
        print(f"{net_name} not supported")
        exit(0)
