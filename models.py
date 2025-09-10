import torch
import torchvision

import torch.nn as nn

from typing import Any

class GeoGuessrModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_features: int, freeze_weights: bool):
        super().__init__()
        self.backbone = backbone
        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Longitude regression head
        self.lon_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()  # map to [-1, 1]
        )

        # Latitude regression head
        self.lat_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()  # map to [-1, 1]
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        lon = self.lon_head(x) * 180  # scale to [-180, 180]
        lat = self.lat_head(x) * 90  # scale to [-90, 90]

        out = torch.cat([lon, lat], dim=1)  # (B, 2)
        return out


def get_convnext(size: str, freeze_weights: bool) -> nn.Module:
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
    num_features = backbone.classifier[2].in_features

    return GeoGuessrModel(backbone.features, num_features, freeze_weights)


def get_net(freeze_weights: bool, net_name:str="convnext-tiny", device="cpu") -> torch.nn.Module | Any:
    if "convnext" in net_name:
        return get_convnext(net_name.split("-")[-1], freeze_weights).to(device)
    else:
        print(f"{net_name} not supported")
        exit(0)
