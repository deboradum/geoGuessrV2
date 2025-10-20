import torch
import torch.nn as nn
from transformers import AutoModel


class GeoEmbedModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_features: int, embedding_dim: int, freeze_weights: bool):
        super().__init__()

        self.backbone = backbone
        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.norm = nn.LayerNorm(num_features)

        widening_factor = 2
        hidden_size = num_features * widening_factor

        self.geo_head = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, 0, :]

        x = self.norm(x)

        return self.geo_head(x)


def get_embed_net(embedding_dim: int=256, freeze_weights: bool=False, size:str="b16", device="cpu") -> torch.nn.Module:
    if size == "s16":  # 21M params
        backbone = AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
    elif size == "s+16":  # 29M parmas
        backbone = AutoModel.from_pretrained("facebook/dinov3-vits16plus-pretrain-lvd1689m")
    elif size == "b16":  # 86M params
        backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    elif size == "l16":  # 300M params
        backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    elif size == "h+16":  # 840M params
        backbone = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
    elif size == "7B16":  # 6,716M params
        backbone = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
    else:
        print(f"{size} not supported")
        exit(0)

    num_features = backbone.config.hidden_size

    return GeoEmbedModel(backbone, num_features, embedding_dim, freeze_weights).to(device)
