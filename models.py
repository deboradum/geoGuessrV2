import os
import torch
import torchvision  # type: ignore[import-untyped]

import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from utils import TrainConfig
from transformers import AutoModel

class TopKRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, k: int):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(embedding_dim, num_experts, bias=False)

    def forward(self, x):
        # Noise injection
        if self.training: # (B, num_features)
            noise = torch.randn_like(self.gate(x)) * 1e-2
            expert_scores = self.gate(x) + noise
        else:
            expert_scores = self.gate(x)

        top_k_logits, top_k_indices = torch.topk(expert_scores, self.k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        all_expert_probs = F.softmax(expert_scores, dim=-1)

        return top_k_probs, top_k_indices, all_expert_probs


class CartesianExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # (x, y, z)
        )

    def forward(self, x):
        return self.net(x)  # (B, input_dim) > (B, 3)


class GeoGuessrModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_features: int, freeze_weights: bool, embedding_dim: int, num_experts: int, k: int, num_s2_classes: int, is_vit: bool = False, device="cpu"):
        super().__init__()
        self.device = device
        self.is_vit = is_vit

        self.backbone = backbone
        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if not self.is_vit:
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.norm = nn.LayerNorm(num_features)

        hidden_size = 512

        # The geo head creates a locational embedding from the features.
        self.geo_head = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # The classifier head predicts the S2 cell I, providing some sort of hint to the experts
        self.s2_classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_s2_classes),
        )

        # The experts are the regression heads, predicting (x, y, z) coordinates
        self.num_experts = num_experts
        self.router = TopKRouter(embedding_dim=embedding_dim, num_experts=num_experts, k=k)
        self.experts = nn.ModuleList([CartesianExpert(embedding_dim, hidden_size) for _ in range(num_experts)])

    def forward(self, x, embedding_only: bool = False):
        bs = x.shape[0]

        x = self.backbone(x)

        if self.is_vit:
            x = x.last_hidden_state[:, 0, :]
        else:
            x = self.pool(x)
            x = torch.flatten(x, 1)
        x = self.norm(x)

        # Produce location-based embedding
        x_embed = self.geo_head(x)
        if embedding_only:
            return x_embed

        # S2 cell classification, used as a 'hint' for the router
        s2_logits = self.s2_classifier_head(x_embed)

        experts_weights, experts_indices, all_expert_probs = self.router(x_embed)  # (B, k), (B, k)
        P = all_expert_probs.mean(dim=0)

        load_balancing_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        out = torch.zeros(bs, 3, dtype=x_embed.dtype, device=self.device)  # (B, 3)

        expert_load = torch.zeros(self.num_experts, dtype=torch.float32, device=self.device)
        total_assignments = bs * self.router.k
        for i, expert in enumerate(self.experts):
            (batch_idx, top_k_idx) = torch.where(experts_indices == i)
            num_tokens_routed_to_i = batch_idx.numel()
            if num_tokens_routed_to_i == 0:
                continue

            weights = experts_weights[batch_idx, top_k_idx]  # (len(batch_idx),)
            expert_out = expert(x_embed[batch_idx])  # (len(batch_idx), 3)
            weighted_out = expert_out * weights.unsqueeze(-1)  # (len(batch_idx), 3)
            out.index_add_(0, batch_idx, weighted_out)

            f_i = num_tokens_routed_to_i / total_assignments  # (1,)
            load_balancing_loss += f_i * P[i]
            expert_load[i] = f_i

        # Normalize to unit sphere
        out = F.normalize(out, p=2, dim=1, eps=1e-8)

        cv_load = torch.std(expert_load) / (expert_load.mean() + 1e-6)
        dead_experts = torch.sum(expert_load == 0)
        router_prob_entropy = -torch.sum(P * torch.log(P + 1e-9))

        load_metrics = {
            "load_balancing_loss": load_balancing_loss * self.num_experts,
            "expert_load_cv": cv_load,
            "dead_experts": dead_experts,
            "router_prob_entropy": router_prob_entropy,
        }

        return out, s2_logits, load_metrics,


def get_convnext(size: str, num_s2_classes: int, config: TrainConfig, device) -> nn.Module:
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

    net = GeoGuessrModel(backbone.features, num_features, config.freeze_weights, config.embedding_dim, config.num_experts, config.router_k, num_s2_classes, is_vit=False, device=device) # type: ignore
    if os.path.isfile(config.pretrained_path):
        print("Resuming training from checkpoint:", config.pretrained_path)
        net.load_state_dict(torch.load(config.pretrained_path, weights_only=True))

    return net


def get_vit(size: str, num_s2_classes: int, config: TrainConfig, device) -> nn.Module:
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

    net = GeoGuessrModel(backbone, num_features, config.freeze_weights, config.embedding_dim, config.num_experts, config.router_k, num_s2_classes, is_vit=True, device=device)
    if os.path.isfile(config.pretrained_path):
        print("Resuming training from checkpoint:", config.pretrained_path)
        net.load_state_dict(torch.load(config.pretrained_path, weights_only=True))

    return net


def get_net(num_s2_classes: int, config: TrainConfig, device="cpu") -> torch.nn.Module | Any:
    net_name = config.net_name
    if "convnext" in net_name:
        return get_convnext(net_name.split("-")[-1], num_s2_classes, config, device).to(device)
    if "vit" in net_name:
        return get_vit(net_name.split("-")[-1], num_s2_classes, config, device).to(device)
    else:
        print(f"{net_name} not supported")
        exit(0)
