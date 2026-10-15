from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def norm_groups(channels: int) -> int:
    for groups in (4, 3, 2):
        if channels % groups == 0:
            return groups
    return 1


class ChannelGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(2, channels // 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class ConvBlock(nn.Module):
    def __init__(self, channels: int, mode: str = "residual"):
        super().__init__()
        kernel = 7 if mode in {"large_kernel", "ssm"} else 3
        dilation = 2 if mode in {"dilated", "axial", "nonlocal"} else 1
        padding = dilation * (kernel // 2)
        groups = channels if mode in {"large_kernel", "ssm"} else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.GroupNorm(norm_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups(channels), channels),
        )
        self.activation = nn.SiLU()
        self.residual = mode != "plain"
        self.channel_gate = ChannelGate(channels) if mode in {"se", "cbam", "global", "moe"} else nn.Identity()
        self.spatial_gate = SpatialGate() if mode in {"cbam", "attention", "shifted"} else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.body(x)
        value = self.channel_gate(value)
        value = self.spatial_gate(value)
        if self.residual:
            value = value + x
        return self.activation(value)


class TinyRefinementModel(nn.Module):
    """Small multi-head screen model; finalists are rebuilt at full fidelity."""

    def __init__(
        self,
        input_channels: int,
        width: int = 8,
        mode: str = "residual",
        residual_prior: bool = True,
        domain_conditioning: str = "none",
        n_domains: int = 9,
        mixture_heads: bool = False,
    ):
        super().__init__()
        self.residual_prior = residual_prior
        self.domain_conditioning = domain_conditioning
        self.mixture_heads = mixture_heads
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups(width), width),
            nn.SiLU(),
        )
        self.block1 = ConvBlock(width, mode)
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.block2 = ConvBlock(width * 2, mode)
        self.up = nn.Conv2d(width * 3, width, 3, padding=1)
        self.block3 = ConvBlock(width, mode)
        if domain_conditioning != "none":
            self.domain_embedding = nn.Embedding(n_domains, width * 4)
        if mixture_heads:
            self.region_heads = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(3)])
        else:
            self.region_head = nn.Conv2d(width, 1, 1)
        self.boundary_head = nn.Conv2d(width, 1, 1)
        self.distance_head = nn.Conv2d(width, 1, 1)
        heads = self.region_heads if mixture_heads else [self.region_head]
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, x: torch.Tensor, prior_probability: torch.Tensor, domain: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        skip = self.block1(self.stem(x))
        low = self.block2(self.down(skip))
        if self.domain_conditioning != "none":
            scale, shift = self.domain_embedding(domain).chunk(2, dim=1)
            scale = 0.1 * torch.tanh(scale)[:, :, None, None]
            shift = 0.1 * shift[:, :, None, None]
            low = low * (1 + scale) + shift
        up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = self.block3(self.up(torch.cat((up, skip), dim=1)))
        if self.mixture_heads:
            expert = torch.stack([head(value) for head in self.region_heads], dim=1)
            anisotropic = torch.isin(domain, torch.tensor([3, 5, 7], device=domain.device))
            group = torch.where(domain >= 8, 2, torch.where(anisotropic, 1, 0))
            correction = expert[torch.arange(len(domain), device=domain.device), group]
        else:
            correction = self.region_head(value)
        if self.residual_prior:
            prior = prior_probability.clamp(1e-4, 1 - 1e-4)
            region = torch.logit(prior) + 2.0 * torch.tanh(correction)
        else:
            region = correction
        return {
            "region": region,
            "boundary": self.boundary_head(value),
            "distance": self.distance_head(value),
        }


def soft_dice_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logit)
    numerator = 2 * (probability * target).sum((1, 2, 3)) + 1
    denominator = (probability + target).sum((1, 2, 3)) + 1
    return 1 - (numerator / denominator).mean()


def tversky_loss(logit: torch.Tensor, target: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    probability = torch.sigmoid(logit)
    tp = (probability * target).sum((1, 2, 3))
    fp = (probability * (1 - target)).sum((1, 2, 3))
    fn = ((1 - probability) * target).sum((1, 2, 3))
    return 1 - ((tp + 1) / (tp + alpha * fp + beta * fn + 1)).mean()


def boundary_target(target: torch.Tensor) -> torch.Tensor:
    eroded = -F.max_pool2d(-target, 3, stride=1, padding=1)
    return (target - eroded).clamp(0, 1)


def segmentation_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, variant: str) -> torch.Tensor:
    region = outputs["region"]
    if variant == "tversky":
        base = tversky_loss(region, target, 0.3, 0.7)
    elif variant == "focal_tversky":
        base = tversky_loss(region, target, 0.3, 0.7).pow(1.33)
    else:
        base = 0.5 * F.binary_cross_entropy_with_logits(region, target) + 0.5 * soft_dice_loss(region, target)
    boundary_weight = 0.0
    distance_weight = 0.0
    if variant in {"boundary", "surface", "connectivity", "affinity", "center_offset", "split_merge"}:
        boundary_weight = 0.30
    if variant in {"distance", "surface", "watershed", "embedding", "trait", "joint"}:
        distance_weight = 0.20
    if variant in {"cldice", "connectivity"}:
        probability = torch.sigmoid(region)
        grad_p = (probability[..., 1:, :] - probability[..., :-1, :]).abs().mean() + (probability[..., :, 1:] - probability[..., :, :-1]).abs().mean()
        grad_t = (target[..., 1:, :] - target[..., :-1, :]).abs().mean() + (target[..., :, 1:] - target[..., :, :-1]).abs().mean()
        base = base + 0.1 * (grad_p - grad_t).abs()
    if variant in {"size_stratified", "trait"}:
        prevalence = target.mean((1, 2, 3)).clamp_min(1e-3)
        per_image = F.binary_cross_entropy_with_logits(region, target, reduction="none").mean((1, 2, 3))
        base = base + 0.1 * (per_image / prevalence.sqrt()).mean()
    if variant == "joint":
        boundary_weight = 0.30
        distance_weight = 0.20
    if boundary_weight:
        base = base + boundary_weight * F.binary_cross_entropy_with_logits(outputs["boundary"], boundary_target(target))
    if distance_weight:
        smooth_target = 2 * F.avg_pool2d(target, 9, stride=1, padding=4) - 1
        base = base + distance_weight * F.smooth_l1_loss(torch.tanh(outputs["distance"]), smooth_target)
    return base
