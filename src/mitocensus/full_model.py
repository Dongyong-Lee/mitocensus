from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, attention: bool = False, dilation: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.SiLU()
        if attention:
            hidden = max(4, channels // 4)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.SiLU(),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3),
                nn.Sigmoid(),
            )
        else:
            self.channel_gate = None
            self.spatial_gate = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.body(value)
        if self.channel_gate is not None:
            residual = residual * self.channel_gate(residual)
            spatial = torch.cat(
                (residual.mean(1, keepdim=True), residual.amax(1, keepdim=True)),
                dim=1,
            )
            residual = residual * self.spatial_gate(spatial)
        return self.activation(value + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, attention: bool):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.block = ResidualBlock(out_channels, attention=attention)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(value))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, attention: bool):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.block = ResidualBlock(out_channels, attention=attention)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.project(torch.cat((value, skip), dim=1)))


class FullRefinementModel(nn.Module):
    """Full-fidelity residual multi-prior model used after the 100H screen.

    The model does not average binary masks. It learns a spatial gate between
    two frozen probability fields and a bounded image-conditioned correction.
    Region, boundary and distance outputs share one encoder-decoder. Optional
    acquisition FiLM uses only spacing/modality class, never a specimen label.
    """

    def __init__(
        self,
        *,
        raw_context: int,
        width: int = 24,
        attention: bool = False,
        acquisition_film: bool = False,
        n_acquisition_classes: int = 3,
        correction_bound: float = 3.0,
    ):
        super().__init__()
        if raw_context not in {3, 5, 7, 9}:
            raise ValueError(f"unsupported raw context: {raw_context}")
        self.raw_context = raw_context
        self.width = width
        self.attention = attention
        self.acquisition_film = acquisition_film
        self.correction_bound = float(correction_bound)
        input_channels = raw_context + 2
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
        )
        self.encoder1 = ResidualBlock(width, attention=attention)
        self.encoder2 = DownBlock(width, width * 2, attention=attention)
        self.encoder3 = DownBlock(width * 2, width * 4, attention=attention)
        self.bottleneck = nn.Sequential(
            ResidualBlock(width * 4, attention=True, dilation=2),
            ResidualBlock(width * 4, attention=attention, dilation=1),
        )
        if acquisition_film:
            self.acquisition_embedding = nn.Embedding(
                n_acquisition_classes, width * 8
            )
        self.decoder2 = UpBlock(width * 4, width * 2, width * 2, attention=attention)
        self.decoder1 = UpBlock(width * 2, width, width, attention=attention)
        self.prior_gate = nn.Conv2d(width, 1, 1)
        self.correction_head = nn.Conv2d(width, 1, 1)
        self.boundary_head = nn.Conv2d(width, 1, 1)
        self.distance_head = nn.Conv2d(width, 1, 1)

        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.prior_gate.weight)
        nn.init.constant_(self.prior_gate.bias, math.log(3.0))

    def forward(
        self,
        raw: torch.Tensor,
        mitonet_probability: torch.Tensor,
        omniem_probability: torch.Tensor,
        acquisition_class: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if raw.shape[1] != self.raw_context:
            raise ValueError(
                f"expected {self.raw_context} raw channels, got {raw.shape[1]}"
            )
        value = torch.cat((raw, mitonet_probability, omniem_probability), dim=1)
        skip1 = self.encoder1(self.stem(value))
        skip2 = self.encoder2(skip1)
        low = self.encoder3(skip2)
        low = self.bottleneck(low)
        if self.acquisition_film:
            scale, shift = self.acquisition_embedding(acquisition_class).chunk(2, dim=1)
            scale = 0.15 * torch.tanh(scale)[:, :, None, None]
            shift = 0.15 * torch.tanh(shift)[:, :, None, None]
            low = low * (1.0 + scale) + shift
        decoded = self.decoder1(self.decoder2(low, skip2), skip1)

        gate = torch.sigmoid(self.prior_gate(decoded))
        mitonet_logit = torch.logit(mitonet_probability.clamp(1e-4, 1.0 - 1e-4))
        omniem_logit = torch.logit(omniem_probability.clamp(1e-4, 1.0 - 1e-4))
        base_logit = gate * mitonet_logit + (1.0 - gate) * omniem_logit
        correction = self.correction_bound * torch.tanh(self.correction_head(decoded))
        return {
            "region": base_logit + correction,
            "boundary": self.boundary_head(decoded),
            "distance": self.distance_head(decoded),
            "prior_gate": gate,
            "correction": correction,
        }


def boundary_target(target: torch.Tensor) -> torch.Tensor:
    eroded = -F.max_pool2d(-target, 3, stride=1, padding=1)
    return (target - eroded).clamp(0.0, 1.0)


def soft_dice_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logit)
    numerator = 2.0 * (probability * target).sum((1, 2, 3)) + 1.0
    denominator = (probability + target).sum((1, 2, 3)) + 1.0
    return 1.0 - (numerator / denominator).mean()


def full_fidelity_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    region = 0.5 * F.binary_cross_entropy_with_logits(outputs["region"], target)
    region = region + 0.5 * soft_dice_loss(outputs["region"], target)
    boundary = F.binary_cross_entropy_with_logits(
        outputs["boundary"], boundary_target(target)
    )
    distance_target = 2.0 * F.avg_pool2d(target, 9, stride=1, padding=4) - 1.0
    distance = F.smooth_l1_loss(torch.tanh(outputs["distance"]), distance_target)
    correction = outputs["correction"].abs().mean()
    total = region + 0.20 * boundary + 0.10 * distance + 0.002 * correction
    return total, {
        "region": float(region.detach()),
        "boundary": float(boundary.detach()),
        "distance": float(distance.detach()),
        "correction_l1": float(correction.detach()),
    }
