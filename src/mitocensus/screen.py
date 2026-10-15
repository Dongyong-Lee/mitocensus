from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from .metrics import aggregate_patch_predictions
from .model import TinyRefinementModel, segmentation_loss


def decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass
class ScreenData:
    raw9: np.ndarray
    mask: np.ndarray
    mitonet: np.ndarray
    omniem: np.ndarray
    split: np.ndarray
    source: np.ndarray
    volume: np.ndarray
    domain: np.ndarray
    positive: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "ScreenData":
        with h5py.File(path, "r") as handle:
            source = np.asarray([decode(value) for value in handle["source"][:]], dtype=object)
            domains = []
            for value in source:
                if value.startswith("MitoEM2/Dataset"):
                    domains.append(int(value.split("Dataset", 1)[1][:3]) - 1)
                else:
                    domains.append(8)
            return cls(
                raw9=np.asarray(handle["raw9"][:], dtype=np.uint8),
                mask=np.asarray(handle["mask"][:], dtype=np.uint8),
                mitonet=np.asarray(handle["mitonet_probability"][:], dtype=np.float16),
                omniem=np.asarray(handle["omniem_probability"][:], dtype=np.float16),
                split=np.asarray([decode(value) for value in handle["split"][:]], dtype=object),
                source=source,
                volume=np.asarray([decode(value) for value in handle["volume"][:]], dtype=object),
                domain=np.asarray(domains, dtype=np.int64),
                positive=np.asarray(handle["positive_sampling"][:], dtype=np.uint8),
            )


def context_width(name: str) -> int:
    explicit = {
        "single_slice": 1,
        "three_slice_25d": 3,
        "five_slice_25d": 5,
        "seven_slice_25d": 7,
        "spacing_scaled_context": 9,
        "dilated_z_context": 9,
        "convgru_section_memory": 9,
        "axial_section_attention": 9,
        "shallow_3d_unet": 9,
        "residual_3d_unet": 9,
        "triplanar_mean": 9,
        "triplanar_learned": 9,
        "usegment3d_consensus": 9,
        "section_memory_prototype": 9,
    }
    return explicit.get(name, 3)


def channel_indices(width: int) -> list[int]:
    half = width // 2
    return list(range(4 - half, 5 + half))


def architecture_mode(name: str) -> tuple[str, bool]:
    mapping = {
        "plain_unet": "plain",
        "residual_unet": "residual",
        "unetpp_dense_skip": "residual",
        "attention_gate_unet": "attention",
        "se_channel_attention": "se",
        "cbam_attention": "cbam",
        "nonlocal_bottleneck": "nonlocal",
        "axial_attention_bottleneck": "axial",
        "shifted_window_attention": "shifted",
        "global_context_token": "global",
        "gcformer_compact": "global",
        "convnext_large_kernel": "large_kernel",
        "gated_depthwise_ssm": "ssm",
        "hrnet_dual_resolution": "residual",
        "laplacian_pyramid_fusion": "large_kernel",
        "domain_mixture_experts": "moe",
        "axial_z_conv": "axial",
        "axial_section_attention": "axial",
        "dilated_z_context": "dilated",
        "convgru_section_memory": "global",
        "section_memory_prototype": "global",
        "shallow_3d_unet": "plain",
        "residual_3d_unet": "residual",
        "triplanar_learned": "attention",
        "usegment3d_consensus": "global",
    }
    mode = mapping.get(name, "residual")
    return mode, name in {"domain_mixture_experts", "router_three_experts"}


def loss_variant(name: str) -> str:
    mapping = {
        "tversky_recall": "tversky",
        "focal_tversky": "focal_tversky",
        "boundary_bce_head": "boundary",
        "signed_distance_head": "distance",
        "surface_loss": "surface",
        "cldice_term": "cldice",
        "connectivity_regularizer": "connectivity",
        "neighbor_affinity_head": "affinity",
        "center_offset_head": "center_offset",
        "watershed_distance": "watershed",
        "embedding_discriminative": "embedding",
        "split_merge_surrogate": "split_merge",
        "size_stratified_dice": "size_stratified",
        "trait_weighted_objective": "trait",
        "joint_region_boundary_distance": "joint",
    }
    return mapping.get(name, "bce_dice")


def model_settings(name: str) -> dict:
    mode, mixture = architecture_mode(name)
    domain_mode = "film" if name in {
        "domain_id_embedding", "conditional_instance_norm", "domain_specific_norm",
        "low_rank_domain_adapter", "prompt_token_adapter", "router_three_experts",
    } else "none"
    direct = name in {"plain_unet", "shallow_3d_unet", "residual_3d_unet", "gcformer_compact"}
    width = 10 if name in {"gcformer_compact", "hrnet_dual_resolution", "domain_mixture_experts"} else 8
    extra = int(name in {"gradient_aux_input", "laplacian_aux_input", "laplacian_pyramid_fusion"})
    context = context_width(name)
    return {
        "mode": mode,
        "mixture": mixture or name == "router_three_experts",
        "domain_mode": domain_mode,
        "residual_prior": not direct,
        "width": width,
        "context": context,
        "input_channels": context + 2 + extra,
        "loss": loss_variant(name),
    }


def _normalize_raw(raw: torch.Tensor, name: str, training: bool) -> torch.Tensor:
    if name == "volume_robust_p1_99":
        flat = raw.flatten(2)
        low = torch.quantile(flat, 0.01, dim=2)[:, :, None, None]
        high = torch.quantile(flat, 0.99, dim=2)[:, :, None, None]
        raw = ((raw - low) / (high - low).clamp_min(1e-3)).clamp(0, 1)
    elif name == "volume_zscore_clipped":
        mean = raw.mean((1, 2, 3), keepdim=True)
        sd = raw.std((1, 2, 3), keepdim=True).clamp_min(1e-3)
        raw = ((raw - mean) / sd).clamp(-3, 3) / 6 + 0.5
    elif name == "slice_zscore_smoothed":
        mean = raw.mean((2, 3), keepdim=True)
        sd = raw.std((2, 3), keepdim=True).clamp_min(1e-3)
        raw = ((raw - mean) / sd).clamp(-3, 3) / 6 + 0.5
    elif name == "grayscale_migration":
        center_mean = raw[:, raw.shape[1] // 2 : raw.shape[1] // 2 + 1].mean((2, 3), keepdim=True)
        center_sd = raw[:, raw.shape[1] // 2 : raw.shape[1] // 2 + 1].std((2, 3), keepdim=True).clamp_min(1e-3)
        mean = raw.mean((2, 3), keepdim=True)
        sd = raw.std((2, 3), keepdim=True).clamp_min(1e-3)
        raw = ((raw - mean) / sd * center_sd + center_mean).clamp(0, 1)
    elif name == "clahe_limited":
        low = F.avg_pool2d(raw, 15, stride=1, padding=7)
        raw = (0.5 + 1.5 * (raw - low)).clamp(0, 1)
    elif name == "local_contrast_31":
        mean = F.avg_pool2d(raw, 31, stride=1, padding=15)
        variance = F.avg_pool2d((raw - mean).square(), 31, stride=1, padding=15)
        raw = ((raw - mean) / variance.add(1e-3).sqrt()).clamp(-3, 3) / 6 + 0.5
    elif name == "gamma_jitter_train" and training:
        gamma = torch.empty((len(raw), 1, 1, 1), device=raw.device).uniform_(0.7, 1.5)
        raw = raw.clamp_min(1e-4).pow(gamma)
    elif name == "fourier_amplitude_mix" and training and len(raw) > 1:
        order = torch.randperm(len(raw), device=raw.device)
        mean = raw.mean((2, 3), keepdim=True)
        sd = raw.std((2, 3), keepdim=True).clamp_min(1e-3)
        raw = ((raw - mean) / sd * sd[order] + mean[order]).clamp(0, 1)
    elif name == "gaussian_denoise_sigma1":
        raw = F.avg_pool2d(raw, 3, stride=1, padding=1)
    elif name == "source_style_randomization" and training:
        gain = torch.empty((len(raw), 1, 1, 1), device=raw.device).uniform_(0.75, 1.25)
        bias = torch.empty((len(raw), 1, 1, 1), device=raw.device).uniform_(-0.12, 0.12)
        raw = (raw * gain + bias + 0.03 * torch.randn_like(raw)).clamp(0, 1)
        if torch.rand((), device=raw.device) < 0.5:
            raw = F.avg_pool2d(raw, 3, stride=1, padding=1)
    return raw


def make_input(
    data: ScreenData, indices: np.ndarray, name: str, settings: dict, device: torch.device, training: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    channels = channel_indices(settings["context"])
    raw = torch.from_numpy(data.raw9[indices][:, channels].astype(np.float32) / 255.0).to(device)
    target = torch.from_numpy(data.mask[indices].astype(np.float32)).to(device)
    mitonet = torch.from_numpy(data.mitonet[indices].astype(np.float32)).to(device)
    omniem = torch.from_numpy(data.omniem[indices].astype(np.float32)).to(device)
    domain = torch.from_numpy(data.domain[indices]).to(device)
    raw = _normalize_raw(raw, name, training)
    if name in {"gradient_aux_input", "laplacian_pyramid_fusion"}:
        center = raw[:, raw.shape[1] // 2 : raw.shape[1] // 2 + 1]
        gx = F.pad(center[..., 1:] - center[..., :-1], (0, 1, 0, 0))
        gy = F.pad(center[..., 1:, :] - center[..., :-1, :], (0, 0, 0, 1))
        raw = torch.cat((raw, torch.sqrt(gx.square() + gy.square() + 1e-6)), dim=1)
    elif name == "laplacian_aux_input":
        center = raw[:, raw.shape[1] // 2 : raw.shape[1] // 2 + 1]
        blur = F.avg_pool2d(center, 3, stride=1, padding=1)
        raw = torch.cat((raw, center - blur), dim=1)
    inputs = torch.cat((raw, mitonet, omniem), dim=1)
    if training and name not in {"cutmix_boundary_safe", "mixup_low_alpha"}:
        for batch_index in range(len(inputs)):
            if torch.rand((), device=device) < 0.5:
                inputs[batch_index] = torch.flip(inputs[batch_index], (-1,))
                target[batch_index] = torch.flip(target[batch_index], (-1,))
            if torch.rand((), device=device) < 0.5:
                inputs[batch_index] = torch.flip(inputs[batch_index], (-2,))
                target[batch_index] = torch.flip(target[batch_index], (-2,))
    if training and name == "mixup_low_alpha" and len(inputs) > 1:
        order = torch.randperm(len(inputs), device=device)
        lam = 0.85
        inputs = lam * inputs + (1 - lam) * inputs[order]
        target = lam * target + (1 - lam) * target[order]
        mitonet = lam * mitonet + (1 - lam) * mitonet[order]
    elif training and name == "cutmix_boundary_safe" and len(inputs) > 1:
        order = torch.randperm(len(inputs), device=device)
        y0, y1, x0, x1 = 24, 72, 24, 72
        inputs[:, :, y0:y1, x0:x1] = inputs[order, :, y0:y1, x0:x1]
        target[:, :, y0:y1, x0:x1] = target[order, :, y0:y1, x0:x1]
        mitonet[:, :, y0:y1, x0:x1] = mitonet[order, :, y0:y1, x0:x1]
    return inputs, target, mitonet, omniem, domain


def training_indices(data: ScreenData, name: str, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(data.split == "train")
    if name == "source_balanced_sampler":
        mito = pool[data.domain[pool] < 8]
        cem = pool[data.domain[pool] == 8]
        return np.concatenate((rng.choice(mito, count // 2, replace=False), rng.choice(cem, count - count // 2, replace=False)))
    if name in {"volume_balanced_sampler", "repeat_factor_2", "adaptive_repeat_factor", "group_dro_worst_domain"}:
        groups = {}
        for index in pool:
            key = data.volume[index] if name == "volume_balanced_sampler" else int(data.domain[index])
            groups.setdefault(key, []).append(index)
        keys = list(groups)
        output = []
        while len(output) < count:
            rng.shuffle(keys)
            for key in keys:
                candidates = groups[key]
                output.append(int(rng.choice(candidates)))
                if len(output) == count:
                    break
        return np.asarray(output)
    foreground = data.mask[pool].mean((1, 2, 3))
    if name == "foreground_half_sampler":
        positive = pool[foreground > 0]
        negative = pool[foreground == 0]
        return np.concatenate((rng.choice(positive, count // 2, replace=False), rng.choice(negative, count - count // 2, replace=len(negative) < count // 2)))
    if name == "boundary_oversample":
        score = np.asarray([np.mean(ndi.morphological_gradient(mask[0], size=(3, 3)) > 0) for mask in data.mask[pool]])
        chosen = pool[np.argsort(score)[-count:]]
        rng.shuffle(chosen)
        return chosen
    if name == "hard_negative_mining":
        error = ((data.mitonet[pool, 0] >= 0.5) & (data.mask[pool, 0] == 0)).mean((1, 2))
        chosen = pool[np.argsort(error)[-count:]]
        rng.shuffle(chosen)
        return chosen
    if name == "small_object_mining":
        valid = np.flatnonzero((foreground > 0) & (foreground < np.quantile(foreground[foreground > 0], 0.5)))
        candidates = pool[valid]
        return rng.choice(candidates, count, replace=len(candidates) < count)
    return rng.choice(pool, count, replace=False)


def optimizer_for(model: torch.nn.Module, name: str):
    if name == "radam_optimizer":
        return torch.optim.RAdam(model.parameters(), lr=6e-4, weight_decay=1e-4)
    if name == "sam_optimizer":
        return torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9, weight_decay=1e-4)
    return torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)


def expected_calibration_error(probability: np.ndarray, reference: np.ndarray, bins: int = 15) -> float:
    probability = probability.ravel()
    reference = reference.ravel().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(probability)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if selected.any():
            value += selected.mean() * abs(probability[selected].mean() - reference[selected].mean())
    return float(value)


def imported_probability(project_v1: Path, engine: str, size: int = 96) -> np.ndarray:
    path = project_v1 / "results/foundation_100h_v2/probabilities" / f"{engine}_validation_probability.npz"
    values = np.load(path)["probability"].astype(np.float32)
    tensor = torch.from_numpy(values[:, None])
    return F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False)[:, 0].numpy()


def stacker_probability(data: ScreenData, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train = np.flatnonzero(data.split == "train")
    pixels = np.column_stack((
        np.logit(np.clip(data.mitonet[train, 0].reshape(-1), 1e-4, 1 - 1e-4)),
        np.logit(np.clip(data.omniem[train, 0].reshape(-1), 1e-4, 1 - 1e-4)),
    ))
    labels = data.mask[train, 0].reshape(-1)
    chosen = rng.choice(len(labels), min(500_000, len(labels)), replace=False)
    model = LogisticRegression(max_iter=100, class_weight="balanced", random_state=seed)
    model.fit(pixels[chosen], labels[chosen])
    val = np.flatnonzero(data.split == "validation")
    features = np.column_stack((
        np.logit(np.clip(data.mitonet[val, 0].reshape(-1), 1e-4, 1 - 1e-4)),
        np.logit(np.clip(data.omniem[val, 0].reshape(-1), 1e-4, 1 - 1e-4)),
    ))
    return model.predict_proba(features)[:, 1].reshape(len(val), 96, 96).astype(np.float32)


def evaluate_probability(
    probability: np.ndarray,
    reference: np.ndarray,
    volumes: list[str],
    best_control_by_volume: dict[str, float],
    best_boundary_by_volume: dict[str, float],
) -> tuple[dict, list[dict]]:
    volume_rows, summary = aggregate_patch_predictions(probability, reference, volumes, 0.5)
    deltas = np.asarray([row["dice"] - best_control_by_volume[row["volume"]] for row in volume_rows])
    boundary_deltas = np.asarray([
        row["boundary_f1_tolerance2_px"] - best_boundary_by_volume[row["volume"]] for row in volume_rows
    ])
    summary.update({
        "mean_delta_vs_control_star": float(deltas.mean()),
        "worst_delta_vs_control_star": float(deltas.min()),
        "volumes_improved_vs_control_star": int((deltas > 0).sum()),
        "mean_boundary_delta_vs_control_star": float(boundary_deltas.mean()),
        "ece15": expected_calibration_error(probability, reference),
        "stage_a_robust_score": float(deltas.mean() + 0.25 * min(0, boundary_deltas.mean()) + 0.10 * deltas.min()),
    })
    for row, delta, boundary_delta in zip(volume_rows, deltas, boundary_deltas):
        row["delta_vs_control_star"] = float(delta)
        row["boundary_delta_vs_control_star"] = float(boundary_delta)
    return summary, volume_rows


def train_and_predict(
    data: ScreenData,
    row: dict,
    device: torch.device,
    checkpoint: Path,
) -> tuple[np.ndarray, dict]:
    name = row["short_name"]
    seed = int(row["stage_a_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    settings = model_settings(name)
    model = TinyRefinementModel(
        input_channels=settings["input_channels"], width=settings["width"], mode=settings["mode"],
        residual_prior=settings["residual_prior"], domain_conditioning=settings["domain_mode"],
        mixture_heads=settings["mixture"],
    ).to(device)
    optimizer = optimizer_for(model, name)
    indices = training_indices(data, name, int(row["stage_a_train_examples"]), seed)
    batch_size = 16
    steps_per_epoch = math.ceil(len(indices) / batch_size)
    total_steps = int(row["stage_a_epochs"]) * steps_per_epoch
    scheduler = None
    if name == "onecycle_schedule":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.2e-3, total_steps=total_steps)
    elif name == "cosine_warmup":
        warmup = max(1, total_steps // 10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
        )
    ema_state = None
    swa_states = []
    history = []
    for epoch in range(int(row["stage_a_epochs"])):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
        running = 0.0
        model.train()
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            inputs, target, mitonet, _, domain = make_input(data, chosen, name, settings, device, True)
            if name == "ema_pseudolabel_source":
                omniem = torch.from_numpy(data.omniem[chosen].astype(np.float32)).to(device)
                pseudo = 0.5 * (mitonet + omniem)
                is_cem = (domain == 8)[:, None, None, None]
                target = torch.where(is_cem, pseudo.detach(), target)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, mitonet, domain)
            loss = segmentation_loss(outputs, target, settings["loss"])
            if name in {"coral_alignment", "mmd_alignment", "domain_adversarial"}:
                features = outputs["region"].mean((2, 3))
                loss = loss + 0.02 * features.var(0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if name == "sam_optimizer":
                with torch.no_grad():
                    grad_norm = torch.norm(torch.stack([
                        parameter.grad.norm(2) for parameter in model.parameters() if parameter.grad is not None
                    ])).clamp_min(1e-12)
                    perturbations = []
                    for parameter in model.parameters():
                        if parameter.grad is None:
                            perturbations.append(None)
                            continue
                        perturbation = parameter.grad * (0.05 / grad_norm)
                        parameter.add_(perturbation)
                        perturbations.append(perturbation)
                optimizer.zero_grad(set_to_none=True)
                second = segmentation_loss(model(inputs, mitonet, domain), target, settings["loss"])
                second.backward()
                with torch.no_grad():
                    for parameter, perturbation in zip(model.parameters(), perturbations):
                        if perturbation is not None:
                            parameter.sub_(perturbation)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if name == "ema_teacher":
                state = model.state_dict()
                if ema_state is None:
                    ema_state = {key: value.detach().clone() for key, value in state.items()}
                else:
                    for key, value in state.items():
                        ema_state[key].mul_(0.99).add_(value.detach(), alpha=0.01)
            running += float(loss.detach()) * len(chosen)
        history.append({"epoch": epoch + 1, "loss": running / len(indices)})
        if name == "swa_tail":
            swa_states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
    if ema_state is not None:
        model.load_state_dict(ema_state)
    if swa_states:
        averaged = {}
        for key in swa_states[0]:
            if swa_states[0][key].is_floating_point():
                averaged[key] = torch.stack([state[key] for state in swa_states]).mean(0)
            else:
                averaged[key] = swa_states[-1][key]
        model.load_state_dict(averaged)
    validation = np.flatnonzero(data.split == "validation")
    batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation), 32):
            chosen = validation[start : start + 32]
            inputs, _, mitonet, _, domain = make_input(data, chosen, name, settings, device, False)
            output = torch.sigmoid(model(inputs, mitonet, domain)["region"])
            if name == "overlap_consistency_tta":
                flipped = torch.flip(inputs, (-1,))
                second = torch.sigmoid(model(flipped, torch.flip(mitonet, (-1,)), domain)["region"])
                output = 0.5 * (output + torch.flip(second, (-1,)))
            batches.append(output[:, 0].cpu().numpy().astype(np.float32))
    probability = np.concatenate(batches)
    if name == "uncertainty_abstention":
        fallback = data.mitonet[validation, 0].astype(np.float32)
        entropy = -(probability * np.log(np.clip(probability, 1e-6, 1)) + (1 - probability) * np.log(np.clip(1 - probability, 1e-6, 1)))
        probability[entropy > 0.62] = fallback[entropy > 0.62]
    elif name == "budgeted_add_remove_gate":
        prior = data.mitonet[validation, 0].astype(np.float32)
        proposed = probability >= 0.5
        old = prior >= 0.5
        confidence = np.abs(probability - 0.5)
        for index in range(len(probability)):
            edit = proposed[index] != old[index]
            budget = max(1, int(0.05 * max(1, old[index].sum())))
            positions = np.flatnonzero(edit)
            if len(positions) > budget:
                keep = positions[np.argsort(confidence[index].ravel()[positions])[-budget:]]
                selected = np.zeros(edit.size, dtype=bool)
                selected[keep] = True
                selected = selected.reshape(edit.shape)
                probability[index][edit & ~selected] = prior[index][edit & ~selected]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "mitocensus.refinement-stage-a-checkpoint.v1",
        "hypothesis_id": row["hypothesis_id"], "short_name": name, "seed": seed,
        "settings": settings, "history": history, "state_dict": model.state_dict(),
    }, checkpoint)
    return probability, {
        "training_history": history,
        "n_parameters": sum(value.numel() for value in model.parameters()),
        "checkpoint_sha256": sha256(checkpoint),
        "execution_fidelity": "independent_low_fidelity_mechanism_screen",
    }


def run_hypothesis(
    data: ScreenData,
    row: dict,
    device: torch.device,
    project_v1: Path,
    checkpoint: Path,
) -> tuple[np.ndarray, dict]:
    name = row["short_name"]
    validation = np.flatnonzero(data.split == "validation")
    if name == "exact_mitonet_control":
        return data.mitonet[validation, 0].astype(np.float32), {"execution_fidelity": "exact_frozen_probability", "n_parameters": 0}
    if name == "official_omniem_control":
        return data.omniem[validation, 0].astype(np.float32), {"execution_fidelity": "exact_frozen_probability", "n_parameters": 0}
    if name == "learned_two_prior_stacker":
        return stacker_probability(data, int(row["stage_a_seed"])), {"execution_fidelity": "independently_fitted_logistic_stacker", "n_parameters": 3}
    imported = {
        "emdino_frozen_features": ("E09", "selection_exposed_v1_E09_proxy_not_clean_EMDINO_ablation"),
        "dinov3_vit_frozen": ("E04", "selection_exposed_v1_E04"),
        "dinov3_convnext_frozen": ("E05", "selection_exposed_v1_E05"),
        "dinov3_dual_frozen": ("E10", "selection_exposed_v1_E10"),
    }
    if name in imported:
        engine, fidelity = imported[name]
        return imported_probability(project_v1, engine), {"execution_fidelity": fidelity, "n_parameters": None}
    if name == "domain_temperature_scaling":
        return data.mitonet[validation, 0].astype(np.float32), {"execution_fidelity": "temperature_scaling_semantic_identity_at_threshold_0.5", "n_parameters": 1}
    return train_and_predict(data, row, device, checkpoint)
