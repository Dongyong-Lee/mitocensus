from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .full_model import FullRefinementModel, full_fidelity_loss


def decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(payload: dict) -> str:
    value = dict(payload)
    value.pop("contract_sha256", None)
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class FullPatchData:
    raw9: np.ndarray
    mask: np.ndarray
    mitonet: np.ndarray
    omniem: np.ndarray
    split: np.ndarray
    source: np.ndarray
    volume: np.ndarray
    domain: np.ndarray
    acquisition: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "FullPatchData":
        with h5py.File(path, "r") as handle:
            source = np.asarray([decode(value) for value in handle["source"][:]], dtype=object)
            domain = []
            acquisition = []
            for value in source:
                number = int(value.split("Dataset", 1)[1][:3]) - 1
                domain.append(number)
                acquisition.append(1 if number in {3, 5, 7} else 0)
            return cls(
                raw9=np.asarray(handle["raw9"][:], dtype=np.uint8),
                mask=np.asarray(handle["mask"][:], dtype=np.uint8),
                mitonet=np.asarray(handle["mitonet_probability"][:], dtype=np.float16),
                omniem=np.asarray(handle["omniem_probability"][:], dtype=np.float16),
                split=np.asarray([decode(value) for value in handle["split"][:]], dtype=object),
                source=source,
                volume=np.asarray([decode(value) for value in handle["volume"][:]], dtype=object),
                domain=np.asarray(domain, dtype=np.int64),
                acquisition=np.asarray(acquisition, dtype=np.int64),
            )


def channel_indices(context: int) -> list[int]:
    if context not in {3, 5, 7, 9}:
        raise ValueError(context)
    half = context // 2
    return list(range(4 - half, 5 + half))


def candidate_model(config: dict) -> FullRefinementModel:
    return FullRefinementModel(
        raw_context=int(config["raw_context"]),
        width=int(config["width"]),
        attention=bool(config["attention"]),
        acquisition_film=bool(config["acquisition_film"]),
        correction_bound=float(config["correction_bound"]),
    )


def training_indices(data: FullPatchData, config: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(data.split == "train")
    count = int(config["examples_per_epoch"])
    if config["sampler"] == "domain_balanced":
        groups = {key: pool[data.domain[pool] == key] for key in sorted(set(data.domain[pool]))}
        result = []
        keys = list(groups)
        while len(result) < count:
            rng.shuffle(keys)
            for key in keys:
                result.append(int(rng.choice(groups[key])))
                if len(result) == count:
                    break
        return np.asarray(result, dtype=np.int64)
    return rng.choice(pool, count, replace=count > len(pool)).astype(np.int64)


def make_batch(
    data: FullPatchData,
    indices: np.ndarray,
    config: dict,
    device: torch.device,
    *,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    channels = channel_indices(int(config["raw_context"]))
    raw = torch.from_numpy(data.raw9[indices][:, channels].astype(np.float32) / 255.0).to(device)
    target = torch.from_numpy(data.mask[indices].astype(np.float32)).to(device)
    mitonet = torch.from_numpy(data.mitonet[indices].astype(np.float32)).to(device)
    omniem = torch.from_numpy(data.omniem[indices].astype(np.float32)).to(device)
    acquisition = torch.from_numpy(data.acquisition[indices]).to(device)

    if training:
        for batch_index in range(len(raw)):
            if torch.rand((), device=device) < 0.5:
                raw[batch_index] = torch.flip(raw[batch_index], (-1,))
                target[batch_index] = torch.flip(target[batch_index], (-1,))
                mitonet[batch_index] = torch.flip(mitonet[batch_index], (-1,))
                omniem[batch_index] = torch.flip(omniem[batch_index], (-1,))
            if torch.rand((), device=device) < 0.5:
                raw[batch_index] = torch.flip(raw[batch_index], (-2,))
                target[batch_index] = torch.flip(target[batch_index], (-2,))
                mitonet[batch_index] = torch.flip(mitonet[batch_index], (-2,))
                omniem[batch_index] = torch.flip(omniem[batch_index], (-2,))

        if config["augmentation"] == "cutmix" and len(indices) > 1:
            order = torch.randperm(len(indices), device=device)
            height, width = target.shape[-2:]
            box_h, box_w = height // 2, width // 2
            y0 = int(torch.randint(0, height - box_h + 1, (), device=device))
            x0 = int(torch.randint(0, width - box_w + 1, (), device=device))
            y1, x1 = y0 + box_h, x0 + box_w
            raw[:, :, y0:y1, x0:x1] = raw[order, :, y0:y1, x0:x1]
            target[:, :, y0:y1, x0:x1] = target[order, :, y0:y1, x0:x1]
            mitonet[:, :, y0:y1, x0:x1] = mitonet[order, :, y0:y1, x0:x1]
            omniem[:, :, y0:y1, x0:x1] = omniem[order, :, y0:y1, x0:x1]
        elif config["augmentation"] == "mixup" and len(indices) > 1:
            order = torch.randperm(len(indices), device=device)
            weight = 0.85
            raw = weight * raw + (1.0 - weight) * raw[order]
            target = weight * target + (1.0 - weight) * target[order]
            mitonet = weight * mitonet + (1.0 - weight) * mitonet[order]
            omniem = weight * omniem + (1.0 - weight) * omniem[order]
    return raw, target, mitonet, omniem, acquisition


def train_candidate(
    data: FullPatchData,
    config: dict,
    device: torch.device,
    checkpoint: Path,
) -> tuple[FullRefinementModel, list[dict]]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = candidate_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=5e-6
    )
    batch_size = int(config["batch_size"])
    history = []
    for epoch in range(int(config["epochs"])):
        indices = training_indices(data, config, seed + epoch)
        np.random.default_rng(seed + 1000 + epoch).shuffle(indices)
        running = {key: 0.0 for key in ("total", "region", "boundary", "distance", "correction_l1")}
        model.train()
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            raw, target, mitonet, omniem, acquisition = make_batch(
                data, chosen, config, device, training=True
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(raw, mitonet, omniem, acquisition)
            loss, parts = full_fidelity_loss(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running["total"] += float(loss.detach()) * len(chosen)
            for key, value in parts.items():
                running[key] += value * len(chosen)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{key: value / len(indices) for key, value in running.items()},
        }
        history.append(row)
        print(
            config["candidate_id"], f"epoch {epoch + 1}/{config['epochs']}",
            f"loss={row['total']:.5f}", flush=True,
        )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "mitocensus.refinement-stage-c-model.v1",
            "config": config,
            "history": history,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    return model.eval(), history


def predict_validation(
    model: FullRefinementModel,
    data: FullPatchData,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    validation = np.flatnonzero(data.split == "validation")
    probabilities, gates = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation), int(config["validation_batch_size"])):
            chosen = validation[start : start + int(config["validation_batch_size"])]
            raw, _, mitonet, omniem, acquisition = make_batch(
                data, chosen, config, device, training=False
            )
            output = model(raw, mitonet, omniem, acquisition)
            probabilities.append(torch.sigmoid(output["region"])[:, 0].cpu().numpy())
            gates.append(output["prior_gate"][:, 0].cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32), np.concatenate(gates).astype(np.float32)


def load_checkpoint(path: Path, device: torch.device) -> tuple[FullRefinementModel, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = candidate_model(config)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), config
