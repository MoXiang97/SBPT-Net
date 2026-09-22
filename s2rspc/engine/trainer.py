"""Two-stage synthetic-data training for the paper-aligned SBPT-Net."""

from __future__ import annotations
import csv, json, random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from ..config import ExperimentConfig, TrainingConfig
from ..data.io import load_tokenized_cloud
from ..evaluation.metrics import binary_metrics, select_macro_threshold
from ..evaluation.projection import project_tokens_to_original_points
from ..models.sbptnet import ARCHITECTURE_ID, SBPTNet
from .checkpoint import save_checkpoint
from .features import (
    attribute_mean_std,
    pointmlp_feature_matrix,
    prepare_structural_features,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dice(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability, target = torch.sigmoid(logit), target.float()
    return 1.0 - (2.0 * (probability * target).sum() + 1.0) / (
        probability.sum() + target.sum() + 1.0
    )


def binary_segmentation_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    pos_weight = torch.as_tensor(
        float(positive_weight), dtype=logit.dtype, device=logit.device
    )
    return F.binary_cross_entropy_with_logits(
        logit, target.float(), pos_weight=pos_weight
    ) + float(dice_weight) * _dice(logit, target)


def structural_objective(
    final_logit: torch.Tensor,
    structural_logit: torch.Tensor,
    initial_logit: torch.Tensor,
    target: torch.Tensor,
    cfg: TrainingConfig,
    deviation_clip_bound: float,
) -> torch.Tensor:
    final_loss = binary_segmentation_loss(
        final_logit, target, cfg.dice_weight, cfg.structural_positive_weight
    )
    auxiliary = binary_segmentation_loss(
        structural_logit, target, cfg.dice_weight, cfg.structural_positive_weight
    )
    reference = initial_logit.detach().clamp(
        -float(deviation_clip_bound), float(deviation_clip_bound)
    )
    return (
        final_loss
        + cfg.structural_auxiliary_weight * auxiliary
        + cfg.deviation_weight * torch.mean((final_logit - reference) ** 2)
    )


def freeze_pointmlp(model: SBPTNet) -> None:
    model.pointmlp.eval()
    for parameter in model.pointmlp.parameters():
        parameter.requires_grad_(False)


def _indices(count: int, size: int, rng: np.random.Generator) -> np.ndarray:
    if count < 1:
        raise ValueError("A cloud must contain at least one token")
    return rng.choice(count, size=size, replace=count < size)


def _structural_indices(count: int, size: int, rng) -> np.ndarray:
    if count < 1:
        raise ValueError("A cloud must contain at least one token")
    return rng.choice(count, size=min(count, size), replace=False)


@torch.no_grad()
def pointmlp_logits_for_cloud(
    model: SBPTNet,
    cloud: Dict[str, np.ndarray],
    device: torch.device,
    record_seed: int | None = None,
) -> np.ndarray:
    features = pointmlp_feature_matrix(cloud)
    original_count = len(features)
    if original_count < 24:
        features = np.tile(features, (int(np.ceil(24 / max(original_count, 1))), 1))[
            :24
        ]
    model.pointmlp.eval()
    if record_seed is None and "_record_seed" in cloud:
        record_seed = int(np.asarray(cloud["_record_seed"]).item())
    cpu_rng_state = torch.get_rng_state() if record_seed is not None else None
    cuda_rng_state = (
        torch.cuda.get_rng_state_all()
        if record_seed is not None and torch.cuda.is_available()
        else None
    )
    try:
        if record_seed is not None:
            torch.manual_seed(int(record_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(record_seed))
        logits = []
        for start in range(0, len(features), 4096):
            tensor = torch.from_numpy(features[start : start + 4096].T[None]).to(device)
            logits.append(model.initial_logits(tensor)[0].cpu().numpy().reshape(-1))
        return np.concatenate(logits)[:original_count].astype(np.float32)
    finally:
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


@torch.no_grad()
def predict_cloud(
    model: SBPTNet,
    cloud: Dict[str, np.ndarray],
    attribute_mean: np.ndarray,
    attribute_std: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    initial = torch.from_numpy(pointmlp_logits_for_cloud(model, cloud, device)).to(
        device
    )
    features = torch.from_numpy(
        prepare_structural_features(cloud, attribute_mean, attribute_std)
    ).to(device)
    model.structural_encoder.eval()
    structural = model.structural_encoder(features)
    final = model.fuse(initial, structural)
    return {
        "initial_logit": initial.cpu().numpy(),
        "structural_logit": structural.cpu().numpy(),
        "final_logit": final.cpu().numpy(),
        "probability": torch.sigmoid(final).cpu().numpy(),
    }


def _miou(model, clouds, mean, std, threshold, device) -> float:
    values = []
    for cloud in clouds:
        probability = predict_cloud(model, cloud, mean, std, device)["probability"]
        if "raw_label" in cloud:
            prediction = project_tokens_to_original_points(
                cloud, probability, threshold
            )["full_prediction"]
            target = cloud["raw_label"]
        else:
            prediction, target = probability >= threshold, cloud["label"]
        values.append(float(binary_metrics(target, prediction)["iou"]))
    return float(np.mean(values)) if values else 0.0


def _pointmlp_loss(class_logits, target, cfg) -> torch.Tensor:
    weight = torch.tensor(
        [1.0, cfg.pointmlp_positive_weight],
        dtype=class_logits.dtype,
        device=class_logits.device,
    )
    foreground = class_logits[:, 1] - class_logits[:, 0]
    return F.cross_entropy(
        class_logits, target.long(), weight=weight
    ) + cfg.dice_weight * _dice(foreground, target)


def _pointmlp_stage(model, train_clouds, validation_clouds, cfg, device, history):
    optimizer = torch.optim.AdamW(
        model.pointmlp.parameters(),
        lr=cfg.training.pointmlp_learning_rate,
        weight_decay=cfg.training.pointmlp_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.training.maximum_epochs)
    )
    best, best_epoch, best_score, stale = None, 0, -1.0, 0
    for epoch in range(cfg.training.maximum_epochs):
        model.pointmlp.train()
        rng = np.random.default_rng(cfg.seed + 10007 * (epoch + 1))
        order, losses = rng.permutation(len(train_clouds)), []
        for start in range(0, len(order), cfg.training.pointmlp_batch_size):
            feature_batch, label_batch = [], []
            for index in order[start : start + cfg.training.pointmlp_batch_size]:
                cloud = train_clouds[int(index)]
                chosen = _indices(
                    len(cloud["label"]), cfg.model.pointmlp_sample_points, rng
                )
                feature_batch.append(pointmlp_feature_matrix(cloud)[chosen].T)
                label_batch.append(np.asarray(cloud["label"], dtype=np.int64)[chosen])
            features = torch.from_numpy(np.stack(feature_batch)).to(device)
            labels = torch.from_numpy(np.stack(label_batch)).to(device)
            output = model.pointmlp(features)
            loss = _pointmlp_loss(
                output[0] if isinstance(output, tuple) else output, labels, cfg.training
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.pointmlp.parameters(), cfg.training.gradient_clip
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        score = _miou(
            model,
            validation_clouds,
            np.zeros(6, np.float32),
            np.ones(6, np.float32),
            cfg.training.checkpoint_selection_threshold,
            device,
        )
        history.append(
            {
                "stage": "pointmlp",
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "synthetic_validation_miou_at_0_5": score,
            }
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best = {
                k: v.detach().cpu().clone()
                for k, v in model.pointmlp.state_dict().items()
            }
        else:
            stale += 1
        if stale >= cfg.training.early_stopping_patience:
            break
    if best is None:
        raise RuntimeError("PointMLP training did not produce a checkpoint")
    model.pointmlp.load_state_dict(best)
    return best_epoch, best_score


def _augment(values: np.ndarray, cfg: TrainingConfig, rng) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    descriptor_ids = np.asarray((4, 9, 11, 12, 14, 15), dtype=np.int64)
    descriptor_scale = rng.uniform(
        cfg.descriptor_scale_low, cfg.descriptor_scale_high, (1, 16)
    ).astype(np.float32)[:, descriptor_ids]
    descriptor_shift = rng.normal(0, cfg.descriptor_shift_std, (1, 16)).astype(
        np.float32
    )[:, descriptor_ids]
    descriptor_noise = rng.normal(
        0, cfg.descriptor_noise_std, (len(result), 16)
    ).astype(np.float32)[:, descriptor_ids]
    result[:, :6] = (
        result[:, :6] * descriptor_scale + descriptor_shift + descriptor_noise
    )
    attribute_scale = rng.uniform(
        cfg.attribute_scale_low, cfg.attribute_scale_high, (1, 6)
    ).astype(np.float32)
    attribute_shift = rng.normal(0, cfg.attribute_shift_std, (1, 6)).astype(
        np.float32
    )
    attribute_keep = (
        rng.random((1, 6)) >= cfg.attribute_drop_probability
    ).astype(np.float32)
    result[:, 6:] = (
        result[:, 6:] * attribute_scale
        + attribute_shift
        + rng.normal(0, cfg.attribute_noise_std, result[:, 6:].shape).astype(
            np.float32
        )
    ) * attribute_keep
    return result.astype(np.float32)

def _structural_stage(
    model, train_clouds, validation_clouds, mean, std, cfg, device, history
):
    freeze_pointmlp(model)
    initial = [
        pointmlp_logits_for_cloud(model, cloud, device) for cloud in train_clouds
    ]
    features = [prepare_structural_features(cloud, mean, std) for cloud in train_clouds]
    optimizer = torch.optim.AdamW(
        model.structural_encoder.parameters(),
        lr=cfg.training.structural_learning_rate,
        weight_decay=cfg.training.structural_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.training.maximum_epochs), eta_min=5e-5
    )
    best, best_epoch, best_score, stale = None, 0, -1.0, 0
    rng = np.random.RandomState(cfg.seed)
    for epoch in range(cfg.training.maximum_epochs):
        model.structural_encoder.train()
        order, losses = rng.permutation(len(train_clouds)), []
        for start in range(0, len(order), cfg.training.structural_batch_clouds):
            augmented_parts, initial_parts, target_parts = [], [], []
            batch_indices = [
                int(index)
                for index in order[
                    start : start + cfg.training.structural_batch_clouds
                ]
            ]
            selected = [
                _structural_indices(
                    len(train_clouds[index]["label"]),
                    cfg.training.structural_tokens_per_cloud,
                    rng,
                )
                for index in batch_indices
            ]
            for index, chosen in zip(batch_indices, selected):
                augmented_parts.append(
                    _augment(features[index][chosen], cfg.training, rng)
                )
                initial_parts.append(initial[index][chosen])
                target_parts.append(
                    np.asarray(train_clouds[index]["label"], np.float32)[chosen]
                )
            x = torch.from_numpy(np.concatenate(augmented_parts)).to(device)
            z0 = torch.from_numpy(np.concatenate(initial_parts)).to(device)
            y = torch.from_numpy(np.concatenate(target_parts)).to(device)
            zs = model.structural_encoder(x)
            loss = structural_objective(
                model.fuse(z0, zs),
                zs,
                z0,
                y,
                cfg.training,
                cfg.model.deviation_clip_bound,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.structural_encoder.parameters(), cfg.training.gradient_clip
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        score = _miou(
            model,
            validation_clouds,
            mean,
            std,
            cfg.training.checkpoint_selection_threshold,
            device,
        )
        history.append(
            {
                "stage": "structural",
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "synthetic_validation_miou_at_0_5": score,
            }
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best = {
                k: v.detach().cpu().clone()
                for k, v in model.structural_encoder.state_dict().items()
            }
        else:
            stale += 1
        if stale >= cfg.training.early_stopping_patience:
            break
    if best is None:
        raise RuntimeError("Structural training did not produce a checkpoint")
    model.structural_encoder.load_state_dict(best)
    return best_epoch, best_score


def train(
    train_paths: Iterable[str | Path],
    validation_paths: Iterable[str | Path],
    output_dir: str | Path,
    cfg: ExperimentConfig,
    device_name: str | None = None,
) -> Dict[str, object]:
    set_seed(cfg.seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_clouds = [load_tokenized_cloud(path) for path in train_paths]
    validation_clouds = [load_tokenized_cloud(path) for path in validation_paths]
    if not train_clouds or not validation_clouds:
        raise RuntimeError("Training and validation token files are both required")
    model, history = SBPTNet(cfg.model).to(device), []
    point_epoch, point_score = _pointmlp_stage(
        model, train_clouds, validation_clouds, cfg, device, history
    )
    mean, std = attribute_mean_std(train_clouds)
    structural_epoch, structural_score = _structural_stage(
        model, train_clouds, validation_clouds, mean, std, cfg, device, history
    )
    probabilities, targets = [], []
    for cloud in validation_clouds:
        probability = predict_cloud(model, cloud, mean, std, device)["probability"]
        if "raw_label" in cloud:
            probabilities.append(
                project_tokens_to_original_points(cloud, probability, 0.5)[
                    "full_probability"
                ]
            )
            targets.append(cloud["raw_label"])
        else:
            probabilities.append(probability)
            targets.append(cloud["label"])
    threshold, threshold_metrics = select_macro_threshold(
        probabilities, targets, cfg.evaluation.threshold_grid
    )
    metadata = {
        "seed": cfg.seed,
        "pointmlp_best_epoch": point_epoch,
        "structural_best_epoch": structural_epoch,
        "pointmlp_validation_miou_at_0_5": point_score,
        "structural_validation_miou_at_0_5": structural_score,
        "threshold": threshold,
        "threshold_selection": "synthetic validation mean per-cloud IoU",
        "real_data_role": "not loaded by this training command",
    }
    save_checkpoint(output_dir / "sbptnet_best.pt", model, mean, std, metadata)
    fields = list(history[0])
    with (output_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    manifest = {
        "architecture_id": ARCHITECTURE_ID,
        "training_protocol": "PointMLP followed by frozen-backbone structural encoding",
        "threshold": threshold,
        "synthetic_validation_threshold_metrics": threshold_metrics,
        "real_data_role": "not loaded by this training command",
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
