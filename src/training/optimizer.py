# src/training/optimizer.py

import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
)

from src.config import CFG


def build_optimizer(model: nn.Module, cfg: CFG) -> Optimizer:
    name = cfg.train.optimizer.lower()
    params = model.parameters()

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: Optimizer, cfg: CFG):
    name = cfg.train.scheduler.lower()

    if name == "none":
        return None

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.train.num_epochs),
            eta_min=cfg.train.eta_min,
        )

    if name == "warmup_cosine":
        warmup_epochs = max(1, cfg.train.warmup_epochs)
        total_epochs = max(1, cfg.train.num_epochs)

        base_lr = cfg.train.lr
        min_factor = cfg.train.eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            progress = float(epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return min_factor + (1.0 - min_factor) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler: {name}")
