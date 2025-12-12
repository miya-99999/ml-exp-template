# src/model/create_model.py

import torch
import timm

from src.config import CFG


def create_model(cfg: CFG) -> torch.nn.Module:
    model = timm.create_model(
        cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )
    return model
