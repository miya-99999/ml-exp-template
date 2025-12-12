# src/utils/wandb_utils.py

from typing import Optional

import wandb
from src.config import CFG
from wandb import Run

cfg = CFG()


def init_wandb(cfg: CFG, fold: int | None = None) -> Optional[Run]:
    if not cfg.wandb.use_wandb:
        return None

    run_name = cfg.common.exp_name
    if fold is not None:
        run_name = f"{run_name}_fold{fold}"

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=cfg.to_dict(),
    )
    return run
