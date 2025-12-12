# src/data/splits.py

from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.config import CFG, DataCfg


def stratified_k_fold_split(
    labels: np.ndarray,
    cfg: CFG,
    data_cfg: DataCfg,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(
        n_splits=data_cfg.n_folds,
        shuffle=data_cfg.shuffle,
        random_state=cfg.common.seed,
    )

    indicies = np.arange(len(labels))
    for i, (tr_idx, val_idx) in enumerate(skf.split(indicies, labels)):
        if i == fold:
            return tr_idx, val_idx
    raise ValueError(f"Fold {fold} is out of range for {data_cfg.n_folds} folds.")


def random_k_fold_split(
    labels: np.ndarray,
    cfg: CFG,
    data_cfg: DataCfg,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    kf = KFold(
        n_splits=data_cfg.n_folds,
        shuffle=data_cfg.shuffle,
        random_state=cfg.common.seed,
    )

    indicies = np.arange(len(labels))
    for i, (tr_idx, val_idx) in enumerate(kf.split(indicies)):
        if i == fold:
            return tr_idx, val_idx
    raise ValueError(f"Fold {fold} is out of range for {data_cfg.n_folds} folds.")


def group_k_fold_split(
    labels: np.ndarray,
    groups: np.ndarray,
    cfg: CFG,
    data_cfg: DataCfg,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gkf = GroupKFold(n_splits=data_cfg.n_folds)
    indicies = np.arange(len(labels))

    for i, (tr_idx, val_idx) in enumerate(gkf.split(indicies, labels, groups)):
        if i == fold:
            return tr_idx, val_idx
    raise ValueError(f"Fold {fold} is out of range for {data_cfg.n_folds} folds.")


def get_fold_indicies(
    labels: np.ndarray,
    cfg: CFG,
    data_cfg: DataCfg,
    fold: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    split_type = data_cfg.split_type.lower()

    if split_type == "stratified":
        return stratified_k_fold_split(labels, cfg, data_cfg, fold)
    elif split_type == "random":
        return random_k_fold_split(labels, cfg, data_cfg, fold)
    elif split_type == "group":
        if groups is None:
            raise ValueError("Groups must be provided for group k-fold splitting.")
        return group_k_fold_split(labels, groups, cfg, data_cfg, fold)
    else:
        raise ValueError(f"Unknown split type: {data_cfg.split_type}")
