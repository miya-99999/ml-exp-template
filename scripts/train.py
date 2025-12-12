#!/usr/bin/env python

# scripts/train.py
import sys
from pathlib import Path

import numpy as np
import tyro

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CFG
from src.data.dataset import create_dataloaders_for_fold
from src.model import create_model
from src.training.train_loop import train_loop
from src.utils.logger import init_logger, make_log_file
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb


def run_one_fold(cfg: CFG, fold: int):
    log_file = make_log_file(cfg)
    logger = init_logger(log_file, cfg.common.logger_name)
    logger.info(f"========== Start fold={fold} ==========")

    set_seed(cfg)

    run = init_wandb(cfg, fold=fold)

    train_loader, valid_loader, classes = create_dataloaders_for_fold(cfg, fold=fold)
    logger.info("----- Task info -----")
    logger.info(f"Number of classes: {len(classes)}")

    model = create_model(cfg)
    logger.info(f"Model: {cfg.model.model_name}")

    fold_results = train_loop(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        logger=logger,
        run=run,
        fold=fold,
    )

    logger.info(f"========== Finished fold={fold} ==========\n\n")
    if run is not None:
        run.finish()

    return fold_results


def main(cfg: CFG):
    # Create necessary directories
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.model_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for fold in range(cfg.data.n_folds):
        fold_results = run_one_fold(cfg, fold)
        all_results.append(fold_results)

    # Print summary after all folds
    log_file = make_log_file(cfg)
    logger = init_logger(log_file, cfg.common.logger_name)

    logger.info("\n" + "=" * 80)
    logger.info("----- Summary: All Folds Results -----")
    logger.info("=" * 80)

    # Combine all predictions and labels
    all_preds = np.concatenate([result["predictions"] for result in all_results])
    all_labels = np.concatenate([result["labels"] for result in all_results])

    # Calculate overall accuracy
    overall_accuracy = (all_preds == all_labels).sum() / len(all_labels)

    logger.info(f"\nTotal samples: {len(all_labels)}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Print per-fold summary table
    logger.info("\nPer-Fold Results:")
    for result in all_results:
        fold_preds = result["predictions"]
        fold_labels = result["labels"]
        fold_acc = (fold_preds == fold_labels).sum() / len(fold_labels)
        logger.info(
            f"  Fold {result['fold']}: Accuracy={fold_acc:.4f}, Val Loss={result['best_val_loss']:.4f}"
        )

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # cfg = tyro.cli(CFG)
    cfg = tyro.cli(CFG)
    main(cfg)
