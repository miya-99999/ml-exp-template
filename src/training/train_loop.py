# src/training/train_loop.py

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.config import CFG
from src.training.optimizer import build_optimizer, build_scheduler
from src.utils.logger import print_cuda_memory, timeit_log
from wandb import Run


def _get_device(cfg: CFG) -> torch.device:
    return torch.device(cfg.common.device)


@timeit_log
def train_one_epoch(
    cfg: CFG,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    logger,
    run: Optional[Run] = None,
    epoch: int = 0,
) -> Tuple[float, float]:
    device = _get_device(cfg)
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    logger.info(
        f"[Train] Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}"
    )

    return epoch_loss, epoch_acc


@timeit_log
def valid_one_epoch(
    cfg: CFG,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    logger,
    run: Optional[Run] = None,
    epoch: int = 0,
    return_predictions: bool = False,
):
    device = _get_device(cfg)
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += batch_size

            if return_predictions:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    logger.info(
        f"[Valid] Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}"
    )

    if return_predictions:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return epoch_loss, epoch_acc, all_preds, all_labels

    return epoch_loss, epoch_acc


def save_checkpoint(
    cfg: CFG,
    model: nn.Module,
    epoch: int,
    val_loss: float,
    is_best: bool,
    logger,
    fold: int | None = None,
) -> None:
    if not is_best:
        return

    model_dir = cfg.paths.model_dir / cfg.common.exp_name
    fold_dir = model_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "config": cfg.to_dict(),
    }

    best_path = fold_dir / "best_model.pth"
    torch.save(ckpt, best_path)
    logger.info(
        f"Saved best model (updated): {best_path} (epoch={epoch}, val_loss={val_loss:.4f})"
    )


def train_loop(
    cfg: CFG,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    logger,
    run: Optional[Run] = None,
    fold: int | None = None,
):
    device = _get_device(cfg)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_val_loss = float("inf")

    logger.info(
        f"Starting training: fold={fold}, "
        f"epochs={cfg.train.num_epochs}, "
        f"optimizer={cfg.train.optimizer}, "
        f"scheduler={cfg.train.scheduler}"
    )

    print_cuda_memory(tag="[Before training]")

    for epoch in range(1, cfg.train.num_epochs + 1):
        logger.info(f"\n----- Epoch {epoch}/{cfg.train.num_epochs} (fold {fold}) -----")

        train_loss, train_acc = train_one_epoch(
            cfg, model, train_loader, criterion, optimizer, logger, run, epoch
        )

        val_loss, val_acc = valid_one_epoch(
            cfg, model, valid_loader, criterion, logger, run, epoch
        )

        logger.info(
            f"[epoch={epoch}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if run is not None:
            lr = optimizer.param_groups[0]["lr"]
            run.log(
                {
                    "Train/Loss": train_loss,
                    "Train/Accuracy": train_acc,
                    "Valid/Loss": val_loss,
                    "Valid/Accuracy": val_acc,
                    "lr": lr,
                    "epoch": epoch,
                },
                step=epoch,
            )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            cfg=cfg,
            model=model,
            epoch=epoch,
            val_loss=val_loss,
            is_best=is_best,
            logger=logger,
            fold=fold,
        )

    print_cuda_memory(tag="[After training]")

    logger.info(f"Finished training fold={fold}. Best val_loss={best_val_loss:.4f}")

    # Load best model and get predictions
    model_dir = cfg.paths.model_dir / cfg.common.exp_name
    fold_dir = model_dir / f"fold_{fold}"
    best_path = fold_dir / "best_model.pth"

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, preds, labels = valid_one_epoch(
        cfg,
        model,
        valid_loader,
        criterion,
        logger,
        run=None,
        epoch=0,
        return_predictions=True,
    )

    return {
        "fold": fold,
        "predictions": preds,
        "labels": labels,
        "best_val_loss": best_val_loss,
    }
