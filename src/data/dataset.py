from pathlib import Path
from typing import Any, Callable, Optional, Tuple, cast

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from src.config import CFG, DataCfg
from src.data.splits import get_fold_indicies


# ==========================
# Albumentations augmentations
# ==========================
def get_transforms(data_cfg: DataCfg, train: bool = True) -> Callable:
    mean = data_cfg.img_mean
    std = data_cfg.img_std
    size = data_cfg.img_size

    if train:
        return A.Compose(
            [
                A.Resize(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


# ==========================
# Dataset
# ==========================
class CustomImageDataset(Dataset):
    def __init__(self, root: Path, transform: Optional[Callable] = None) -> None:
        self.dataset = datasets.ImageFolder(root=str(root), transform=None)
        self.transform = transform
        self.targets: list[int] = self.dataset.targets

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        image_np = np.array(image)
        if self.transform is not None:
            transformed: dict[str, Any] = self.transform(image=image_np)
            image_tensor = cast(torch.Tensor, transformed["image"])
        else:
            image_tensor = torch.from_numpy(image_np)
        return image_tensor, label


def create_datasets(cfg: CFG):
    data_cfg = cfg.data
    root: Path = cfg.paths.data_dir / data_cfg.data_folder

    train = CustomImageDataset(
        root=root,
        transform=get_transforms(data_cfg, train=True),
    )

    val = CustomImageDataset(
        root=root,
        transform=get_transforms(data_cfg, train=False),
    )

    labels = np.array(train.targets)
    assert len(train) == len(val)
    return train, val, labels, train.classes


# ==========================
# DataLoader
# ==========================
def create_dataloaders_for_fold(cfg: CFG, fold: int):
    train, valid, labels, classes = create_datasets(cfg)

    groups = None

    train_idx, valid_idx = get_fold_indicies(
        labels=labels,
        cfg=cfg,
        data_cfg=cfg.data,
        fold=fold,
        groups=groups,
    )

    train_dataset = Subset(train, train_idx.tolist())
    valid_dataset = Subset(valid, valid_idx.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.data.loader_num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.loader_num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, classes
