# src/config.py
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsCfg:
    """Paths configuration for the project directories."""

    data_dir: Path = PROJECT_ROOT / "datasets"
    """Directory containing the dataset."""
    model_dir: Path = PROJECT_ROOT / "models"
    """Directory for pretrained and trained models."""
    log_dir: Path = PROJECT_ROOT / "logs"
    """Directory for storing logs."""
    output_dir: Path = PROJECT_ROOT / "outputs"
    """Directory for storing output results."""


@dataclass
class CommonCfg:
    """Common configuration settings for the experiment."""

    exp_name: str = "exp001"
    """Name of the experiment."""
    seed: int = 42
    """Random seed for reproducibility."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for training (cuda or cpu)."""
    logger_name: str = "logger"
    """Name of the logger."""


@dataclass
class DataCfg:
    """Data loading and preprocessing configuration."""

    data_folder: str = "Plant_leave_diseases_dataset_without_augmentation"
    """Name of the dataset folder."""

    # cv
    split_type: Literal["stratified", "random", "group"] = "stratified"
    """Type of cross-validation split: 'stratified', 'random', or 'group'."""
    group_col: str | None = None
    """Column name for group split (used when split_type='group')."""
    n_folds: int = 5
    """Number of folds for cross-validation."""
    shuffle: bool = True
    """Whether to shuffle the data before splitting."""

    # image
    img_size: int = 224
    """Input image size (height and width)."""
    img_mean: tuple = (0.485, 0.456, 0.406)
    """Mean values for image normalization (RGB channels)."""
    img_std: tuple = (0.229, 0.224, 0.225)
    """Standard deviation values for image normalization (RGB channels)."""
    loader_num_workers: int = 4
    """Number of worker processes for data loading."""


@dataclass
class ModelCfg:
    """Model architecture configuration."""

    model_name: str = "resnet18d"
    """Name of the model architecture."""
    pretrained: bool = True
    """Whether to use pretrained weights."""
    num_classes: int = 39
    """Number of output classes for classification."""


@dataclass
class WandbCfg:
    """Weights & Biases logging configuration."""

    use_wandb: bool = True
    """Whether to use Weights & Biases for logging."""
    project: str = "sample"
    """W&B project name."""
    entity: str | None = "username"
    """W&B entity (username or team name)."""
    mode: Literal["online", "offline", "disabled", "shared"] = "online"
    """W&B logging mode: 'online', 'offline', 'disabled', or 'shared'."""


@dataclass
class TrainingCfg:
    """Training hyperparameters configuration."""

    batch_size: int = 32
    """Batch size for training."""
    shuffle: bool = True
    """Whether to shuffle the training data."""
    num_epochs: int = 3
    """Number of training epochs."""
    lr: float = 1e-3
    """Learning rate (alias for learning_rate)."""

    # optimizer
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    """Optimizer type: 'adam', 'adamw', or 'sgd'."""
    weight_decay: float = 1e-4
    """Weight decay (L2 regularization) for the optimizer."""
    momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    # scheduler
    scheduler: Literal["cosine", "warmup_cosine"] = "warmup_cosine"
    """Learning rate scheduler type: 'cosine' or 'warmup_cosine'."""
    warmup_epochs: int = 1
    """Number of warmup epochs for the scheduler."""
    eta_min: float = 1e-6
    """Minimum learning rate for cosine annealing."""


@dataclass
class CFG:
    """Main configuration class that aggregates all configuration settings."""

    paths: PathsCfg = field(default_factory=PathsCfg)
    """Paths configuration."""
    common: CommonCfg = field(default_factory=CommonCfg)
    """Common configuration."""
    data: DataCfg = field(default_factory=DataCfg)
    """Data configuration."""
    model: ModelCfg = field(default_factory=ModelCfg)
    """Model configuration."""
    wandb: WandbCfg = field(default_factory=WandbCfg)
    """Weights & Biases configuration."""
    train: TrainingCfg = field(default_factory=TrainingCfg)
    """Training configuration."""

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return asdict(self)
