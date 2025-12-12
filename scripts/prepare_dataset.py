# scripts/prepare_dataset.py
import sys
from pathlib import Path

from torchvision import datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CFG


def main():
    cfg = CFG()
    data_root = cfg.paths.data_dir / "Plant_leave_diseases_dataset_without_augmentation" # https://data.mendeley.com/datasets/tywbtsjrjv/1

    ds = datasets.ImageFolder(root=data_root)

    print(f"Total dataset size: {len(ds)} images")
    print(f"Classes: {len(ds.classes)}")
    print(f"Class names: {ds.classes}")


if __name__ == "__main__":
    main()
