# scripts/setup_directories.py
from pathlib import Path


def setup_directories():
    project_root = Path(__file__).parent.parent
    directories = ["datasets", "logs", "models", "notebooks", "outputs"]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)


if __name__ == "__main__":
    setup_directories()
