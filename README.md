# ML Experiment Template

A comprehensive template for managing machine learning experiments with modern Python tooling and best practices.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Directory Setup](#directory-setup)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Running Inference](#running-inference)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This template provides a production-ready structure for machine learning experimentation. It includes integrated experiment tracking, structured logging, and a modular codebase designed to accelerate your ML workflow from research to deployment.

## Features

- 🚀 **Fast Dependency Management**: Lightning-fast package management with `uv`
- 📊 **Experiment Tracking**: Built-in [Weights & Biases](https://wandb.ai/) integration for metrics visualization
- 📝 **Structured Logging**: Comprehensive logging to both console and files
- ⚙️ **Type-Safe CLI**: Automatic CLI generation with [tyro](https://github.com/brentyi/tyro)
- 🔧 **Modular Architecture**: Clean separation of concerns for data, models, training, and inference
- 📦 **Reproducible Environments**: Locked dependencies for consistent results across machines

## Tech Stack

| Tool | Purpose |
|------|---------|
| [uv](https://github.com/astral-sh/uv) | Project & package management |
| [Python logging](https://docs.python.org/3/library/logging.html) | Structured logging to `.log` files |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking and visualization |
| [tyro](https://github.com/brentyi/tyro) | Type-safe command-line interfaces |

## Getting Started

### Installation

1. **Install uv**

   On Linux/macOS:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   On macOS (via Homebrew):
   ```bash
   brew install uv
   ```

   On Windows:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/miya-99999/ml-exp-template.git
   cd ml-exp-template
   ```

3. **Sync dependencies**

   ```bash
   uv sync --frozen
   ```

4. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate
   ```

### Directory Setup

Create the necessary directory structure for experiments:

```bash
uv run scripts/setup_directories.py
```

This creates the following directories:
- `datasets/`: Dataset storage
- `logs/`: Training logs and metrics
- `models/`: Saved model checkpoints
- `outputs/`: Inference results and visualizations
- `notebooks/`: Jupyter notebooks for exploratory data analysis

## Usage

### Training a Model

Run a training experiment:

```bash
uv run scripts/train.py
```

Training metrics and logs will be saved to `logs/` and synchronized with Weights & Biases.

### Running Inference

Execute inference on trained models:

```bash
uv run scripts/infer.py
```

Results will be saved to the `outputs/` directory.

## Project Structure

```
.
├── datasets/              # Dataset storage
├── logs/                  # Training logs (exp001, exp002, ...)
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for EDA
├── outputs/               # Inference results
├── scripts/               # Entry point scripts
│   ├── prepare_dataset.py # Dataset preparation
│   ├── setup_directories.py
│   ├── train.py          # Training pipeline
│   └── infer.py          # Inference pipeline
├── src/                   # Source code
│   ├── config.py         # Configuration and hyperparameters
│   ├── data/             # Data loading and preprocessing
│   ├── model/            # Model architectures
│   ├── training/         # Training logic
│   ├── inference/        # Inference logic
│   └── utils/            # Utility functions
├── wandb/                 # W&B run data
├── pyproject.toml         # Project dependencies and metadata
└── README.md
```

## Customization

To adapt this template for your specific use case, you can modify the following components (examples):

1. **Data Pipeline** (`scripts/prepare_dataset.py`, `src/data/`)
   - Implement custom data loading and preprocessing logic

2. **Configuration** (`src/config.py`)
   - Define hyperparameters, paths, and experiment settings

3. **Model Architecture** (`src/model/`)
   - Implement your neural network architectures

4. **Training Loop** (`src/training/`)
   - Customize the training procedure, loss functions, and optimization

5. **Inference** (`src/inference/`)
   - Define how models make predictions on new data

## Documentation

For a detailed explanation of this template's design and usage, see the accompanying Zenn article (coming soon).

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
