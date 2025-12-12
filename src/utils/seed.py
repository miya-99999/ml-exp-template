# src/utils/seed.py

import logging
import os
import random

import numpy as np
import torch

from src.config import CFG
from src.utils.logger import LOGGER_NAME


def set_seed(cfg: CFG):
    logger = logging.getLogger(LOGGER_NAME)
    seed = cfg.common.seed

    logger.info("\n----- Seed info -----")
    logger.info(f"Set seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
