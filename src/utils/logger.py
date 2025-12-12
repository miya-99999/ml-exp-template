# src/utils/logger.py

import datetime
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import torch

from src.config import CFG

cfg = CFG()
LOGGER_NAME = cfg.common.logger_name


def make_log_file(cfg: CFG) -> Path:
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = cfg.paths.log_dir / cfg.common.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    return log_file


def init_logger(log_file: Path, logger_name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger  # Logger already initialized

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def timeit_log(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(cfg: CFG, *args, **kwargs) -> Any:
        logger = logging.getLogger(cfg.common.logger_name)
        device = cfg.common.device
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()

        try:
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()

                result = func(cfg, *args, **kwargs)

                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                logger.info(f"[{func.__name__}] {elapsed_ms / 1000:.6f} sec (CUDA)")
                return result
            else:
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available -> CPU fallback")

                start_time = time.perf_counter()
                result = func(cfg, *args, **kwargs)
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                logger.info(f"[{func.__name__}] {elapsed:.6f} sec (CPU)")
                return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper


def print_cuda_memory(tag: str = "") -> None:
    logger = logging.getLogger(LOGGER_NAME)

    if not torch.cuda.is_available():
        logger.info("Try to check CUDA memory but CUDA is not available.")
        return

    free, total = torch.cuda.mem_get_info()
    used = total - free
    logger.info(
        f"{tag}  Used: {used / 1024**2:.1f}MB / total: {total / 1024**2:.1f}MB "
        f"({used / total * 100:.2f}% used)"
    )
