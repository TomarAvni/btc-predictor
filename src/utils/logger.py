"""Application logging with file + console output."""

import logging
import sys
from pathlib import Path

import yaml

from src import CONFIG_PATH

_configured = False


def setup_logger(
    name: str = "btc_predictor",
    level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """Get or create a logger with console + optional file output.

    On first call, reads log settings from config/settings.yaml.
    Subsequent calls reuse the root configuration and just return
    a child logger for the given *name*.
    """
    global _configured

    logger = logging.getLogger(name)

    if not _configured:
        _configure_root(level, log_file)
        _configured = True

    return logger


def _configure_root(level: str | None = None, log_file: str | None = None) -> None:
    """One-time root logger setup."""
    cfg_level = "INFO"
    cfg_file = None
    cfg_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        log_cfg = config.get("logging", {})
        cfg_level = log_cfg.get("level", cfg_level)
        cfg_file = log_cfg.get("file")
        cfg_fmt = log_cfg.get("format", cfg_fmt)
    except Exception:
        pass

    resolved_level = level or cfg_level
    resolved_file = log_file or cfg_file

    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))

    formatter = logging.Formatter(cfg_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if not root.handlers:
        console = logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        )
        console.setFormatter(formatter)
        root.addHandler(console)

        if resolved_file:
            Path(resolved_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(resolved_file, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)
