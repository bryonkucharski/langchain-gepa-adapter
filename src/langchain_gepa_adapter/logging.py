from __future__ import annotations

import logging

LOGGER_NAME = "langchain_gepa_adapter"


def enable_verbose_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Turn on verbose logging for the package.

    - `level=logging.INFO` (default): one line per evaluate batch + per proposal call.
    - `level=logging.DEBUG`: also includes per-example messages, model responses,
      and the full reflection prompt/response text.

    Returns the package logger so callers can attach extra handlers if desired.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s %(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def disable_verbose_logging() -> None:
    """Silence the package logger again."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.WARNING)
