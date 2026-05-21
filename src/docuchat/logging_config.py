"""Logging configuration for the application."""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with a consistent format.

    Call this once at application startup (in app.py lifespan).
    All modules then use logging.getLogger(__name__) and inherit this config.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR. Read from Settings.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
