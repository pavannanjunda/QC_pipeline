"""
utils/logger.py
---------------
Structured, coloured logging via loguru.
"""

import sys
from loguru import logger

logger.remove()  # Remove default handler

logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

logger.add(
    "logs/pipeline.log",
    rotation="50 MB",
    retention="14 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
)

__all__ = ["logger"]
