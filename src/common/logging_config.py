# Logging module

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = ROOT_DIR / "logs" / "project.log"
DEFAULT_LOG_PATH.mkdir(parents=True, exist_ok=True)

class ANSIColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        no_style = "\033[0m"
        bold = "\033[91m"
        cyan = "\u001b[34m"
        grey = "\033[90m"
        yellow = "\033[93m"
        red = "\033[31m"
        red_light = "\033[91m"
        start_style = {
            "DEBUG": grey,
            "INFO": cyan,
            "WARNING": yellow,
            "ERROR": red,
            "CRITICAL": red_light + bold,
        }.get(record.levelname, no_style)
        end_style = no_style
        return f"{start_style}{super().format(record)}{end_style}"

def configure_logging(
    level: str = "INFO",
    log_to_file: bool = False,
    file_path: str | Path = DEFAULT_LOG_PATH
) -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", level).upper(), logging.INFO)

    formatter = ANSIColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("project")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_to_file:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = RotatingFileHandler(file_path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"project.{name}")