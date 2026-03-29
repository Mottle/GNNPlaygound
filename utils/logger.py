import logging
from datetime import datetime
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logger(
    name: str = "training",
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=False,
    )
    rich_handler.setLevel(getattr(logging, console_level.upper()))
    logger.addHandler(rich_handler)

    return logger


def get_logger(name: str = "training") -> logging.Logger:
    return logging.getLogger(name)
