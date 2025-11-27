"""Main entry point for ValueCell Server Backend."""

import io
import sys
from datetime import datetime

import uvicorn
from loguru import logger

from valuecell.server.api.app import create_app
from valuecell.server.config.settings import get_settings

# Set stdout encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def setup_logging() -> None:
    """Configure loguru to output logs to both console and file."""
    settings = get_settings()
    logs_dir = settings.LOGS_DIR

    # Create timestamp-based log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = logs_dir / timestamp
    log_subdir.mkdir(parents=True, exist_ok=True)

    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Add file handler for all logs
    log_file = log_subdir / "server.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # Add error file handler for errors and above
    error_log_file = log_subdir / "error.log"
    logger.add(
        error_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
        encoding="utf-8",
    )

    logger.info(f"Logging configured. Log files: {log_subdir}")


# Setup logging before creating app
setup_logging()

# Create app instance for uvicorn
app = create_app()


def main():
    """Start the server."""
    settings = get_settings()

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
    )


if __name__ == "__main__":
    main()
