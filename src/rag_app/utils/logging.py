"""Logging configuration for the RAG application."""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = Path("logs/rag_app.log")
    log_file.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def setup_langfuse_logging():
    """Setup Langfuse logging if configured."""
    from ..config.settings import settings

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse

            langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )

            # Configure Langfuse logging
            # This is a simplified setup - in production you'd integrate more deeply
            logger = get_logger("langfuse")
            logger.info("Langfuse logging enabled")

        except ImportError:
            logger = get_logger(__name__)
            logger.warning("Langfuse package not available for logging")
    else:
        logger = get_logger(__name__)
        logger.info("Langfuse credentials not provided - logging disabled")