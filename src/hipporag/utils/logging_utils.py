import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specific name."""
    return logging.getLogger(name)
